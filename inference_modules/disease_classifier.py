"""
Disease classification module for fundus image analysis.
Based on quantitative metrics from segmentation and analysis results.
"""

import json
import os
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

from .config import InferenceConfig
from .utils import Logger, convert_numpy_types

# Import disease classification functions
try:
    from inference_modules.disease_cls import (
        grade_dr_unsupervised,
        grade_glaucoma_unsupervised, 
        grade_amd_unsupervised,
        grade_myopia_unsupervised,
        grade_cataract_unsupervised
    )
except ImportError:
    from disease_cls import (
        grade_dr_unsupervised,
        grade_glaucoma_unsupervised,
        grade_amd_unsupervised,
        grade_myopia_unsupervised,
        grade_cataract_unsupervised
    )


class DiseaseClassifier:
    """Handles disease classification based on quantitative analysis results."""
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize disease classifier.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        
        # Mapping from disease codes to classification functions
        self.classification_functions = {
            'dr': grade_dr_unsupervised,
            'glaucoma': grade_glaucoma_unsupervised,
            'amd': grade_amd_unsupervised,
            'myopia': grade_myopia_unsupervised,
            'cataract': grade_cataract_unsupervised
        }
        
    def classify_diseases(self, metrics: Dict, disease_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Classify diseases based on quantitative metrics.
        
        Args:
            metrics: Dictionary containing quantitative analysis metrics
            disease_types: List of diseases to classify. If None, classifies all supported diseases
            
        Returns:
            Dictionary containing classification results for each disease
        """
        if disease_types is None:
            disease_types = self.config.get_supported_diseases()
            
        results = {}
        
        for disease in disease_types:
            if disease not in self.config.get_supported_diseases():
                Logger.warning(f"Unsupported disease type: {disease}")
                continue
                
            try:
                # Pass thresholds from config to grading functions if they accept it
                thresholds = self.config.get_disease_thresholds(disease)
                func = self.classification_functions[disease]
                try:
                    grade, reason = func(metrics, thresholds)
                except TypeError:
                    grade, reason = func(metrics)
                
                results[disease] = {
                    'disease_code': disease,
                    'disease_name': self.config.get_disease_name(disease),
                    'grade': grade,
                    'grade_description': self.config.get_disease_grade_description(disease, grade),
                    'classification_reason': reason,
                    'severity': self._assess_severity(disease, grade)
                }
                
            except Exception as e:
                Logger.error(f"Classification failed for {disease}: {str(e)}")
                results[disease] = {
                    'disease_code': disease,
                    'disease_name': self.config.get_disease_name(disease),
                    'error': str(e)
                }
                
        return results
        
    def classify_from_analysis_results(self, analysis_results: Dict, 
                                     disease_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Classify diseases from complete analysis results.
        
        Args:
            analysis_results: Complete analysis results from inference
            disease_types: List of diseases to classify
            
        Returns:
            Disease classification results
        """
        # Extract metrics from analysis results
        metrics = self._extract_metrics_from_analysis(analysis_results)
        
        # Classify diseases
        return self.classify_diseases(metrics, disease_types)
        
    def classify_from_json_file(self, json_path: Union[str, Path], 
                              disease_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Classify diseases from a JSON file containing analysis results.
        
        Args:
            json_path: Path to JSON file with analysis results
            disease_types: List of diseases to classify
            
        Returns:
            Disease classification results
        """
        json_path = Path(json_path)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract metrics from JSON data
            if isinstance(data, dict):
                if 'metrics' in data:
                    metrics = data['metrics']
                elif 'analysis_results' in data and isinstance(data['analysis_results'], dict):
                    metrics = self._extract_metrics_from_analysis(data['analysis_results'])
                else:
                    # Heuristic: if keys look like task names, treat as analysis_results
                    possible_task_keys = {'artery_vein', 'od_oc', 'lesion_s1', 'lesion_s2', 'lesion_s3', 'possible_lesions', 'tessellation', 'myopia'}
                    if any(k in data for k in possible_task_keys):
                        metrics = self._extract_metrics_from_analysis(data)
                    else:
                        # Assume already-flat metrics
                        metrics = data
            else:
                metrics = {}
                
            return self.classify_diseases(metrics, disease_types)
            
        except Exception as e:
            Logger.error(f"Failed to classify from JSON file {json_path}: {str(e)}")
            return {}
            
    def save_classification_results(self, results: Dict[str, Dict], 
                                  output_path: Union[str, Path],
                                  include_summary: bool = True) -> None:
        """
        Save classification results to JSON file.
        
        Args:
            results: Classification results
            output_path: Output file path
            include_summary: Whether to include summary statistics
        """
        output_data = {
            'classification_results': results
        }
        
        if include_summary:
            output_data['summary'] = self._create_classification_summary(results)
            
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(output_data), f, ensure_ascii=False, indent=2)
            
        Logger.success(f"Classification results saved to {output_path}")
        
    def _extract_metrics_from_analysis(self, analysis_results: Dict) -> Dict:
        """Extract metrics from our quantitative_analysis.json-like structure.
        Creates a flattened metrics dict that our grading functions expect.
        """
        metrics: Dict[str, float] = {}

        for task_name, task_data in analysis_results.items():
            if not isinstance(task_data, dict):
                continue

            # 1) OD/OC mapping
            if task_name == 'od_oc':
                cdr = task_data.get('cd_ratio')
                if isinstance(cdr, (int, float)):
                    metrics['od_oc_cup_to_disc_ratio'] = float(cdr)
                isnt_list = task_data.get('isnt_list')
                if isinstance(isnt_list, list) and len(isnt_list) == 4:
                    metrics['od_oc_rim_thickness_inferior'] = float(isnt_list[0])
                    metrics['od_oc_rim_thickness_superior'] = float(isnt_list[1])
                    metrics['od_oc_rim_thickness_nasal'] = float(isnt_list[2])
                    metrics['od_oc_rim_thickness_temporal'] = float(isnt_list[3])
                continue

            # 2) Lesion S1 mapping (per-category structure)
            if task_name == 'lesion_s1':
                summary = task_data.get('summary', {})
                metrics['lesion_s1_total_lesion_area'] = float(summary.get('total_area_pixels', 0))

                # Aggregate counts and quadrant stats from detailed_analysis
                detail = task_data.get('lesion_categories_analysis', {}).get('detailed_analysis', {})
                total_count = 0
                small = 0
                medium = 0
                large = 0
                quadrants = {'superior_temporal': 0, 'superior_nasal': 0, 'inferior_temporal': 0, 'inferior_nasal': 0}
                he_quadrant_counts = {'superior_temporal': 0, 'superior_nasal': 0, 'inferior_temporal': 0, 'inferior_nasal': 0}
                for category_name, cat in detail.items():
                    meas = cat.get('measurements', {})
                    total_count += int(meas.get('count', 0))
                    size_dist = cat.get('size_distribution', {})
                    small += int(size_dist.get('small_lesions', 0))
                    medium += int(size_dist.get('medium_lesions', 0))
                    large += int(size_dist.get('large_lesions', 0))
                    qd = cat.get('quadrant_distribution', {})
                    qcounts = qd.get('counts', {})
                    for k in quadrants.keys():
                        quadrants[k] += int(qcounts.get(k, 0))

                    # Per-category metrics for DR PDR rule
                    metrics[f"lesion_s1_{category_name}_count"] = int(meas.get('count', 0))
                    metrics[f"lesion_s1_{category_name}_total_area_pixels"] = float(meas.get('total_area_pixels', 0))
                    metrics[f"lesion_s1_{category_name}_coverage_percentage"] = float(meas.get('coverage_percentage', 0.0))
                    # Per-category size breakdown (hemorrhage-specific small/medium/large)
                    if category_name == 'hemorrhage':
                        metrics['lesion_s1_hemorrhage_small_count'] = int(size_dist.get('small_lesions', 0))
                        metrics['lesion_s1_hemorrhage_medium_count'] = int(size_dist.get('medium_lesions', 0))
                        metrics['lesion_s1_hemorrhage_large_count'] = int(size_dist.get('large_lesions', 0))

                    # Track hemorrhage quadrant-specific counts if available
                    if category_name == 'hemorrhage':
                        for k in he_quadrant_counts.keys():
                            he_quadrant_counts[k] = int(qcounts.get(k, 0))
                        metrics['lesion_s1_hemorrhage_quadrants_with_lesions'] = int(qd.get('quadrants_with_lesions', 0))
                        metrics['lesion_s1_hemorrhage_qc_superior_temporal'] = he_quadrant_counts['superior_temporal']
                        metrics['lesion_s1_hemorrhage_qc_superior_nasal'] = he_quadrant_counts['superior_nasal']
                        metrics['lesion_s1_hemorrhage_qc_inferior_temporal'] = he_quadrant_counts['inferior_temporal']
                        metrics['lesion_s1_hemorrhage_qc_inferior_nasal'] = he_quadrant_counts['inferior_nasal']

                quadrants_with_lesions = sum(1 for v in quadrants.values() if v > 0)
                metrics['lesion_s1_lesion_count'] = total_count
                metrics['lesion_s1_small_lesion_count'] = small
                metrics['lesion_s1_medium_lesion_count'] = medium
                metrics['lesion_s1_large_lesion_count'] = large
                metrics['lesion_s1_quadrants_with_lesions'] = quadrants_with_lesions
                continue

            # 2b) Lesion S3 mapping (per-category if available; else fallback later)
            if task_name == 'lesion_s3':
                # Prefer per-category structure
                detail_container = task_data.get('lesion_categories_analysis', {})
                detail = detail_container.get('detailed_analysis', {}) if isinstance(detail_container, dict) else {}
                if isinstance(detail, dict) and detail:
                    for category_name, cat in detail.items():
                        meas = cat.get('measurements', {})
                        metrics[f"lesion_s3_{category_name}_count"] = int(meas.get('count', 0))
                        metrics[f"lesion_s3_{category_name}_total_area_pixels"] = float(meas.get('total_area_pixels', 0))
                        metrics[f"lesion_s3_{category_name}_coverage_percentage"] = float(meas.get('coverage_percentage', 0.0))
                    # do not fallback to flat mapping if per-category exists
                    continue

            # 3) Lesion S2/S3 mapping (flat metrics available)
            if task_name in ('lesion_s2', 'lesion_s3'):
                prefix = task_name
                for k in (
                    'lesion_count', 'total_lesion_area', 'small_lesion_count', 'medium_lesion_count',
                    'large_lesion_count', 'small_lesion_total_area', 'medium_lesion_total_area',
                    'large_lesion_total_area'
                ):
                    if k in task_data:
                        metrics[f'{prefix}_{k}'] = float(task_data[k]) if isinstance(task_data[k], (int, float)) else 0.0
                continue

            # 4) Tessellation mapping
            if task_name == 'tessellation':
                tess_area = float(task_data.get('total_tessellate_area', 0))
                tessellate_ratio = float(task_data.get('tessellate_ratio', 0))
                metrics['tessellation_s1_total_lesion_area'] = tess_area
                metrics['tessellation_coverage_ratio'] = tessellate_ratio
                continue

            # 5) Myopia mapping (arc lesion)
            if task_name == 'myopia':
                metrics['myopia_s1_lesion_count'] = int(task_data.get('count', 0))
                metrics['myopia_s1_total_lesion_area'] = float(task_data.get('total_area', 0))
                # large/within_2dd not available -> default 0
                continue

            # 6) Fallback: flatten other numeric fields with prefix
            for key, value in task_data.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}_{key}"] = float(value)

        return metrics
        
    def _assess_severity(self, disease: str, grade: int) -> str:
        """Assess severity based on disease and grade."""
        if disease == 'cataract':
            return 'Abnormal' if grade == 1 else 'Normal'
            
        if grade == 0:
            return 'Normal'
        elif grade == 1:
            return 'Minimal'
        elif grade == 2:
            return 'Mild'
        elif grade == 3:
            return 'Moderate'
        elif grade == 4:
            return 'Severe'
        else:
            return 'Unknown'
            
    def _create_classification_summary(self, results: Dict[str, Dict]) -> Dict:
        """Create summary of classification results."""
        summary = {
            'total_diseases_analyzed': len(results),
            'classification_success': 0,
            'classification_failed': 0,
            'abnormal_findings': [],
            'grade_distribution': {}
        }
        
        for disease, result in results.items():
            if 'error' in result:
                summary['classification_failed'] += 1
            else:
                summary['classification_success'] += 1
                grade = result.get('grade', 0)
                
                # Track grade distribution
                if disease not in summary['grade_distribution']:
                    summary['grade_distribution'][disease] = {}
                summary['grade_distribution'][disease][grade] = 1
                
                # Track abnormal findings
                if grade > 0:
                    summary['abnormal_findings'].append({
                        'disease': result.get('disease_name', disease),
                        'grade': grade,
                        'severity': result.get('severity', 'Unknown'),
                        'description': result.get('grade_description', '')
                    })
                    
        return summary
        
    def print_classification_results(self, results: Dict[str, Dict], 
                                   show_details: bool = False) -> None:
        """
        Print classification results in a formatted way.
        
        Args:
            results: Classification results
            show_details: Whether to show detailed reasons
        """
        Logger.info("Disease Classification Results:")
        Logger.info("=" * 50)
        
        for disease, result in results.items():
            if 'error' in result:
                Logger.error(f"{result.get('disease_name', disease)}: Classification failed - {result['error']}")
            else:
                disease_name = result.get('disease_name', disease)
                grade = result.get('grade', 0)
                description = result.get('grade_description', '')
                severity = result.get('severity', '')
                
                Logger.info(f"{disease_name}: Grade {grade} - {description} ({severity})")
                
                if show_details and 'classification_reason' in result:
                    Logger.info(f"  Details: {result['classification_reason']}")
                    
        # Print summary
        abnormal_count = sum(1 for r in results.values() 
                           if 'grade' in r and r['grade'] > 0)
        if abnormal_count > 0:
            Logger.warning(f"\nFound {abnormal_count} abnormal disease(s)")
        else:
            Logger.success("\nAll examinations are normal")


def classify_single_image_results(analysis_results: Dict, 
                                disease_types: Optional[List[str]] = None,
                                output_path: Optional[Union[str, Path]] = None,
                                show_details: bool = False) -> Dict[str, Dict]:
    """
    Convenience function to classify diseases from single image analysis results.
    
    Args:
        analysis_results: Analysis results from inference
        disease_types: List of diseases to classify
        output_path: Optional output file path
        show_details: Whether to show detailed classification reasons
        
    Returns:
        Disease classification results
    """
    config = InferenceConfig()
    classifier = DiseaseClassifier(config)
    
    # Classify diseases
    results = classifier.classify_from_analysis_results(analysis_results, disease_types)
    
    # Print results
    classifier.print_classification_results(results, show_details)
    
    # Save results if requested
    if output_path:
        classifier.save_classification_results(results, output_path)
        
    return results


def classify_from_json(json_path: Union[str, Path],
                      disease_types: Optional[List[str]] = None,
                      output_path: Optional[Union[str, Path]] = None,
                      show_details: bool = False) -> Dict[str, Dict]:
    """
    Convenience function to classify diseases from JSON file.
    
    Args:
        json_path: Path to JSON file with analysis results
        disease_types: List of diseases to classify
        output_path: Optional output file path
        show_details: Whether to show detailed classification reasons
        
    Returns:
        Disease classification results
    """
    config = InferenceConfig()
    classifier = DiseaseClassifier(config)
    
    # Classify diseases
    results = classifier.classify_from_json_file(json_path, disease_types)
    
    # Print results
    classifier.print_classification_results(results, show_details)
    
    # Save results if requested
    if output_path:
        classifier.save_classification_results(results, output_path)
        
    return results