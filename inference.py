#!/usr/bin/env python3
"""
Refactored inference script for RETSam multitask segmentation.
This is the main entry point for running inference on fundus images.

Output Structure:
- Each image creates its own folder: output_dir/image_name/
- quantitative_analysis.json: Comprehensive analysis results (saved by Analyzer)
- disease_classification.json: Disease grades and reasoning (saved by DiseaseClassifier)
- visualizations/: Segmentation overlays (saved by Visualizer)
- masks/: Binary masks for each task (saved by Visualizer)

Features:
- 5-task multitask segmentation
- Advanced quantitative analysis
- Disease classification with English reasoning
- Modular architecture with clear separation of concerns
"""

import os
import json
import torch
import argparse
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from inference_modules.config import InferenceConfig
from inference_modules.model_loader import ModelLoader
from inference_modules.image_processor import ImageProcessor
from inference_modules.predictor import Predictor
from inference_modules.visualizer import Visualizer
from inference_modules.analyzer import Analyzer
from inference_modules.disease_classifier import DiseaseClassifier
from inference_modules.utils import Logger, convert_numpy_types


class FundusInference:
    """
    Main inference class for fundus image segmentation and analysis.
    
    This class orchestrates the entire pipeline from model loading to result saving.
    Each processed image gets its own output folder with organized results.
    """
    
    def __init__(self, model_path: str, output_dir: str, device: str = 'cuda',
                 multitask: bool = True, output_channels: Optional[Tuple] = None,
                 has_coordinate_head: Optional[bool] = None,
                 num_coordinates: Optional[int] = None,
                 enable_disease_classification: bool = False,
                 disease_types: Optional[List[str]] = None,
                 enable_noise_filter: bool = True,
                 input_root: Optional[str] = None,
                 binary_masks_only: bool = False,
                 analysis_only: bool = False,
                 save_visualizations: bool = True,
                 quiet: bool = True):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to model checkpoint
            output_dir: Base directory for saving results (each image gets its own subfolder)
            device: Device to run inference on ('cuda' or 'cpu')
            multitask: Whether to use multitask model
            output_channels: Output channel configuration (default: (2,3,2,4,6))
            enable_disease_classification: Whether to enable disease classification
            disease_types: List of diseases to classify ['dr', 'glaucoma', 'amd', 'myopia', 'cataract'] (if None, classifies all)
            enable_noise_filter: Whether to filter small noise from predictions (default: True)
            input_root: Optional root to preserve folder structure in outputs
            binary_masks_only: If True, only save per-class binary masks (no visualizations/analysis)
            quiet: Suppress verbose Logger output; progress shown via tqdm when True
        
        Output Structure:
            output_dir/
            ├── image_name_1/
            │   ├── quantitative_analysis.json      # Detailed analysis results
            │   ├── disease_classification.json     # Disease grades (if enabled)
            │   ├── visualizations/                 # Overlays and combined views
            │   └── masks/                          # Binary masks for each task
        
        Note: Noise filtering removes lesions < 5 pixels, vessels < 10 pixels
        """
        self.config = InferenceConfig()
        self.device = device
        self.multitask = multitask
        self.output_dir = output_dir
        # Root directory of inputs to preserve subfolder structure in outputs
        self.input_root = Path(input_root).resolve() if input_root else None
        self.binary_masks_only = binary_masks_only
        self.analysis_only = analysis_only
        self.save_visualizations = save_visualizations
        self.quiet = quiet
        Logger.set_quiet(quiet)

        if binary_masks_only and enable_disease_classification:
            Logger.warning("Binary-mask-only mode disables disease classification.")
            enable_disease_classification = False
        self.enable_disease_classification = enable_disease_classification
        self.disease_types = disease_types if enable_disease_classification else None
        
        # Initialize components
        self.model_loader = ModelLoader(device)
        self.image_processor = ImageProcessor()
        self.predictor = Predictor(device, enable_noise_filter=enable_noise_filter)
        self.visualizer = Visualizer(self.config)
        self.analyzer = Analyzer(self.config)
        self.disease_classifier = DiseaseClassifier(self.config) if enable_disease_classification else None
        
        # Load model
        Logger.info(f"Loading model from {model_path}")
        self.model = self.model_loader.load(
            model_path, multitask, output_channels,
            has_coordinate_head=has_coordinate_head,
            num_coordinates=num_coordinates
        )
        self.output_channels = output_channels or self.config.default_output_channels
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log disease classification status
        if self.enable_disease_classification:
            diseases = self.disease_types or self.config.get_supported_diseases()
            Logger.info(f"Disease classification enabled for: {', '.join(diseases)}")
        
    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single fundus image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing metrics and analysis results
            
        Creates:
            - output_dir/image_name/masks/: Binary masks for each task  
            - output_dir/image_name/visualizations/: Combined and individual visualizations
            - output_dir/image_name/quantitative_analysis.json: Detailed metrics (if run_analysis=True)
            - output_dir/image_name/disease_classification.json: Disease classification (if enabled)
        """
        image_name = os.path.basename(image_path)
        if not self.quiet:
            Logger.progress(1, 1, image_name)
        
        # Create output directory for this image, preserving subfolder structure
        image_path_obj = Path(image_path).resolve()
        try:
            if self.input_root:
                rel_dir = image_path_obj.parent.relative_to(self.input_root)
            else:
                rel_dir = Path()
        except Exception:
            rel_dir = Path()
        image_output_dir = os.path.join(self.output_dir, str(rel_dir), Path(image_name).stem)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Load and preprocess image
        image_data = self.image_processor.load_and_preprocess(image_path)
        
        # Run inference with noise filtering
        predictions = self.predictor.predict(self.model, image_data['tensor'], 
                                           task_names=self.config.task_names)
        
        # Extract macula center from coordinate predictions if available
        macula_center = None
        if len(predictions) > len(self.config.task_names):
            # Check if we have coordinate predictions (extra tensor after segmentation tasks)
            coord_tensor = predictions[-1]  # Last tensor should be coordinates
            if torch.is_tensor(coord_tensor) and coord_tensor.numel() >= 2:
                # Extract first coordinate (assuming it's macula center)
                coord_np = coord_tensor.squeeze().cpu().numpy()
                if len(coord_np) >= 2:
                    # Convert normalized coordinates to original image coordinates
                    original_height, original_width = image_data['original_shape']
                    macula_y = coord_np[0] * original_height
                    macula_x = coord_np[1] * original_width
                    macula_center = (macula_y, macula_x)
        
        # Save prediction outputs (mode dependent)
        save_results = self._save_results(
            predictions, image_data, image_output_dir, image_path
        )
        
        # Run analysis if enabled
        if (hasattr(self, 'run_analysis') and self.run_analysis) and not self.binary_masks_only:
            analysis_results = self.analyzer.analyze_all_tasks(
                save_results['mask_paths'],
                image_path,
                image_data['original_shape'],
                output_dir=image_output_dir,  # Let analyzer handle saving
                macula_center=macula_center
            )
            save_results['analysis'] = analysis_results
            
            # Run disease classification if enabled
            if self.enable_disease_classification and self.disease_classifier:
                disease_results = self.disease_classifier.classify_from_analysis_results(
                    analysis_results, self.disease_types
                )
                save_results['disease_classification'] = disease_results
                
                # Save disease classification results separately
                disease_output_path = os.path.join(image_output_dir, 'disease_classification.json')
                self.disease_classifier.save_classification_results(
                    disease_results, disease_output_path
                )
                
                # Print classification results
                Logger.info(f"Disease classification for {image_name}:")
                self.disease_classifier.print_classification_results(disease_results)
            
        # Add metadata
        save_results['image_name'] = image_name
        save_results['image_path'] = image_path
        save_results['processing_params'] = {
            'device': self.device,
            'multitask': self.multitask,
            'output_channels': str(self.output_channels)
        }
        
        if not self.quiet:
            Logger.success(f"Successfully processed: {image_name}")

        # Cleanup masks/visualizations for analysis-only mode to leave only JSON
        if self.analysis_only:
            self._cleanup_analysis_only_outputs(image_output_dir, save_results)

        return save_results

    def _cleanup_analysis_only_outputs(self, image_output_dir: str, save_results: Dict) -> None:
        """Remove masks/visuals so only quantitative JSON remains (analysis-only mode)."""
        try:
            masks_dir = os.path.join(image_output_dir, 'masks')
            visuals_dir = os.path.join(image_output_dir, 'visualizations')
            combined_png = os.path.join(image_output_dir, 'combined_predictions.png')
            coord_png = os.path.join(visuals_dir, 'coordinates_overlay.png')
            for path in [combined_png, coord_png]:
                if os.path.isfile(path):
                    os.remove(path)
            for d in [masks_dir, visuals_dir]:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            # Clear mask_paths reference since files are removed
            save_results['mask_paths'] = {}
            save_results['visualizations'] = {'combined': None}
        except Exception as e:
            Logger.warning(f"Cleanup failed in analysis-only mode: {e}")
        
    def process_batch(self, image_paths: List[str]) -> Dict:
        """
        Process multiple fundus images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Summary dictionary with results for all images
        """
        processed_count = 0
        failed_count = 0
        all_metrics = []
        
        # Disease classification counters
        disease_stats = {}
        if self.enable_disease_classification:
            # Initialize counters for all supported diseases
            supported_diseases = self.disease_types or self.config.get_supported_diseases()
            for disease in supported_diseases:
                disease_stats[disease] = {
                    'grade_0': 0, 'grade_1': 0, 'grade_2': 0, 'grade_3': 0, 'grade_4': 0,
                    'total_abnormal': 0, 'total_images': 0
                }
        
        for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
            metrics = self.process_single_image(image_path)
            all_metrics.append(metrics)
            processed_count += 1
            
            # Collect disease classification statistics
            if self.enable_disease_classification and 'disease_classification' in metrics:
                disease_results = metrics['disease_classification']
                for disease_code, disease_info in disease_results.items():
                    if disease_code in disease_stats:
                        grade = disease_info.get('grade', 0)
                        disease_stats[disease_code][f'grade_{grade}'] += 1
                        disease_stats[disease_code]['total_images'] += 1
                        if grade > 0:
                            disease_stats[disease_code]['total_abnormal'] += 1
                
            # except Exception as e:
            #     Logger.error(f"Failed to process {os.path.basename(image_path)}: {str(e)}")
            #     failed_count += 1
            #     continue
                
        # Create simplified summary (aggregate only)
        summary = {
            'total_images': len(image_paths),
            'processed_successfully': processed_count,
            'failed': failed_count,
        }
        
        # Add per-disease positive counts if classification enabled
        if self.enable_disease_classification and disease_stats:
            disease_positive_counts = {
                disease_code: stats.get('total_abnormal', 0)
                for disease_code, stats in disease_stats.items()
            }
            summary['disease_positive_counts'] = disease_positive_counts
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'batch_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(summary), f, ensure_ascii=False, indent=2)
            
        if not self.quiet:
            Logger.info(f"\n📊 Batch processing completed:")
            Logger.info(f"   Total images: {len(image_paths)}")
            Logger.info(f"   Successfully processed: {processed_count}")
            Logger.info(f"   Failed: {failed_count}")
            Logger.info(f"   Results saved to: {self.output_dir}")
        
        # Print disease classification summary
        if self.enable_disease_classification and disease_stats and not self.quiet:
            self._print_disease_summary(disease_stats, processed_count)
        
        return summary
        
    def _print_disease_summary(self, disease_stats: Dict, total_processed: int):
        """
        Print disease classification summary to console.
        
        Args:
            disease_stats: Dictionary containing disease statistics
            total_processed: Total number of successfully processed images
        """
        Logger.info(f"\n🏥 Disease Classification Summary:")
        Logger.info(f"   Total images analyzed: {total_processed}")
        Logger.info("   " + "="*60)
        
        # Define disease names mapping
        disease_names = {
            'dr': 'Diabetic Retinopathy',
            'glaucoma': 'Glaucoma',
            'amd': 'Age-related Macular Degeneration',
            'myopia': 'Pathological Myopia',
            'cataract': 'Cataract'
        }
        
        # Define grade descriptions
        grade_descriptions = {
            'dr': ['Normal', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'],
            'glaucoma': ['Normal', 'Suspect', 'Mild', 'Moderate', 'Severe'],
            'amd': ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD'],
            'myopia': ['Normal', 'Tessellated', 'Diffuse atrophy', 'Patchy atrophy', 'Macular atrophy'],
            'cataract': ['Normal', 'Suspected cataract']
        }
        
        for disease_code, stats in disease_stats.items():
            if stats['total_images'] == 0:
                continue
                
            disease_name = disease_names.get(disease_code, disease_code.upper())
            abnormal_rate = (stats['total_abnormal'] / stats['total_images']) * 100
            
            Logger.info(f"\n   📋 {disease_name}:")
            Logger.info(f"      Images diagnosed: {stats['total_images']}")
            Logger.info(f"      Abnormal cases: {stats['total_abnormal']} ({abnormal_rate:.1f}%)")
            Logger.info(f"      Normal cases: {stats['grade_0']} ({(stats['grade_0']/stats['total_images']*100):.1f}%)")
            
            # Show breakdown by grade
            descriptions = grade_descriptions.get(disease_code, [f'Grade {i}' for i in range(5)])
            for grade in range(5):
                count = stats[f'grade_{grade}']
                if count > 0 and grade > 0:  # Skip grade 0 as it's already shown as "normal"
                    percentage = (count / stats['total_images']) * 100
                    grade_desc = descriptions[grade] if grade < len(descriptions) else f'Grade {grade}'
                    Logger.info(f"         - {grade_desc}: {count} ({percentage:.1f}%)")
        
        Logger.info("   " + "="*60)
        
    def classify_diseases_only(self, image_path: str, 
                             disease_types: Optional[List[str]] = None,
                             save_results: bool = True) -> Dict[str, Dict]:
        """
        Only perform disease classification without full inference pipeline.
        Requires existing analysis results in JSON format.
        
        Args:
            image_path: Path to image or existing JSON analysis file
            disease_types: List of diseases to classify
            save_results: Whether to save classification results
            
        Returns:
            Disease classification results
        """
        if not self.disease_classifier:
            self.disease_classifier = DiseaseClassifier(self.config)
            
        # Check if input is a JSON file with analysis results
        image_path = Path(image_path)
        if image_path.suffix.lower() == '.json':
            # Classify from existing JSON file
            Logger.info(f"Classifying diseases from existing analysis: {image_path}")
            results = self.disease_classifier.classify_from_json_file(
                image_path, disease_types or self.disease_types
            )
        else:
            # Need to run full pipeline first
            Logger.error("For image files, use process_single_image with analysis enabled")
            return {}
            
        # Print results
        self.disease_classifier.print_classification_results(results, show_details=True)
        
        # Save results if requested
        if save_results:
            output_path = image_path.parent / f"{image_path.stem}_disease_classification.json"
            self.disease_classifier.save_classification_results(results, output_path)
            
        return results
        
    def _save_results(self, predictions: List[torch.Tensor], image_data: Dict,
                      output_dir: str, original_path: str) -> Dict:
        """Save prediction results and create visualizations."""
        if self.binary_masks_only:
            mask_paths = {}
            for task_idx, pred in enumerate(predictions):
                if task_idx >= len(self.config.task_names):
                    break
                task_name = self.config.get_task_name(task_idx)
                class_paths = self.visualizer.save_binary_masks_per_class(
                    pred, task_name, output_dir, image_data, task_idx=task_idx
                )
                if class_paths:
                    mask_paths[task_name] = class_paths
            return {
                'mask_paths': mask_paths,
                'metrics': {},
                'visualizations': {}
            }

        mask_paths = {}
        metrics = {}
        legacy_lesion_tasks = set(self.config.lesion_task_names)
        save_masks_flag = not self.analysis_only and not self.binary_masks_only
        
        # Process each task
        lesion_task_masks = {}
        for task_idx, pred in enumerate(predictions):
            task_name = self.config.get_task_name(task_idx)
            if task_idx >= len(self.config.task_names):
                break
            # Derive mask on original canvas for downstream use
            mask_arr = self.visualizer._get_task_mask(
                pred, task_name, image_data['original_shape'], task_idx=task_idx
            )

            # Save masks to disk only when allowed; otherwise keep in-memory
            mask_path = None
            if save_masks_flag:
                mask_path = self.visualizer.save_task_masks(
                    pred, task_name, output_dir, image_data, task_idx=task_idx
                )
            else:
                # Keep in-memory for analysis
                mask_path = mask_arr
            
            # Collect lesion task masks for regrouping
            if task_name in self.config.lesion_task_names:
                lesion_task_masks[task_name] = mask_arr
            
            # Only add valid paths/masks to mask_paths (skip None and placeholders)
            if (
                mask_path is not None
                and (not isinstance(mask_path, str) or not mask_path.startswith('placeholder_'))
                and task_name not in legacy_lesion_tasks
            ):
                # For artery/vein analysis, pass in-memory masks to avoid I/O losses
                if task_name == 'artery_vein':
                    mask_paths[task_name] = mask_arr
                else:
                    mask_paths[task_name] = mask_path
            
            # Defer all quantitative metrics to dedicated analysis modules only.
            # Here we do not compute per-task metrics to avoid duplication/inconsistency.
            # Detailed metrics will be produced by analyze_all_tasks (if enabled).
            if task_name not in legacy_lesion_tasks:
                metrics[task_name] = {
                    'task': task_name,
                    'original_dimensions': {
                        'width': image_data['original_shape'][0],
                        'height': image_data['original_shape'][1]
                    }
                }
            
        # Save grouped lesion masks and add their paths/arrays to mask_paths
        if lesion_task_masks and self.config.get_grouped_lesion_names():
            grouped_paths = self.visualizer.save_grouped_lesion_masks(
                lesion_task_masks, output_dir, image_data,
                save_visuals=self.save_visualizations,
                save_masks=save_masks_flag
            )
            mask_paths.update(grouped_paths)
        
        combined_path = None
        if self.save_visualizations and not self.binary_masks_only:
            combined_path = self.visualizer.create_combined_visualization(
                predictions, original_path, output_dir, image_data, save_visuals=self.save_visualizations
            )
        
        return {
            'mask_paths': mask_paths,
            'metrics': metrics,
            'visualizations': {
                'combined': combined_path
            }
        }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run inference on fundus images using RETSam models'
    )
    
    # Required arguments
    parser.add_argument('--input_dir', required=True, 
                       help='Directory containing input images')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save results')
    parser.add_argument('--model_path', required=True,
                       help='Path to model checkpoint')
    
    # Optional arguments
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--multitask', action='store_true', default=True,
                       help='Use multitask model')
    parser.add_argument('--output_channels', type=str, default=None,
                       help='Output channel configuration, e.g., "(2,3,2,4,6)"')
    parser.add_argument('--has_coordinate_head', action='store_true',
                       help='Indicate the model includes a coordinate prediction head')
    parser.add_argument('--num_coordinates', type=int, default=None,
                       help='Number of coordinates (2 for one point, 4 for two points, etc.)')
    parser.add_argument('--tasks', type=str, default=None,
                       help='Comma-separated list of tasks to perform')
    parser.add_argument('--image_extensions', type=str, 
                       default='jpg,jpeg,png,bmp,tiff',
                       help='Supported image extensions')
    parser.add_argument('--enable_analysis', action='store_true',
                       help='Enable advanced analysis features')
    parser.add_argument('--classify_diseases', action='store_true',
                       help='Enable disease classification (requires --enable_analysis)')
    parser.add_argument('--disease_types', type=str, nargs='+',
                       choices=['dr', 'glaucoma', 'amd', 'myopia', 'cataract'],
                       help='Specific diseases to classify (default: all supported diseases)')
    parser.add_argument('--classification_only', type=str, metavar='JSON_FILE',
                       help='Only perform disease classification on existing analysis JSON file')
    parser.add_argument('--disable_noise_filter', action='store_true',
                       help='Disable noise filtering for predictions (default: enabled)')
    parser.add_argument('--binary_masks_only', action='store_true',
                       help='Only save per-class 0/255 binary masks; skip visualizations and analysis outputs')
    parser.add_argument('--analysis_only', action='store_true',
                       help='Only save quantitative analysis (no visualization overlays)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging instead of tqdm-only progress')
    
    args = parser.parse_args()
    Logger.set_quiet(not args.verbose)
    
    if args.binary_masks_only:
        if args.enable_analysis:
            Logger.warning("Binary-mask-only mode ignores quantitative analysis output.")
            args.enable_analysis = False
        if args.classify_diseases:
            Logger.warning("Binary-mask-only mode ignores disease classification.")
            args.classify_diseases = False
    if args.analysis_only:
        # Analysis-only implies we still need masks but skip visual overlays
        Logger.info("Analysis-only mode: visualizations will not be saved.")
        if args.binary_masks_only:
            Logger.warning("analysis_only overrides visualization saving; binary_masks_only already skips visuals.")

    # Handle classification-only mode
    if args.classification_only:
        # Create a minimal inference object for classification only
        from inference_modules.config import InferenceConfig
        from inference_modules.disease_classifier import DiseaseClassifier
        config = InferenceConfig()
        classifier = DiseaseClassifier(config)
        
        Logger.info("Disease classification mode (analysis file input)")
        results = classifier.classify_from_json_file(
            args.classification_only, 
            args.disease_types
        )
        
        classifier.print_classification_results(results, show_details=True)
        
        # Save results
        output_path = Path(args.output_dir) / 'disease_classification_results.json'
        os.makedirs(args.output_dir, exist_ok=True)
        classifier.save_classification_results(results, output_path)
        return
    
    # Validate disease classification requirements
    if args.classify_diseases and not args.enable_analysis:
        Logger.error("Disease classification requires --enable_analysis to be enabled")
        return
    
    # Parse output channels
    if args.output_channels:
        import ast
        output_channels = ast.literal_eval(args.output_channels)
    else:
        output_channels = None
        
    # Find image files recursively
    extensions = [e.strip() for e in args.image_extensions.split(',') if e.strip()]
    image_files = []
    input_root_path = Path(args.input_dir).resolve()
    for ext in extensions:
        image_files.extend(list(input_root_path.rglob(f'*.{ext}')))
        image_files.extend(list(input_root_path.rglob(f'*.{ext.upper()}')))
        
    if not image_files:
        Logger.error(f"No image files found in {args.input_dir}")
        return
        
    Logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize inference pipeline
    inference = FundusInference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        multitask=args.multitask,
        output_channels=output_channels,
        has_coordinate_head=args.has_coordinate_head,
        num_coordinates=args.num_coordinates,
        enable_disease_classification=args.classify_diseases,
        disease_types=args.disease_types,
        enable_noise_filter=not args.disable_noise_filter,
        input_root=args.input_dir,
        binary_masks_only=args.binary_masks_only,
        analysis_only=args.analysis_only,
        save_visualizations=not args.analysis_only and not args.binary_masks_only,
        quiet=not args.verbose
    )
    
    # Enable analysis if requested
    if args.enable_analysis and not args.binary_masks_only:
        inference.run_analysis = True
        
    # Process images
    inference.process_batch([str(f) for f in image_files])


if __name__ == "__main__":
    main()
