import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Union


def grade_dr_unsupervised(lesion_metrics: Dict[str, Union[float, int]], thresholds: Dict[str, float] = None) -> Tuple[int, str]:
    """
    Unsupervised grading for diabetic retinopathy (DR) based on lesion metrics.

    Grading (proxy rules):
    - 0: No DR (no lesion_s1 signals)
    - 1: Only microaneurysms/small hemorrhages (<50 px)
    - 2: Moderate lesions
    - 3: Hemorrhages in all four quadrants with high counts
    - 4: Proliferative DR or severe surrogate

    Args:
        lesion_metrics: metrics related to lesion_s1 (and grouped lesions)

    Returns:
        Tuple[int, str]: (DR grade, rationale)
    """
    # Focus on hemorrhage/microaneurysm (H/Ma)
    he_count = int(lesion_metrics.get('lesion_dr_hemorrhage_count', 0))
    he_area = float(lesion_metrics.get('lesion_dr_hemorrhage_total_area_pixels', 0.0))
    he_cov = float(lesion_metrics.get('lesion_dr_hemorrhage_coverage_percentage', 0.0))
    # Size buckets fallback if needed
    small_lesion_count = int(lesion_metrics.get('lesion_dr_small_lesion_count', 0))
    medium_lesion_count = int(lesion_metrics.get('lesion_dr_medium_lesion_count', 0))
    large_lesion_count = int(lesion_metrics.get('lesion_dr_large_lesion_count', 0))
    
    # Quadrant coverage (prefer hemorrhage distribution)
    quadrants_with_lesions = int(lesion_metrics.get('lesion_dr_hemorrhage_quadrants_with_lesions', 
                                                    lesion_metrics.get('lesion_dr_quadrants_with_lesions', 0)))
    he_qc_st = int(lesion_metrics.get('lesion_dr_hemorrhage_qc_superior_temporal', 0))
    he_qc_sn = int(lesion_metrics.get('lesion_dr_hemorrhage_qc_superior_nasal', 0))
    he_qc_it = int(lesion_metrics.get('lesion_dr_hemorrhage_qc_inferior_temporal', 0))
    he_qc_in = int(lesion_metrics.get('lesion_dr_hemorrhage_qc_inferior_nasal', 0))
    
    # Large lesion coverage
    total_area = lesion_metrics.get('lesion_dr_total_lesion_area', 0)
    large_lesion_area = lesion_metrics.get('lesion_dr_large_lesion_total_area', 0)
    
    # DR grading rules (simplified)
    # Shortcut: CNV in lesion_s3 => PDR
    cnv_count = int(lesion_metrics.get('lesion_amd_cnv_count', 0))
    cnv_area = float(lesion_metrics.get('lesion_amd_cnv_total_area_pixels', 0.0))
    if cnv_count > 0 or cnv_area > 0:
        return 4, f"PDR: CNV detected in lesion_s3 (count={cnv_count}, area={cnv_area})"
    # PDR surrogate: hemorrhage + exudate + CWS present with high count/coverage
    ex_count = int(lesion_metrics.get('lesion_dr_exudate_count', 0))
    cws_count = int(lesion_metrics.get('lesion_dr_cotton_wool_spot_count', 0))
    ex_area = float(lesion_metrics.get('lesion_dr_exudate_total_area_pixels', 0.0))
    cws_area = float(lesion_metrics.get('lesion_dr_cotton_wool_spot_total_area_pixels', 0.0))
    ex_cov = float(lesion_metrics.get('lesion_dr_exudate_coverage_percentage', 0.0))
    cws_cov = float(lesion_metrics.get('lesion_dr_cotton_wool_spot_coverage_percentage', 0.0))
    tri_count = he_count + ex_count + cws_count
    tri_cov = he_cov + ex_cov + cws_cov
    thr = thresholds or {}
    PDR_AREA_THRESHOLD = float(thr.get('pdr_area_threshold', 0.03))
    PER_QUADRANT_HEAVY = int(thr.get('per_quadrant_heavy_threshold', 20))
    TRI_COUNT = int(thr.get('pdr_tri_count_threshold', 50))

    if he_count > 0 and ex_count > 0 and cws_count > 0 and (tri_count > TRI_COUNT or tri_cov > PDR_AREA_THRESHOLD):
        return 4, f"PDR: hemorrhage/exudate/CWS all present (counts: {he_count},{ex_count},{cws_count}; total>{tri_count}; cov>{tri_cov:.3f})"
    # 0: No DR (no hemorrhage/exudate/CWS)
    if he_count == 0 and ex_count == 0 and cws_count == 0:
        return 0, "No hemorrhages detected in lesion_s1"
    # 1: Mild NPDR proxy
    # Low threshold: H/Ma present, per-quadrant counts below heavy threshold
    he_small = int(lesion_metrics.get('lesion_dr_hemorrhage_small_count', 0))
    he_medium = int(lesion_metrics.get('lesion_dr_hemorrhage_medium_count', 0))
    he_large = int(lesion_metrics.get('lesion_dr_hemorrhage_large_count', 0))

    # Only small hemorrhages and no EX/CWS => mild
    if he_count > 0 and he_small == he_count and he_medium == 0 and he_large == 0 and ex_count == 0 and cws_count == 0:
        return 1, f"Only small hemorrhages (<50 px): {he_small}"

    if he_count > 0 and all(x < PER_QUADRANT_HEAVY for x in [he_qc_st, he_qc_sn, he_qc_it, he_qc_in]) and ex_count == 0 and cws_count == 0:
        if quadrants_with_lesions <= 1:
            return 1, f"Mild H/Ma in <=1 quadrant (counts per quadrant <20)"
    
    # 3: Severe NPDR proxy (H/Ma >=20 in all quadrants)
    if all(x >= PER_QUADRANT_HEAVY for x in [he_qc_st, he_qc_sn, he_qc_it, he_qc_in]):
        return 3, "Severe NPDR proxy: >=20 H/Ma in all four quadrants"

    # 2: Moderate NPDR proxy (not mild, not severe)
    if he_count > 0 and 1 <= quadrants_with_lesions <= 3:
        return 2, f"Hemorrhages present but not meeting severe threshold (quadrants={quadrants_with_lesions})"

    # Fallback
    return 0, "No DR criteria met"


def grade_glaucoma_unsupervised(od_oc_metrics: Dict[str, Union[float, int]], thresholds: Dict[str, float] = None) -> Tuple[int, str]:
    """
    Unsupervised glaucoma grading from disc/cup metrics.

    Criteria (C/D based proxies):
    - 0: Normal (C/D < 0.4)
    - 1: Suspect (0.4-0.6)
    - 2: Mild (0.6-0.7)
    - 3: Moderate (0.7-0.8)
    - 4: Severe (>0.8 or ISNT violation)

    Args:
        od_oc_metrics: metrics for od_oc analysis

    Returns:
        Tuple[int, str]: (glaucoma grade, rationale)
    """
    # Cup-to-disc ratio
    cdr = od_oc_metrics.get('od_oc_cup_to_disc_ratio', 0.0)
    
    # ISNT rule inputs
    isnt_inferior = od_oc_metrics.get('od_oc_rim_thickness_inferior', 0.0)
    isnt_superior = od_oc_metrics.get('od_oc_rim_thickness_superior', 0.0)
    isnt_nasal = od_oc_metrics.get('od_oc_rim_thickness_nasal', 0.0)
    isnt_temporal = od_oc_metrics.get('od_oc_rim_thickness_temporal', 0.0)
    
    # Check ISNT rule: I >= S >= N >= T
    isnt_violated = False
    if isnt_inferior > 0 and isnt_superior > 0 and isnt_nasal > 0 and isnt_temporal > 0:
        if not (isnt_inferior >= isnt_superior >= isnt_nasal >= isnt_temporal):
            isnt_violated = True
    
    # Glaucoma grading
    thr = thresholds or {}
    cdr_n = float(thr.get('cdr_normal_upper', 0.4))
    cdr_s = float(thr.get('cdr_suspect_upper', 0.6))
    cdr_mi = float(thr.get('cdr_mild_upper', 0.7))
    cdr_mo = float(thr.get('cdr_moderate_upper', 0.8))
    enforce_isnt = bool(thr.get('enforce_isnt_rule', True))

    if cdr < cdr_n:
        return 0, f"Cup-to-disc ratio {cdr:.2f} < 0.4, normal"
    elif cdr < cdr_s:
        if isnt_violated and enforce_isnt:
            return 2, f"Cup-to-disc ratio {cdr:.2f} with ISNT rule violation, mild glaucoma"
        return 1, f"Cup-to-disc ratio {cdr:.2f} between 0.4-0.6, glaucoma suspect"
    elif cdr < cdr_mi:
        return 2, f"Cup-to-disc ratio {cdr:.2f} between 0.6-0.7, mild glaucoma"
    elif cdr < cdr_mo:
        return 3, f"Cup-to-disc ratio {cdr:.2f} between 0.7-0.8, moderate glaucoma"
    else:
        return 4, f"Cup-to-disc ratio {cdr:.2f} > 0.8, severe glaucoma"


def grade_amd_unsupervised(lesion_metrics: Dict[str, Union[float, int]], thresholds: Dict[str, float] = None) -> Tuple[int, str]:
    """
    Unsupervised AMD grading based on lesion metrics.

    Proxy grading:
    - 0: Normal (no AMD-related findings)
    - 1: Presence of drusen (>=50) or notable area ratio
    - 2: Extensive drusen
    - 3: Reserved for patch hemorrhage (not implemented yet)

    Args:
        lesion_metrics: lesion-related metrics

    Returns:
        Tuple[int, str]: (AMD grade, rationale)
    """
    # Drusen info (lesion_s2)
    drusen_count = lesion_metrics.get('lesion_amd_drusen_count', 0)
    drusen_area = lesion_metrics.get('lesion_amd_drusen_total_area', 0)
    
    # Drusen size distribution
    small_drusen = lesion_metrics.get('lesion_amd_small_lesion_count', 0)
    medium_drusen = lesion_metrics.get('lesion_amd_medium_lesion_count', 0)
    large_drusen = lesion_metrics.get('lesion_amd_large_lesion_count', 0)
    
    # Drusen area ratio (assuming 640x640)
    total_fundus_area = 640 * 640
    drusen_area_ratio = drusen_area / total_fundus_area if total_fundus_area > 0 else 0
    
    # AMD grading (grade 3 reserved for future patch hemorrhage support)
    
    # Grade 2: Large amount of drusen
    # Criteria: ≥100 drusen OR ≥20 medium/large drusen OR area ratio >5%
    thr = thresholds or {}
    g2_cnt = int(thr.get('grade2_drusen_count', 100))
    g2_ml = int(thr.get('grade2_med_large_count', 20))
    g2_area = float(thr.get('grade2_area_ratio', 0.05))
    g1_cnt = int(thr.get('grade1_drusen_count', 50))
    g1_area = float(thr.get('grade1_area_ratio', 0.02))

    if drusen_count >= g2_cnt or (medium_drusen + large_drusen) >= g2_ml or drusen_area_ratio > g2_area:
        return 2, f"Extensive drusen (total: {drusen_count}, medium/large: {medium_drusen + large_drusen}, area: {drusen_area_ratio:.1%}), AMD grade 2"
    
    # Grade 1: Significant drusen presence
    # Criteria: ≥50 drusen OR area ratio >2%
    if drusen_count >= g1_cnt or drusen_area_ratio > g1_area:
        return 1, f"Significant drusen ({drusen_count} count, {drusen_area_ratio:.1%} area), AMD grade 1"
    
    # Grade 0: Normal (few drusen below thresholds)
    if drusen_count > 0:
        return 0, f"Few drusen ({drusen_count}) below grading threshold, normal"
    
    return 0, "No AMD-related findings"


def grade_cataract_unsupervised(metrics: Dict[str, Union[float, int]], thresholds: Dict[str, float] = None) -> Tuple[int, str]:
    """
    Heuristic cataract detection from multiple analysis signals.

    Criteria:
    - 0: No cataract
    - 1: Likely cataract (very low segmentation signals, poor image quality)

    Logic:
    - Very low vessel signal
    - Disc/cup segmentation missing or tiny
    - Lesions largely absent
    - When multiple signals are abnormally low, mark as cataract

    Args:
        metrics: aggregated analysis metrics

    Returns:
        Tuple[int, str]: (cataract flag, rationale)
    """
    # Compatible with newer metric names (no av_*; no disc/cup area)
    # 1) Vessel signal proxy: fg_rate and fractal dimensions
    fg_rate = float(metrics.get('artery_vein_fg_rate', 0.0))
    fda = float(metrics.get('artery_vein_FDa', 0.0))
    fdv = float(metrics.get('artery_vein_FDv', 0.0))
    thr = thresholds or {}
    vessel_signal = max(fg_rate, (fda + fdv) / 4.0)

    # 2) Disc/cup signal proxy: presence of CDR and ISNT thickness
    cdr = float(metrics.get('od_oc_cup_to_disc_ratio', 0.0))
    isnt_vals = [
        float(metrics.get('od_oc_rim_thickness_inferior', 0.0)),
        float(metrics.get('od_oc_rim_thickness_superior', 0.0)),
        float(metrics.get('od_oc_rim_thickness_nasal', 0.0)),
        float(metrics.get('od_oc_rim_thickness_temporal', 0.0)),
    ]
    disc_signal = 1.0 if (cdr > 0 or any(v > 0 for v in isnt_vals)) else 0.0

    # 3) Lesion/other structure signals
    lesion_total = int(metrics.get('lesion_dr_lesion_count', 0)) \
                 + int(metrics.get('lesion_amd_lesion_count', 0)) \
                 + int(metrics.get('lesion_others_lesion_count', 0))
    other_structures = int(metrics.get('tessellation_s1_lesion_count', 0)) \
                     + int(metrics.get('myopia_s1_lesion_count', 0))

    # 4) Evaluate proxy signals
    failed = 0
    reasons = []
    vessel_min = float(thr.get('vessel_signal_min', 0.001))
    if vessel_signal < vessel_min:
        failed += 1
        reasons.append(f"Very low vessel signal (fg_rate={fg_rate:.4f}, FDa={fda:.2f}, FDv={fdv:.2f})")
    if disc_signal == 0.0:  # missing disc/cup signal
        failed += 1
        reasons.append("No disc/cup signal (CDR/ISNT missing)")
    if lesion_total < 2:  # nearly no lesions
        failed += 1
        reasons.append(f"Minimal lesions detected (total={lesion_total})")
    if other_structures == 0:  # no tessellation/myopia detected
        failed += 1
        reasons.append("No tessellation/myopia detected")

    # 5) Decision: flag cataract only if enough signals fail
    min_fail = int(thr.get('min_failures_for_cataract', 3))
    if failed >= min_fail:
        return 1, f"Multiple detection failures ({'; '.join(reasons[:3])}...) suspected cataract"

    return 0, "Normal image quality, no cataract signs"


def grade_myopia_unsupervised(metrics: Dict[str, Union[float, int]], thresholds: Dict[str, float] = None) -> Tuple[int, str]:
    """
    Unsupervised pathological myopia grading using multiple metrics.

    Grading proxy:
    - 0: Normal (no myopic changes)
    - 1: Mild tessellation
    - 2: Diffuse atrophy
    - 3: Patchy atrophy
    - 4: Macular atrophy

    Args:
        metrics: tessellation and myopia-related metrics

    Returns:
        Tuple[int, str]: (myopia grade, rationale)
    """
    # Tessellation indicators
    tessellation_count = int(metrics.get('tessellation_s1_lesion_count', 0))
    tessellation_area = float(metrics.get('tessellation_s1_total_lesion_area', 0))
    tessellation_cov = float(metrics.get('tessellation_coverage_ratio', 0.0))
    
    # Myopia-related atrophy indicators (myopia_s1)
    myopia_lesion_count = metrics.get('myopia_s1_lesion_count', 0)
    myopia_lesion_area = metrics.get('myopia_s1_total_lesion_area', 0)
    myopia_large_lesions = metrics.get('myopia_s1_large_lesion_count', 0)
    
    # Macular atrophy within 2DD of disc center
    macular_atrophy = metrics.get('myopia_s1_within_2dd_optic_disc_count', 0)
    
    # Atrophy area ratio (assume 640x640 canvas)
    total_fundus_area = 640 * 640
    atrophy_ratio = myopia_lesion_area / total_fundus_area if total_fundus_area > 0 else 0
    
    # Myopia grading
    # Grade 4: Macular atrophy
    thr = thresholds or {}
    if macular_atrophy > 0 and myopia_lesion_count > 0:
        return 4, f"Macular atrophy present ({macular_atrophy} areas), pathological myopia grade 4"
    
    # Grade 3: Patchy atrophy (large atrophic areas)
    patchy_area = float(thr.get('patchy_area_threshold_px', 5000))
    if myopia_large_lesions > 0 or (myopia_lesion_count > 0 and myopia_lesion_area > patchy_area):
        return 3, f"Patchy atrophy ({myopia_large_lesions} large areas, total area: {myopia_lesion_area}), pathological myopia grade 3"
    
    # Grade 2: Diffuse atrophy (widespread small atrophic areas)
    diffuse_min_cnt = int(thr.get('diffuse_min_count', 3))
    diffuse_area_px = float(thr.get('diffuse_area_threshold_px', 1000))
    diffuse_ratio = float(thr.get('diffuse_ratio_threshold', 0.02))
    if myopia_lesion_count > diffuse_min_cnt or (myopia_lesion_area > diffuse_area_px and atrophy_ratio > diffuse_ratio):
        return 2, f"Diffuse atrophy ({myopia_lesion_count} areas, total area: {myopia_lesion_area}), pathological myopia grade 2"
    
    # Grade 1: Mild tessellation changes
    # Rule: coverage below threshold => normal
    tess_cov_g1 = float(thr.get('tessellation_coverage_grade1', 0.20))
    if tessellation_count > 0 and tessellation_cov >= tess_cov_g1:
        return 1, f"Mild tessellation (coverage {tessellation_cov:.0%}, {tessellation_count} areas)"
    
    # Grade 0: Normal
    return 0, "No myopia-related changes (tessellation coverage <20% or absent)"


def process_json_file(json_path: Union[str, Path], disease_types: List[str] = None) -> Dict[str, Union[int, str]]:
    """
    Process a single JSON analysis file and run disease grading.
    
    Args:
        json_path: path to analysis JSON
        disease_types: diseases to grade (default: all)
        
    Returns:
        dict with grading results
    """
    json_path = Path(json_path)
    
    if disease_types is None:
        disease_types = ['dr', 'glaucoma', 'amd', 'myopia', 'cataract']
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filename
    filename = json_path.stem
    
    # Extract metrics field
    if 'metrics' in data:
        metrics = data['metrics']
    elif 'analysis_results' in data:
        # Handle nested JSON format
        metrics = {}
        for task, task_data in data['analysis_results'].items():
            if isinstance(task_data, dict) and 'metrics' in task_data:
                for key, value in task_data['metrics'].items():
                    metrics[f"{task}_{key}"] = value
    else:
        # Fallback to raw data as metrics
        metrics = data
    
    result = {
        'filename': filename,
    }
    
    # Run grading
    if 'dr' in disease_types:
        dr_grade, dr_reason = grade_dr_unsupervised(metrics)
        result['dr_grade'] = dr_grade
        result['dr_reason'] = dr_reason
    
    if 'glaucoma' in disease_types:
        glaucoma_grade, glaucoma_reason = grade_glaucoma_unsupervised(metrics)
        result['glaucoma_grade'] = glaucoma_grade
        result['glaucoma_reason'] = glaucoma_reason
    
    if 'amd' in disease_types:
        amd_grade, amd_reason = grade_amd_unsupervised(metrics)
        result['amd_grade'] = amd_grade
        result['amd_reason'] = amd_reason
    
    if 'myopia' in disease_types:
        myopia_grade, myopia_reason = grade_myopia_unsupervised(metrics)
        result['myopia_grade'] = myopia_grade
        result['myopia_reason'] = myopia_reason
    
    if 'cataract' in disease_types:
        cataract_grade, cataract_reason = grade_cataract_unsupervised(metrics)
        result['cataract_grade'] = cataract_grade
        result['cataract_reason'] = cataract_reason
    
    return result


def batch_process_directory(input_dir: Union[str, Path], output_file: Union[str, Path] = None, 
                          disease_types: List[str] = None) -> List[Dict]:
    """
    Batch process all JSON files in a directory.
    
    Args:
        input_dir: directory containing JSON files
        output_file: output file path
        disease_types: diseases to grade
        
    Returns:
        List of grading results
    """
    input_dir = Path(input_dir)
    results = []
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    for json_file in json_files:
        try:
            result = process_json_file(json_file, disease_types)
            results.append(result)
            
            # Print summary
            grade_info = []
            if 'dr_grade' in result:
                grade_info.append(f"DR:{result['dr_grade']}")
            if 'glaucoma_grade' in result:
                grade_info.append(f"Glaucoma:{result['glaucoma_grade']}")
            if 'amd_grade' in result:
                grade_info.append(f"AMD:{result['amd_grade']}")
            if 'myopia_grade' in result:
                grade_info.append(f"Myopia:{result['myopia_grade']}")
            if 'cataract_grade' in result:
                grade_info.append(f"Cataract:{'yes' if result['cataract_grade'] == 1 else 'no'}")
            
            print(f"Processed: {result['filename']} - {', '.join(grade_info)}")
            
        except Exception as e:
            print(f"Failed to process {json_file}: {str(e)}")
            continue
    
    # Aggregate grade distributions
    stats = {}
    for disease in ['dr', 'glaucoma', 'amd', 'myopia', 'cataract']:
        grade_key = f'{disease}_grade'
        if any(grade_key in r for r in results):
            # AMD has grades 0-3, cataract 0-1, others 0-4
            if disease == 'amd':
                grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            elif disease == 'cataract':
                grade_counts = {0: 0, 1: 0}
            else:
                grade_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for result in results:
                if grade_key in result:
                    grade_counts[result[grade_key]] += 1
            stats[disease] = grade_counts
    
    print("\nSummary:")
    for disease, counts in stats.items():
        print(f"\n{disease.upper()} grade distribution:")
        if disease == 'cataract':
            print(f"  No cataract: {counts[0]}")
            print(f"  Cataract: {counts[1]}")
        else:
            for grade, count in counts.items():
                print(f"  Grade {grade}: {count}")
    
    # Save results
    if output_file:
        output_file = Path(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_files': len(results),
                    'grade_distribution': stats
                },
                'details': results
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Disease grading from retina analysis results')
    parser.add_argument('input', type=str, help='Input JSON file or directory containing JSON files')
    parser.add_argument('-o', '--output', type=str, help='Output file path (JSON)')
    parser.add_argument('--diseases', nargs='+', 
                       choices=['dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'all'],
                       default=['all'],
                       help='Diseases to analyze (default: all)')
    parser.add_argument('--show-details', action='store_true', help='Show grading rationale')
    
    args = parser.parse_args()
    
    if 'all' in args.diseases:
        disease_types = None  # None means analyze all
    else:
        disease_types = args.diseases
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        result = process_json_file(input_path, disease_types)
        print(f"\nFile: {result['filename']}")
        
        if 'dr_grade' in result:
            print(f"DR grade: {result['dr_grade']}")
            if args.show_details:
                print(f"  Reason: {result['dr_reason']}")
        
        if 'glaucoma_grade' in result:
            print(f"Glaucoma grade: {result['glaucoma_grade']}")
            if args.show_details:
                print(f"  Reason: {result['glaucoma_reason']}")
        
        if 'amd_grade' in result:
            print(f"AMD grade: {result['amd_grade']}")
            if args.show_details:
                print(f"  Reason: {result['amd_reason']}")
        
        if 'myopia_grade' in result:
            print(f"Myopia grade: {result['myopia_grade']}")
            if args.show_details:
                print(f"  Reason: {result['myopia_reason']}")
        
        if 'cataract_grade' in result:
            print(f"Cataract: {'yes' if result['cataract_grade'] == 1 else 'no'}")
            if args.show_details:
                print(f"  Reason: {result['cataract_reason']}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nSaved to: {args.output}")
            
    elif input_path.is_dir():
        # Batch process directory
        batch_process_directory(input_path, args.output, disease_types)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    

if __name__ == '__main__':
    main()