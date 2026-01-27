import cv2
import numpy as np
from skimage.measure import label, regionprops
import os
from math import atan2, degrees
from datasets.utils import remove_black_edge


def analyze_lesion_mask(mask, disc_area=None, fundus_area=None, valid_lesion_types=None, macula_center=None, is_left_eye=None):
    """
    Analyze lesion segmentation.

    Args:
        mask: lesion mask
        disc_area: optic disc area, for ratio metrics
        fundus_area: valid fundus area, for ratio metrics
        valid_lesion_types: list of lesion class ids; defaults to [1,2,3,4,5]
        macula_center: macula center (y, x) for quadrant split
        is_left_eye: True for left eye, False for right eye (required)

    Returns:
        dict: aggregated metrics
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Define valid lesion classes (default 1-5)
    if valid_lesion_types is None:
        valid_lesion_types = [1, 2, 3, 4, 5]  # adjust if needed
    
    # Create binary mask for valid lesion classes
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    for lesion_type in valid_lesion_types:
        binary_mask |= (mask == lesion_type)
    
    # Connected components
    labeled = label(binary_mask, connectivity=2)
    props = regionprops(labeled)

    # Filter out small regions (noise) - minimum 10 pixels
    min_component_area = 10
    filtered_props = [p for p in props if p.area >= min_component_area]

    # Basic stats on filtered components
    lesion_count = len(filtered_props)
    lesion_areas = [p.area for p in filtered_props]
    lesion_perimeters = [p.perimeter for p in filtered_props]
    lesion_centroids = [p.centroid for p in filtered_props]
    lesion_aspect_ratios = [
        p.major_axis_length / p.minor_axis_length if p.minor_axis_length != 0 else 0
        for p in filtered_props
    ]
    lesion_circularities = [
        4 * np.pi * p.area / (p.perimeter ** 2) if p.perimeter != 0 else 0
        for p in filtered_props
    ]

    # Total lesion area
    total_lesion_area = np.sum(lesion_areas)

    # Global image info
    image_area = mask.shape[0] * mask.shape[1]
    effective_fundus_area = fundus_area if fundus_area else image_area

    # Ratios to fundus area
    lesion_density = lesion_count / effective_fundus_area
    lesion_coverage = total_lesion_area / effective_fundus_area

    # Ratio to disc area
    lesion_disc_ratio = total_lesion_area / disc_area if disc_area and disc_area > 0 else 0

    # Size buckets
    small_lesions = [area for area in lesion_areas if area < 50]
    medium_lesions = [area for area in lesion_areas if 50 <= area < 200]
    large_lesions = [area for area in lesion_areas if area >= 200]

    # Spatial distribution metrics
    if lesion_centroids:
        centroids_array = np.array(lesion_centroids)
        centroid_std_x = np.std(centroids_array[:, 1])
        centroid_std_y = np.std(centroids_array[:, 0])
        centroid_range_x = np.max(centroids_array[:, 1]) - np.min(centroids_array[:, 1])
        centroid_range_y = np.max(centroids_array[:, 0]) - np.min(centroids_array[:, 0])
        
        # Dispersion (mean distance to centroid)
        centroid_diffs = np.linalg.norm(centroids_array - centroids_array.mean(axis=0), axis=1)
        centroid_dispersion = np.mean(centroid_diffs)
    else:
        centroid_std_x = centroid_std_y = 0
        centroid_range_x = centroid_range_y = 0
        centroid_dispersion = 0

    # Quadrant analysis using macula center
    h, w = mask.shape
    
    # Use macula center; fallback to image center
    if macula_center is not None:
        center_y, center_x = macula_center
    else:
        center_y, center_x = h // 2, w // 2
    
    # Require laterality
    if is_left_eye is None:
        raise ValueError("is_left_eye parameter is required. Please specify True for left eye or False for right eye.")
    
    # Count lesions per quadrant
    quadrant_counts = {
        'superior_temporal': 0,
        'superior_nasal': 0,
        'inferior_temporal': 0,
        'inferior_nasal': 0
    }
    
    for centroid in lesion_centroids:
        y, x = centroid
        
        # Quadrant split around macula with eye laterality
        if y < center_y:
            if x < center_x:
                quadrant_counts['superior_temporal' if is_left_eye else 'superior_nasal'] += 1
            else:
                quadrant_counts['superior_nasal' if is_left_eye else 'superior_temporal'] += 1
        else:
            if x < center_x:
                quadrant_counts['inferior_temporal' if is_left_eye else 'inferior_nasal'] += 1
            else:
                quadrant_counts['inferior_nasal' if is_left_eye else 'inferior_temporal'] += 1

    # Aggregate results
    results = {
        "lesion_count": lesion_count,
        "total_lesion_area": total_lesion_area,
        
        "lesion_density": lesion_density,
        "lesion_coverage_ratio": lesion_coverage,
        "lesion_disc_ratio": lesion_disc_ratio,
        
        "mean_lesion_area": np.mean(lesion_areas) if lesion_areas else 0,
        "max_lesion_area": np.max(lesion_areas) if lesion_areas else 0,
        "min_lesion_area": np.min(lesion_areas) if lesion_areas else 0,
        "std_lesion_area": np.std(lesion_areas) if lesion_areas else 0,
        
        "small_lesion_count": len(small_lesions),
        "medium_lesion_count": len(medium_lesions),
        "large_lesion_count": len(large_lesions),
        "small_lesion_total_area": np.sum(small_lesions),
        "medium_lesion_total_area": np.sum(medium_lesions),
        "large_lesion_total_area": np.sum(large_lesions),
        
        "mean_circularity": np.mean(lesion_circularities) if lesion_circularities else 0,
        "mean_aspect_ratio": np.mean(lesion_aspect_ratios) if lesion_aspect_ratios else 0,
        
        "centroid_dispersion": centroid_dispersion,
        "centroid_std_x": centroid_std_x,
        "centroid_std_y": centroid_std_y,
        "centroid_range_x": centroid_range_x,
        "centroid_range_y": centroid_range_y,
        
        "quadrant_superior_temporal_count": quadrant_counts['superior_temporal'],
        "quadrant_superior_nasal_count": quadrant_counts['superior_nasal'],
        "quadrant_inferior_temporal_count": quadrant_counts['inferior_temporal'],
        "quadrant_inferior_nasal_count": quadrant_counts['inferior_nasal']
    }

    return results


def analyze_tessellate_mask(mask, original_image_path=None, background_threshold=10, fundus_area=None, min_component_size: int = 0):
    """
    Analyze tessellation (tessellate) segmentation results.

    Args:
        mask: tessellation mask
        original_image_path: optional original fundus image path to estimate fundus area
        background_threshold: pixel threshold to exclude background
        fundus_area: precomputed fundus area if available

    Returns:
        dict: analysis results
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Analyze regions where value==1 (tessellation regions)
    binary = (mask == 1).astype(np.uint8)
    labeled = label(binary, connectivity=2)
    props_all = regionprops(labeled)
    # Filter tiny components to reduce noise inflation
    min_area = max(1, int(min_component_size))
    props = [p for p in props_all if p.area >= min_area]

    # Region metrics
    areas = [p.area for p in props]
    perimeters = [p.perimeter for p in props]
    aspect_ratios = [
        p.major_axis_length / p.minor_axis_length if p.minor_axis_length != 0 else 0
        for p in props
    ]
    circularities = [
        4 * np.pi * p.area / (p.perimeter ** 2) if p.perimeter != 0 else 0
        for p in props
    ]
    centroids = np.array([p.centroid for p in props])
    
    # Dispersion (mean distance to centroid)
    if len(centroids) > 1:
        centroid_diffs = np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)
        dispersion = np.mean(centroid_diffs)
    else:
        dispersion = 0

    # Total tessellate area
    total_tessellate_area = np.sum(areas)

    # Use fundus area
    if fundus_area is not None:
        # Use precomputed fundus area
        effective_fundus_area = fundus_area
        tessellate_ratio = total_tessellate_area / effective_fundus_area if effective_fundus_area > 0 else 0
    elif original_image_path and os.path.exists(original_image_path):
        # Estimate fundus area from original image
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is not None:
            # Extract fundus region (non-background)
            fundus_mask = (original_image > background_threshold).astype(np.uint8)
            
            # Remove small noise with morphology
            kernel = np.ones((5,5), np.uint8)
            fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_OPEN, kernel)
            fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_CLOSE, kernel)
            
            # Compute fundus area
            effective_fundus_area = np.sum(fundus_mask)
            
            # Ratio of tessellate area to fundus area
            tessellate_ratio = total_tessellate_area / effective_fundus_area if effective_fundus_area > 0 else 0
        else:
            effective_fundus_area = mask.shape[0] * mask.shape[1]
            tessellate_ratio = total_tessellate_area / effective_fundus_area
    else:
        # Fallback to full image area
        effective_fundus_area = mask.shape[0] * mask.shape[1]
        tessellate_ratio = total_tessellate_area / effective_fundus_area

    results = {
        "tessellate_count": len(props),
        "total_tessellate_area": total_tessellate_area,
        "tessellate_ratio": tessellate_ratio,
        "fundus_area_pixels": int(effective_fundus_area) if effective_fundus_area is not None else None,
        "std_tessellate_area": np.std(areas) if areas else 0,
        "mean_circularity": np.mean(circularities) if circularities else 0,
        "mean_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0,
        "centroid_dispersion": dispersion
    }

    return results


def analyze_arc_lesion(mask_path, optic_disc_center, disc_area=None):
    """
    Quantitative analysis for arc lesions (mask value == 1).

    Args:
        mask_path (str): path to PNG mask (1 = arc lesion)
        optic_disc_center (tuple): disc center (y, x)
        disc_area (float): disc area for ratios

    Returns:
        dict: metrics
    """
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read image from {mask_path}")
    
    # Only class 1
    binary = (mask == 1).astype(np.uint8)
    labeled = label(binary, connectivity=2)
    props = regionprops(labeled)

    # Image size and disc center
    h, w = mask.shape
    cy, cx = optic_disc_center

    # Infer laterality (disc center on left half => left eye)
    is_left_eye = cx < (w // 2)

    # Metrics
    areas = [p.area for p in props]
    centroids = [p.centroid for p in props]
    min_dists = [np.linalg.norm(np.array(p.centroid) - np.array([cy, cx])) for p in props]
    angles = [degrees(atan2(p.centroid[0] - cy, p.centroid[1] - cx)) % 360 for p in props]
    angular_coverage = max(angles) - min(angles) if angles else 0

    # Total area
    total_arc_area = np.sum(areas)

    # Ratio to disc area
    if disc_area and disc_area > 0:
        arc_disc_ratio = total_arc_area / disc_area
    else:
        arc_disc_ratio = 0

    # Locate largest lesion
    if areas:
        max_idx = np.argmax(areas)
        y0, x0 = centroids[max_idx]

        # Superior/Inferior
        vertical_pos = "Superior" if y0 < cy else "Inferior"

        # Nasal/Temporal with laterality
        if is_left_eye:
            horizontal_pos = "Temporal" if x0 > cx else "Nasal"
        else:  # right eye
            horizontal_pos = "Temporal" if x0 < cx else "Nasal"

        max_arc_location = f"{vertical_pos}-{horizontal_pos}"
    else:
        max_arc_location = "None"

    results = {
        "count": len(props),
        "total_area": total_arc_area,
        "arc_disc_ratio": arc_disc_ratio,
        "mean_area": np.mean(areas) if areas else 0,
        "max_area": np.max(areas) if areas else 0,
        "min_area": np.min(areas) if areas else 0,
        "std_area": np.std(areas) if areas else 0,
        "mean_distance_to_disc": np.mean(min_dists) if min_dists else 0,
        "angular_coverage_degrees": angular_coverage,
        "max_location": max_arc_location,
        "is_left_eye": is_left_eye
    }

    return results


def analyze_lesions_by_categories(masks_dir: str,
                                  display_name: str,
                                  lesion_types_map: dict,
                                  analysis_size: int,
                                  min_component_size: int,
                                  original_image_path: str,
                                  macula_center: tuple = None,
                                  is_left_eye: bool = None) -> dict:
    """Per-category lesion analysis using saved per-class masks under masks_dir.

    - Aligns with black-edge removal of original image
    - Resizes to a unified analysis size (e.g., 640x640)
    - Filters connected components smaller than min_component_size

    Returns schema:
    {
      'summary': {...},
      'lesion_categories_analysis': {
        'total_categories_detected': int,
        'categories_overview': {<lesion_name>: {...}},
        'detailed_analysis': {<lesion_name>: {...}}
      }
    }
    """
    import cv2
    import numpy as np

    if not masks_dir or not os.path.isdir(masks_dir) or not lesion_types_map:
        return {
            'summary': {
                'total_lesion_categories': 0,
                'total_area_pixels': 0,
                'total_coverage_percentage': 0.0,
            },
            'lesion_categories_analysis': {
                'total_categories_detected': 0,
                'categories_overview': {},
                'detailed_analysis': {},
            }
        }

    # Prepare crop box from original image (black-edge removal)
    crop_box = None
    try:
        ori = cv2.imread(original_image_path)
        if ori is not None:
            _, ymin, ymax, xmin, xmax = remove_black_edge(ori)
            crop_box = (int(ymin), int(ymax), int(xmin), int(xmax))
    except Exception:
        crop_box = None

    # Build fundus mask (effective analysis area) from original image after black-edge removal
    fundus_mask_resized = None
    if original_image_path and os.path.exists(original_image_path):
        try:
            ori_img = cv2.imread(original_image_path)
            if ori_img is not None:
                _, ymin, ymax, xmin, xmax = remove_black_edge(ori_img)
                h0, w0 = ori_img.shape[:2]
                ymin = max(0, min(int(ymin), h0))
                ymax = max(0, min(int(ymax), h0))
                xmin = max(0, min(int(xmin), w0))
                xmax = max(0, min(int(xmax), w0))
                if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                    ori_img = ori_img[ymin:ymax, xmin:xmax]
                gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
                bg_thr = 10
                fundus_mask_resized = (gray > bg_thr).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                fundus_mask_resized = cv2.morphologyEx(fundus_mask_resized, cv2.MORPH_OPEN, kernel)
                fundus_mask_resized = cv2.morphologyEx(fundus_mask_resized, cv2.MORPH_CLOSE, kernel)
                fundus_mask_resized = cv2.resize(fundus_mask_resized, (analysis_size, analysis_size), interpolation=cv2.INTER_NEAREST)
        except Exception:
            fundus_mask_resized = None

    analysis_area = float(np.sum(fundus_mask_resized)) if fundus_mask_resized is not None else float(analysis_size * analysis_size)
    # Use fundus area (eye surface) as denominator for all downstream coverage calculations.
    fundus_area_pixels = analysis_area if analysis_area > 0 else float(analysis_size * analysis_size)
    categories_overview = {}
    detailed_analysis = {}
    total_area_pixels = 0
    categories_detected = 0

    def compute_mask_stats(binary_mask: np.ndarray) -> dict:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        areas = []
        perimeters = []
        circularities = []
        aspect_ratios = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < max(1, int(min_component_size)):
                continue
            areas.append(area)
            comp_mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                peri = float(cv2.arcLength(cnt, True))
                perimeters.append(peri)
                if peri > 0:
                    circ = float(min(1.0, 4 * np.pi * area / (peri * peri)))
                    circularities.append(circ)
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if h > 0 and w > 0:
                    aspect_ratios.append(float(max(w, h) / min(w, h)))

        # Quadrant distribution based on centroid relative to macula center
        h, w = binary_mask.shape[:2]
        
        # Use macula center as origin; fallback to image center
        if macula_center is not None:
            # Scale macula center to analysis size
            cy = int(macula_center[0] * h / analysis_size)
            cx = int(macula_center[1] * w / analysis_size)
        else:
            cy, cx = h // 2, w // 2
        
        # Require laterality
        if is_left_eye is None:
            raise ValueError("is_left_eye parameter is required. Please specify True for left eye or False for right eye.")
            
        q_counts = {
            'superior_temporal': 0,
            'superior_nasal': 0,
            'inferior_temporal': 0,
            'inferior_nasal': 0,
        }
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < max(1, int(min_component_size)):
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT]) + int(stats[label_id, cv2.CC_STAT_WIDTH]) // 2
            y = int(stats[label_id, cv2.CC_STAT_TOP]) + int(stats[label_id, cv2.CC_STAT_HEIGHT]) // 2
            
            # Quadrant split around macula with eye laterality
            if y < cy:
                if x < cx:
                    q_counts['superior_temporal' if is_left_eye else 'superior_nasal'] += 1
                else:
                    q_counts['superior_nasal' if is_left_eye else 'superior_temporal'] += 1
            else:
                if x < cx:
                    q_counts['inferior_temporal' if is_left_eye else 'inferior_nasal'] += 1
                else:
                    q_counts['inferior_nasal' if is_left_eye else 'inferior_temporal'] += 1
        quadrants_with_lesions = sum(1 for v in q_counts.values() if v > 0)
        return {
            'count': len(areas),
            'areas': areas,
            'perimeters': perimeters,
            'circularities': circularities,
            'aspect_ratios': aspect_ratios,
            'quadrant_counts': q_counts,
            'quadrants_with_lesions': quadrants_with_lesions,
        }

    for class_id, lesion_name in lesion_types_map.items():
        file_path = os.path.join(masks_dir, f'{display_name}_{lesion_name}.png')
        if not os.path.isfile(file_path):
            continue
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if crop_box is not None:
            ymin, ymax, xmin, xmax = crop_box
            h, w = mask.shape[:2]
            ymin = max(0, min(int(ymin), h))
            ymax = max(0, min(int(ymax), h))
            xmin = max(0, min(int(xmin), w))
            xmax = max(0, min(int(xmax), w))
            if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                mask = mask[ymin:ymax, xmin:xmax]

        # Resize to analysis size
        mask_resized = cv2.resize((mask > 0).astype(np.uint8), (analysis_size, analysis_size), interpolation=cv2.INTER_NEAREST)

        stats = compute_mask_stats(mask_resized)
        if stats['count'] == 0:
            continue

        categories_detected += 1
        area_sum = int(np.sum(stats['areas']))
        total_area_pixels += area_sum
        coverage = float(area_sum / fundus_area_pixels) if fundus_area_pixels > 0 else 0.0

        categories_overview[lesion_name] = {
            'count': stats['count'],
            'coverage_percentage': coverage,
            'severity': (
                'Minimal' if coverage < 0.01 else 'Mild' if coverage < 0.03 else 'Moderate' if coverage < 0.1 else 'Severe'
            )
        }

        size_small = sum(1 for a in stats['areas'] if a < 50)
        size_medium = sum(1 for a in stats['areas'] if 50 <= a < 200)
        size_large = sum(1 for a in stats['areas'] if a >= 200)

        detailed_analysis[lesion_name] = {
            'measurements': {
                'count': stats['count'],
                'total_area_pixels': area_sum,
                'coverage_percentage': coverage,
                'average_size_pixels': float(np.mean(stats['areas'])) if stats['areas'] else 0.0,
            },
            'shape_characteristics': {
                'circularity': float(np.mean(stats['circularities'])) if stats['circularities'] else 0.0,
                'aspect_ratio': float(np.mean(stats['aspect_ratios'])) if stats['aspect_ratios'] else 0.0,
            },
            'size_distribution': {
                'small_lesions': int(size_small),
                'medium_lesions': int(size_medium),
                'large_lesions': int(size_large),
            },
            'quadrant_distribution': {
                'quadrants_with_lesions': int(stats.get('quadrants_with_lesions', 0)),
                'counts': stats.get('quadrant_counts', {
                    'superior_temporal': 0,
                    'superior_nasal': 0,
                    'inferior_temporal': 0,
                    'inferior_nasal': 0,
                })
            },
                'clinical_assessment': {
                    # Keep legacy key while switching to fundus-area-based ratio.
                    'lesion_to_disc_ratio': coverage,
                    'lesion_to_fundus_ratio': coverage,
                    'severity_level': (
                        'Minimal' if coverage < 0.01 else 'Mild' if coverage < 0.03 else 'Moderate' if coverage < 0.1 else 'Severe'
                    ),
                }
        }

    result = {
        'summary': {
            'total_lesion_categories': categories_detected,
            'total_area_pixels': total_area_pixels,
            'total_coverage_percentage': float(total_area_pixels / fundus_area_pixels) if fundus_area_pixels > 0 else 0.0,
            'fundus_area_pixels': int(fundus_area_pixels),
        },
        'lesion_categories_analysis': {
            'total_categories_detected': categories_detected,
            'categories_overview': categories_overview,
            'detailed_analysis': detailed_analysis,
        }
    }

    # Removed fundus_area from output per requirement
    return result
