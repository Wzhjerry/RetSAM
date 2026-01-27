#!/usr/bin/python

import os
import numpy as np
import cv2
import copy
import numpy.linalg as LA
from skimage import morphology
from scipy import signal


def make_skeleton(mask):
    """Create vessel skeleton"""
    vessel_mask = np.zeros_like(mask)
    vessel_mask[mask > 0] = 1
    skeleton = morphology.skeletonize(vessel_mask)
    skeleton = skeleton.astype(np.uint8)
    skeleton[skeleton > 0] = 1
    return skeleton


def create_skeleton(mask):
    """Create skeleton and split branches"""
    skeleton = make_skeleton(mask)
    skeleton_branch = copy.deepcopy(skeleton)
    
    # Filtering kernels
    kernel_1 = np.ones((3, 3), dtype=np.float32)
    kernel_1[1, 1] = 0
    kernel_2 = np.ones((5, 5), dtype=np.float32)
    kernel_2[1:4, 1:4] = 0
    
    corner_mask_3 = cv2.filter2D(skeleton, -1, kernel_1)
    corner_mask_5 = cv2.filter2D(skeleton, -1, kernel_2)
    
    rows, cols = np.where((corner_mask_3 >= 3) & (corner_mask_5 >= 3))
    points = [f for f in zip(cols, rows)]
    
    for x, y in points:
        if skeleton_branch[y, x] == 1:
            skeleton_branch[y - 1: y + 2, x - 1: x + 2] = 0
    
    return skeleton, skeleton_branch


def quality_control_mask(pr_mask, qc_threshold):
    """Quality control"""
    fg_mask = np.zeros_like(pr_mask)
    fg_mask[pr_mask > 0] = 1
    fg_rate = round(float(np.sum(fg_mask)) / (pr_mask.shape[0] * pr_mask.shape[1]), 4)
    qc_flag = fg_rate >= qc_threshold
    return qc_flag, fg_rate


def split_image(img, x, y):
    """Split image into upper and lower parts"""
    img1 = np.zeros_like(img)
    img2 = np.zeros_like(img)
    img1[:y, :] = img[:y, :]
    img2[y:, :] = img[y:, :]
    return img1, img2


def PJcurvature(x, y):
    """Compute curvature"""
    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])
    
    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])
    
    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)
    
    kappa = round(2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** 1.5, 3)
    return kappa


def vascular_tortuosity(skeleton_branch, top_n=5, step=7):
    """Compute vascular tortuosity"""
    skeleton_branch[0, :] = 0
    skeleton_branch[:, 0] = 0
    skeleton_branch[skeleton_branch.shape[0] - 1, :] = 0
    skeleton_branch[:, skeleton_branch.shape[1] - 1] = 0
    
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0
    corner_mask = cv2.filter2D(skeleton_branch, -1, kernel)
    
    lines_pixel_list = []
    step_core = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    ret, markers = cv2.connectedComponents(skeleton_branch)
    
    if ret > 1:
        for i in range(1, ret):
            tmp_mask = np.zeros_like(skeleton_branch)
            tmp_mask[markers == i] = 1
            if np.sum(tmp_mask) > 50:
                rows_point, cols_point = np.where(tmp_mask * corner_mask == 1)
                rows_bbox, cols_bbox = np.where(tmp_mask)
                
                if len(rows_point) == 2:
                    xmin = np.min(cols_bbox) - 1
                    xmax = np.max(cols_bbox) + 2
                    ymin = np.min(rows_bbox) - 1
                    ymax = np.max(rows_bbox) + 2
                    
                    patch_mask = tmp_mask[ymin: ymax, xmin: xmax]
                    patch_unfinish = np.ones(patch_mask.shape, dtype=bool)
                    
                    line_points = []
                    y = rows_point[0] - ymin
                    x = cols_point[0] - xmin
                    patch_unfinish[y, x] = False
                    line_points.append([y + ymin, x + xmin])
                    flag = True
                    
                    while flag:
                        flag = False
                        pixel_rows, pixel_cols = np.where(
                            patch_mask[y - 1:y + 2, x - 1:x + 2] * step_core * patch_unfinish[y - 1:y + 2, x - 1:x + 2])
                        if len(pixel_rows) > 0:
                            y = pixel_rows[0] + y - 1
                            x = pixel_cols[0] + x - 1
                            line_points.append([y + ymin, x + xmin])
                            patch_unfinish[y, x] = False
                            flag = True
                    lines_pixel_list.append(line_points)
    
    tortuosity_sum_list = []
    for lins_list in lines_pixel_list:
        value_list = []
        for i in range(step, len(lins_list) - step, int(step/2)):
            pixel_l = lins_list[i - step]
            pixel_m = lins_list[i]
            pixel_r = lins_list[i + step]
            
            pixel_x = np.array([pixel_l[1], pixel_m[1], pixel_r[1]], dtype=np.int32)
            pixel_y = np.array([pixel_l[0], pixel_m[0], pixel_r[0]], dtype=np.int32)
            
            value_list.append(PJcurvature(pixel_x, pixel_y))
        
        value_array = np.array(value_list, dtype=np.float32)
        value_positive = np.sort(value_array[np.where(value_array > 0)])[::-1]
        value_negative = np.sort(np.abs(value_array[np.where(value_array < 0)]))[::-1]
        
        length = min(value_positive.shape[0], value_negative.shape[0])
        sum_k = np.sum(value_positive[0:length] * value_negative[0:length])
        tortuosity_sum_list.append(sum_k)
    
    index = np.argsort(tortuosity_sum_list)[::-1]
    tortuosity_result = np.array(tortuosity_sum_list)[index[0:min(len(index), top_n)]]
    
    return None, tortuosity_result


def box_counter(image, box_size):
    """Box counting"""
    image_size = len(image)
    total_box_number = int(image_size / box_size)
    reduced = np.zeros((total_box_number, total_box_number))
    
    for row in range(0, image_size, box_size)[:-1]:
        for col in range(0, image_size, box_size)[:-1]:
            sum_m = np.sum(image[row:box_size + row, col:box_size + col])
            if sum_m >= 1:
                reduced_row = int(row / box_size)
                reduced_col = int(col / box_size)
                reduced[reduced_row, reduced_col] = 1
    
    reduced_box_number = np.sum(reduced)
    return reduced_box_number


def box_size_iterator(image):
    """Iterate box sizes for fractal dimension"""
    list_box_size = range(10, 200, 5)
    list_box_number = np.zeros(len(list_box_size))
    
    for (box_size_ctr, box_size) in enumerate(list_box_size):
        list_box_number[box_size_ctr] = box_counter(image, box_size)
    return list_box_number, list_box_size


def slope_finder(image):
    """Compute fractal dimension"""
    list_of_all = box_size_iterator(image)
    list_box_number = list_of_all[0]
    list_box_size = list_of_all[1]
    list_box_size_inv = np.divide(1.0, list_box_size)
    
    # Hausdorff dimension
    slope_haus = np.polyfit(np.log10(list_box_size_inv), np.log10(list_box_number), 1)
    
    return np.round(slope_haus, 3)


def analyze_vessel_mask(vessel_mask, artery_mask, vein_mask, disc_center=None, disc_radius=None):
    """
    Compute vessel metrics from vessel/artery/vein masks.
    
    Args:
        vessel_mask (numpy.ndarray): vessel mask (H, W), binary 1=vessel
        artery_mask (numpy.ndarray): artery mask (H, W), binary
        vein_mask (numpy.ndarray): vein mask (H, W), binary
        disc_center (tuple, optional): optic disc center (x, y); auto-estimated if None
        disc_radius (int, optional): optic disc radius; auto-estimated if None
        
    Returns:
        dict with vessel metrics, including:
            av_ratio_upper/lower, CRAE/CRVE upper/lower, FDa/FDv,
            artery_tortuosity, vein_tortuosity, qc_flag, fg_rate,
            disc_center, disc_radius, pixel_len
    """
    
    # Input validation
    for name, m in [('vessel_mask', vessel_mask), ('artery_mask', artery_mask), ('vein_mask', vein_mask)]:
        if not isinstance(m, np.ndarray) or len(m.shape) != 2:
            return {'error': f'Input {name} must be a 2D numpy array'}
    if vessel_mask.shape != artery_mask.shape or vessel_mask.shape != vein_mask.shape:
        return {'error': 'vessel_mask, artery_mask, vein_mask must share the same shape'}
    # Normalize to 0/1
    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    artery_mask = (artery_mask > 0).astype(np.uint8)
    vein_mask = (vein_mask > 0).astype(np.uint8)
    
    try:
        # 0. Auto-compute pixel_len
        pixel_len = round(float(24.0 * 500.0 / vessel_mask.shape[1]), 2)
        
        # 1. Build skeleton
        skeleton, skeleton_branch = create_skeleton(vessel_mask)
        
        # 2. Quality control
        qc_flag, fg_rate = quality_control_mask(skeleton, 0.02)
        
        # 3. Estimate disc center/radius if not provided
        if disc_center is None or disc_radius is None:
            estimated_center, estimated_radius = estimate_disc_from_vessels(vessel_mask)
            if disc_center is None:
                disc_center = estimated_center
            if disc_radius is None:
                disc_radius = estimated_radius
        
        # 4. Create annulus mask for diameter measurement
        ring_mask = create_ring_mask_from_center(vessel_mask.shape, disc_center, disc_radius)
        
        # 5. Split into upper/lower regions
        upper_mask, lower_mask = split_image(vessel_mask, disc_center[0], disc_center[1])
        upper_skeleton, lower_skeleton = split_image(skeleton, disc_center[0], disc_center[1])
        upper_artery, lower_artery = split_image(artery_mask, disc_center[0], disc_center[1])
        upper_vein, lower_vein = split_image(vein_mask, disc_center[0], disc_center[1])
        
        # 6. Measure vessel diameters (upper)
        upper_results = measure_vessel_diameters_in_region(
            upper_artery, upper_vein, upper_skeleton, ring_mask, disc_radius, pixel_len
        )
        
        # 7. Measure vessel diameters (lower)
        lower_results = measure_vessel_diameters_in_region(
            lower_artery, lower_vein, lower_skeleton, ring_mask, disc_radius, pixel_len
        )
        
        # 8. Build artery/vein skeletons
        artery_skeleton, artery_skeleton_branch = create_skeleton(artery_mask)
        vein_skeleton, vein_skeleton_branch = create_skeleton(vein_mask)
        
        # 9. Tortuosity
        _, artery_tortuosity_result = vascular_tortuosity(artery_skeleton_branch)
        _, vein_tortuosity_result = vascular_tortuosity(vein_skeleton_branch)
        
        # 10. Fractal dimension
        artery_fd = slope_finder(artery_skeleton)
        vein_fd = slope_finder(vein_skeleton)
        
        # 11. Aggregate results
        results = {
            'av_ratio_upper': upper_results.get('av_ratio'),
            'av_ratio_lower': lower_results.get('av_ratio'),
            'CRAE_upper': upper_results.get('artery_diameters'),
            'CRAE_lower': lower_results.get('artery_diameters'),
            'CRVE_upper': upper_results.get('vein_diameters'),
            'CRVE_lower': lower_results.get('vein_diameters'),
            'FDa': artery_fd[0] if len(artery_fd) > 0 else None,
            'FDv': vein_fd[0] if len(vein_fd) > 0 else None,
            'artery_tortuosity': np.sum(artery_tortuosity_result) if len(artery_tortuosity_result) > 0 else 0,
            'vein_tortuosity': np.sum(vein_tortuosity_result) if len(vein_tortuosity_result) > 0 else 0,
            'qc_flag': qc_flag,
            'fg_rate': fg_rate,
            'disc_center': disc_center,
            'disc_radius': disc_radius,
            'pixel_len': pixel_len
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Error during analysis: {str(e)}'}


def estimate_disc_from_vessels(vessel_mask):
    """
    Estimate disc center and radius from vessel distribution.
    
    Args:
        vessel_mask: vessel segmentation mask
        
    Returns:
        tuple: (center, radius)
    """
    # Find vessel pixels
    vessel_pixels = np.where(vessel_mask > 0)
    
    if len(vessel_pixels[0]) == 0:
        # If no vessels, fall back to image center
        h, w = vessel_mask.shape
        return (w//2, h//2), min(h, w)//8
    
    # Vessel density map
    h, w = vessel_mask.shape
    grid_size = 32
    density_map = np.zeros((h//grid_size + 1, w//grid_size + 1))
    
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            patch = vessel_mask[i:i+grid_size, j:j+grid_size]
            density_map[i//grid_size, j//grid_size] = np.sum(patch > 0)
    
    # Use highest density area as disc center
    max_density_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
    center_y = max_density_idx[0] * grid_size + grid_size // 2
    center_x = max_density_idx[1] * grid_size + grid_size // 2
    
    # Radius estimate based on image size
    radius = min(h, w) // 8
    
    return (center_x, center_y), radius


def create_ring_mask_from_center(shape, center, radius):
    """
    Create an annulus mask from center and radius.
    
    Args:
        shape: image shape (H, W)
        center: center point (x, y)
        radius: radius
        
    Returns:
        numpy.ndarray: ring mask with 1.5D~2.0D set to 1
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    
    # Distance to center
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Annulus (1.5D to 2.0D)
    inner_radius = 1.5 * radius
    outer_radius = 2.0 * radius
    
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    ring_mask[(dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)] = 1
    
    return ring_mask


def measure_vessel_diameters_in_region(artery_mask, vein_mask, skeleton, ring_mask, disc_radius, pixel_len):
    """
    Measure vessel diameters within a ring region.
    
    Args:
        artery_mask: binary artery mask
        vein_mask: binary vein mask
        skeleton: vessel skeleton
        ring_mask: annulus mask
        disc_radius: disc radius
        pixel_len: pixel length for physical units
        
    Returns:
        dict: artery/vein diameters and ratio
    """
    # Apply annulus
    masked_skeleton = skeleton * ring_mask
    masked_artery = (artery_mask > 0).astype(np.uint8) * ring_mask
    masked_vein = (vein_mask > 0).astype(np.uint8) * ring_mask
    
    # Artery diameters
    artery_diameters = measure_diameters_for_vessel_type(masked_artery, masked_skeleton, disc_radius)
    
    # Vein diameters
    vein_diameters = measure_diameters_for_vessel_type(masked_vein, masked_skeleton, disc_radius)
    
    # Convert to physical units
    artery_diameters_real = [d * pixel_len for d in artery_diameters if d > 0]
    vein_diameters_real = [d * pixel_len for d in vein_diameters if d > 0]
    
    # Stats
    if len(artery_diameters_real) > 0 and len(vein_diameters_real) > 0:
        artery_stats = [max(artery_diameters_real), min(artery_diameters_real), np.mean(artery_diameters_real)]
        vein_stats = [max(vein_diameters_real), min(vein_diameters_real), np.mean(vein_diameters_real)]
        av_ratio = round(np.mean(artery_diameters_real) / np.mean(vein_diameters_real), 3)
    else:
        artery_stats = [None, None, None]
        vein_stats = [None, None, None]
        av_ratio = None
    
    return {
        'artery_diameters': artery_stats,
        'vein_diameters': vein_stats,
        'av_ratio': av_ratio
    }


def measure_diameters_for_vessel_type(vessel_mask, skeleton, disc_radius, top_n=3):
    """
    Measure diameters for a single vessel type.
    
    Args:
        vessel_mask: single-type vessel mask
        skeleton: vessel skeleton
        disc_radius: disc radius
        top_n: return top-n diameters
        
    Returns:
        list: vessel diameters
    """
    diameters = []
    
    # Connected components
    ret, markers = cv2.connectedComponents(vessel_mask)
    
    if ret > 1:
        for i in range(1, ret):
            component_mask = np.zeros_like(vessel_mask)
            component_mask[markers == i] = 1
            
            # Skip tiny components
            if np.sum(component_mask) > disc_radius // 4:
                # Distance transform
                dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
                
                # Evaluate only on skeleton
                dist_transform[skeleton == 0] = 0
                
                # Max value as radius
                max_radius = np.max(dist_transform)
                if max_radius > 0:
                    diameter = max_radius * 2
                    diameters.append(diameter)
    
    # Top-N diameters
    diameters.sort(reverse=True)
    return diameters[:top_n] if len(diameters) >= top_n else diameters


def analyze_vessel_mask_overall(vessel_mask, artery_mask, vein_mask, disc_mask=None, disc_center=None, disc_radius=None):
    """
    Compute global vessel metrics without splitting upper/lower fields.
    
    Args:
        vessel_mask (numpy.ndarray): vessel mask, binary
        artery_mask (numpy.ndarray): artery mask, binary
        vein_mask (numpy.ndarray): vein mask, binary
        disc_mask (numpy.ndarray, optional): disc/cup mask (H, W); >0 is disc region
        disc_center (tuple, optional): disc center (x, y); overridden if disc_mask provided
        disc_radius (int, optional): disc radius; overridden if disc_mask provided
        
    Returns:
        dict with vessel metrics (av_ratio, CRAE/CRVE, FDa/FDv, tortuosity, qc_flag, fg_rate, disc_center, disc_radius, pixel_len)
    """
    
    # Input validation
    for name, m in [('vessel_mask', vessel_mask), ('artery_mask', artery_mask), ('vein_mask', vein_mask)]:
        if not isinstance(m, np.ndarray) or len(m.shape) != 2:
            return {'error': f'Input {name} must be a 2D numpy array'}
    if vessel_mask.shape != artery_mask.shape or vessel_mask.shape != vein_mask.shape:
        return {'error': 'vessel_mask, artery_mask, vein_mask must share the same shape'}
    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    artery_mask = (artery_mask > 0).astype(np.uint8)
    vein_mask = (vein_mask > 0).astype(np.uint8)
    
    if disc_mask is not None:
        if not isinstance(disc_mask, np.ndarray) or len(disc_mask.shape) != 2:
            return {'error': 'disc_mask must be a 2D numpy array'}
        if disc_mask.shape != vessel_mask.shape:
            return {'error': 'disc_mask and vessel_mask must share the same shape'}
    
    try:
        # 0. Auto-compute pixel_len
        pixel_len = round(float(24.0 * 500.0 / vessel_mask.shape[1]), 2)
        
        # 1. Build skeleton
        skeleton, skeleton_branch = create_skeleton(vessel_mask)
        
        # 2. Quality control
        qc_flag, fg_rate = quality_control_mask(skeleton, 0.02)
        
        # 3. Disc center/radius from mask or estimate
        if disc_mask is not None:
            disc_center, disc_radius = calculate_disc_from_mask(disc_mask)
        elif disc_center is None or disc_radius is None:
            estimated_center, estimated_radius = estimate_disc_from_vessels(vessel_mask)
            if disc_center is None:
                disc_center = estimated_center
            if disc_radius is None:
                disc_radius = estimated_radius
        
        # 4. Annulus mask for diameter measurement
        ring_mask = create_ring_mask_from_center(vessel_mask.shape, disc_center, disc_radius)
        
        # 5. Overall diameters (no upper/lower split)
        overall_results = measure_vessel_diameters_in_region(
            artery_mask, vein_mask, skeleton, ring_mask, disc_radius, pixel_len
        )
        
        # 6. Artery/vein skeletons
        artery_skeleton, artery_skeleton_branch = create_skeleton(artery_mask)
        vein_skeleton, vein_skeleton_branch = create_skeleton(vein_mask)
        
        # 7. Tortuosity
        _, artery_tortuosity_result = vascular_tortuosity(artery_skeleton_branch)
        _, vein_tortuosity_result = vascular_tortuosity(vein_skeleton_branch)
        
        # 8. Fractal dimension
        artery_fd = slope_finder(artery_skeleton)
        vein_fd = slope_finder(vein_skeleton)
        
        # 9. Aggregate results
        results = {
            'av_ratio': overall_results.get('av_ratio'),
            'CRAE': overall_results.get('artery_diameters'),
            'CRVE': overall_results.get('vein_diameters'),
            'FDa': artery_fd[0] if len(artery_fd) > 0 else None,
            'FDv': vein_fd[0] if len(vein_fd) > 0 else None,
            'artery_tortuosity': np.sum(artery_tortuosity_result) if len(artery_tortuosity_result) > 0 else 0,
            'vein_tortuosity': np.sum(vein_tortuosity_result) if len(vein_tortuosity_result) > 0 else 0,
            'qc_flag': qc_flag,
            'fg_rate': fg_rate,
            'disc_center': disc_center,
            'disc_radius': disc_radius,
            'pixel_len': pixel_len
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Error during analysis: {str(e)}'}


def calculate_disc_from_mask(disc_mask):
    """
    Compute disc center and radius from disc/cup mask.
    
    Args:
        disc_mask (numpy.ndarray): disc/cup mask; >0 is disc region
        
    Returns:
        tuple: (center, radius)
    """
    # Disc region
    disc_pixels = np.where(disc_mask > 0)
    
    if len(disc_pixels[0]) == 0:
        # No disc region; fallback to image center
        h, w = disc_mask.shape
        return (w//2, h//2), min(h, w)//8
    
    # Centroid as center
    center_y = int(np.mean(disc_pixels[0]))
    center_x = int(np.mean(disc_pixels[1]))
    
    # Bounding box
    min_y, max_y = np.min(disc_pixels[0]), np.max(disc_pixels[0])
    min_x, max_x = np.min(disc_pixels[1]), np.max(disc_pixels[1])
    
    # Radius as half of the smaller bbox edge
    width = max_x - min_x
    height = max_y - min_y
    radius = min(width, height) // 2
    
    # Enforce minimum radius
    if radius < 20:
        radius = 20
    
    return (center_x, center_y), radius


if __name__ == '__main__':
    print("Vessel analysis utilities")
    print("="*50)
    print("Functions:")
    print("")
    print("1) analyze_vessel_mask_overall() - global metrics (recommended)")
    print("   vessel_mask: 2D numpy array, 1=artery, 2=vein")
    print("   disc_mask: 2D numpy array, >0 is disc region (optional)")
    print("   returns av_ratio, CRAE, CRVE, etc.")
    print("")
    print("2) analyze_vessel_mask() - split upper/lower metrics")
    print("   returns separate upper/lower metrics")
    print("")
    print("Example:")
    print("-"*30)
    print("import numpy as np")
    print("from utils.airdoc_av_seg import analyze_vessel_mask_overall")
    print("")
    print("# create sample masks")
    print("vessel_mask = np.zeros((512, 512), dtype=np.uint8)")
    print("disc_mask = np.zeros((512, 512), dtype=np.uint8)")
    print("# add arteries/veins...")
    print("# add disc region...")
    print("")
    print("# Method 1: use disc mask to auto-compute center/radius (recommended)")
    print("results = analyze_vessel_mask_overall(vessel_mask, disc_mask=disc_mask)")
    print("")
    print("# Method 2: provide disc center/radius manually")
    print("results = analyze_vessel_mask_overall(vessel_mask, disc_center=(256, 256), disc_radius=64)")
    print("")
    print("# Method 3: fully auto estimate (less accurate)")
    print("results = analyze_vessel_mask_overall(vessel_mask)")
    print("")
    print("print(results)")
    print("")
    print("Results include:")
    print("- av_ratio")
    print("- CRAE [max, min, avg]")
    print("- CRVE [max, min, avg]")
    print("- FDa / FDv (fractal dimensions)")
    print("- artery_tortuosity / vein_tortuosity")
    print("- qc_flag")
    print("- fg_rate")
    print("- disc_center")
    print("- disc_radius")
    print("- pixel_len")
    print("")
    print("For upper/lower metrics, use analyze_vessel_mask().")
