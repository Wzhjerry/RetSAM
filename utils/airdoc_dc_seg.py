"""
Glaucoma Analysis and Visualization Module

This module provides functions for analyzing glaucoma-related features from fundus images,
including disc and cup segmentation analysis, ISNT (Inferior, Superior, Nasal, Temporal) 
measurements, and C/D ratio calculations.

Main Function: visualize_glaucoma()

INPUT:
- image: RGB fundus image (numpy array, shape: H x W x 3)
- mask: Segmentation mask (numpy array, shape: H x W, values: 0=background, 1=disc, 2=cup)
- eyeside_value: Eye side indicator (int, 0 for left eye, 1 for right eye, None for auto-detection)
- flag_isnt: Whether to draw ISNT regions (bool, True/False, default: True)
- visual_type: Visualization type (int, 0=no visualization, 1=basic, 2=detailed, default: 1)

MASK FORMAT:
- Value 0: Background (no structure)
- Value 1: Optic disc region (excluding cup)
- Value 2: Optic cup region
- For analysis: disc = all pixels > 0, cup = pixels == 2

OUTPUT (Dictionary):
{
    'image': Processed image with annotations (numpy array or None if failed),
    'cd_ratio': Cup-to-disc ratio (float, 0.0-1.0),
    'vh_ratio': Vertical-to-horizontal ratio (float),
    'angle_str': Angle measurement as string (str, e.g., '+15.2' or '-8.7'),
    'isnt_list': ISNT measurements (list of 4 lists, each containing 5 values),
    'cd_list': Cup and disc diameter measurements (list of 4 values: [hori_cd, vert_cd, hori_dd, vert_dd]),
    'eyeside_detected': Auto-detected eye side (int, 0=left eye, 1=right eye)
}

FEATURES:
- Automatic eye side detection: If eyeside_value is None, the function will automatically 
  determine whether it's a left eye or right eye based on the position of the optic disc 
  and cup in the image (left half = left eye, right half = right eye).
- Single mask input: Simplified interface using one mask with different values for disc and cup.

USAGE EXAMPLE:
```python
import cv2
import numpy as np
from utils.airdoc_dc_seg import visualize_glaucoma

# Load your data
image = cv2.imread('fundus_image.jpg')
mask = cv2.imread('segmentation_mask.png', 0)  # Values: 0=background, 1=disc, 2=cup

# Analyze glaucoma features with automatic eye side detection
result = visualize_glaucoma(
    image=image,
    mask=mask,
    eyeside_value=None,  # Auto-detect eye side
    flag_isnt=True,      # Draw ISNT regions
    visual_type=1,       # Basic visualization
    macula_center=(320, 240)  # Optional: macula center for improved eye side detection
)

# Access results
if result['image'] is not None:
    print(f"Detected eye side: {'Left' if result['eyeside_detected'] == 0 else 'Right'}")
    print(f"C/D Ratio: {result['cd_ratio']:.4f}")
    print(f"V/H Ratio: {result['vh_ratio']:.2f}")
    print(f"Angle: {result['angle_str']}")
    cv2.imwrite('result.jpg', result['image'])
else:
    print("Analysis failed - check input mask")

# Or specify eye side manually
result_manual = visualize_glaucoma(
    image=image,
    mask=mask,
    eyeside_value=1,     # Manually specify right eye
    flag_isnt=True,
    visual_type=1
)
```
"""

import copy
import os
import cv2
import numpy as np

def get_edge_width(img, disc_edge, cup_edge, mean_centroid):
    x = int(mean_centroid[0])
    y = int(mean_centroid[1])
    hori_line = disc_edge[y, :]
    col_idx = np.where(hori_line==1)[0]
    if len(col_idx)==0:
        disc_hori_edge_pnts = None
    else:
        disc_hori_edge_pnts = [(col_idx[0], y), (col_idx[-1],y)]

    vert_line = disc_edge[:,x]
    rs_idx = np.where(vert_line==1)[0]
    if len(rs_idx)==0:
        disc_vert_edge_pnts = None
    else:
        disc_vert_edge_pnts = [(x, rs_idx[0]), (x, rs_idx[-1])]

    hori_line = cup_edge[y, :]
    col_idx = np.where(hori_line==1)[0]
    if len(col_idx)==0:
        cup_hori_edge_pnts = None
    else:
        cup_hori_edge_pnts = [(col_idx[0], y), (col_idx[-1],y)]

    vert_line = cup_edge[:,x]
    rs_idx = np.where(vert_line==1)[0]
    if len(rs_idx)==0:
        cup_vert_edge_pnts = None
    else:
        cup_vert_edge_pnts = [(x, rs_idx[0]), (x, rs_idx[-1])]

    # draw for test
    if disc_hori_edge_pnts and disc_vert_edge_pnts:
        cv2.circle(img, disc_hori_edge_pnts[0], 1, (0, 0, 255), 2)
        cv2.circle(img, disc_hori_edge_pnts[1], 1, (0, 0, 255), 2)
        cv2.circle(img, disc_vert_edge_pnts[0], 1, (0, 0, 255), 2)
        cv2.circle(img, disc_vert_edge_pnts[1], 1, (0, 0, 255), 2)

        cv2.line(img, disc_hori_edge_pnts[0], disc_hori_edge_pnts[1], (0, 255,255), 1) # draw disc hori line
        cv2.line(img, disc_vert_edge_pnts[0], disc_vert_edge_pnts[1], (0, 255,255), 1) # draw disc vert line

    if cup_hori_edge_pnts and cup_vert_edge_pnts:
        cv2.circle(img, cup_hori_edge_pnts[0], 1, (0, 0, 255), 2)
        cv2.circle(img, cup_hori_edge_pnts[1], 1, (0, 0, 255), 2)
        cv2.circle(img, cup_vert_edge_pnts[0], 1, (0, 0, 255), 2)
        cv2.circle(img, cup_vert_edge_pnts[1], 1, (0, 0, 255), 2)

        cv2.line(img, cup_hori_edge_pnts[0], cup_hori_edge_pnts[1], (0, 0, 255), 2) # draw cup hori line
        cv2.line(img, cup_vert_edge_pnts[0], cup_vert_edge_pnts[1], (100, 100, 150), 2) # draw cup vert line

        cv2.putText(img, 'A', (cup_hori_edge_pnts[0][0]+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1) # cup hori
        cv2.putText(img, 'B', (x+10, cup_vert_edge_pnts[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 150), 1) # cup vert
    return img

def extract_line_at_specific_angle(edge_mask, mean_cent, angle):
    """extract the pixels and their coordinates along a line in image
    
    args:
    disc_cup_comb: binary image containing 0 or 1
    """

    pnts = []
    values = []
    h, w = edge_mask.shape
    cx, cy = mean_cent
    if (angle >= -np.pi/4  and angle <= np.pi/4) or \
       (angle >= 7*np.pi/4 and angle <= 2*np.pi) or \
       (angle >= 3*np.pi/4 and angle <= 5*np.pi/4):
        k = np.tan(angle)
        for x in range(w):
            y = int(k * (cx - x) + cy)
            pnts.append((x, y))
            if y>=0 and y<h:
                values.append(edge_mask[y, x])
            else:
                values.append(-1)
    else:
        angle += np.pi/2
        k = np.tan(angle)
        for y in range(h):
            x = int(k * (y - cy) + cx)
            pnts.append((x,y))
            if x >=0 and x < w:
                values.append(edge_mask[y, x])
            else:
                values.append(-1)
    #print('values {}'.format(values))
    return pnts, values


def get_intersect_pnts_from_line(mean_cent, pnts, values, direction):
    """
    args:
    direction: 0 - top, 1 - bottom, 2 - left, 3 - right, 4 - return both side
    """

    cx, cy = mean_cent

    if direction == 0:
       # pick closest point above center
       best = None
       for val, pnt in zip(values, pnts):
           if val == 1 and pnt[1] <= cy:
               if best is None or pnt[1] > best[1]:
                   best = pnt
       return best if best is not None else (0, 0)
    elif direction == 1:
       # closest point below center
       best = None
       for val, pnt in zip(values, pnts):
           if val == 1 and pnt[1] >= cy:
               if best is None or pnt[1] < best[1]:
                   best = pnt
       return best if best is not None else (0, 0)
    elif direction == 2:
       # closest point to the left of center
       best = None
       for val, pnt in zip(values, pnts):
           if val == 1 and pnt[0] <= cx:
               if best is None or pnt[0] > best[0]:
                   best = pnt
       return best if best is not None else (0, 0)
    elif direction == 3:
       # closest point to the right of center
       best = None
       for val, pnt in zip(values, pnts):
           if val == 1 and pnt[0] >= cx:
               if best is None or pnt[0] < best[0]:
                   best = pnt
       return best if best is not None else (0, 0)
    elif direction == 4:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       if len(temp_p) <= 1:
           return [(0,0),(0,0)]
       else:
           # Use the first and last intersections so diameters span the shape, not just the first two pixels
           return [tuple(temp_p[0]), tuple(temp_p[-1])]


def pnt_distance(pnt1, pnt2):
    x1, y1 = pnt1
    x2, y2 = pnt2
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

def _ray_endpoint(mask, center, direction):
    """Walk from center along axis direction until mask ends; return last in-mask point."""
    h, w = mask.shape
    cx, cy = center
    if direction == 0:   # up
        dx, dy = 0, -1
    elif direction == 1: # down
        dx, dy = 0, 1
    elif direction == 2: # left
        dx, dy = -1, 0
    elif direction == 3: # right
        dx, dy = 1, 0
    else:
        return (0, 0)

    x, y = int(cx), int(cy)
    last = None
    while 0 <= x < w and 0 <= y < h:
        if mask[y, x] == 0:
            break
        last = (x, y)
        x += dx
        y += dy
    return last if last is not None else (0, 0)

def _axis_angle_for_direction(direction):
    # 0/1 are vertical (up/down), 2/3 are horizontal (left/right)
    return np.pi/2 if direction in (0, 1) else 0

def compute_edge_width_on_axis(disc, cup, mean_cent, direction, img=None):
    """Compute disc–cup distance along a single axis-aligned line through center."""
    disc = np.where(disc > 0, 1, 0)
    cup = np.where(cup > 0, 1, 0)
    disc_pnt = _ray_endpoint(disc, mean_cent, direction)
    cup_pnt = _ray_endpoint(cup, mean_cent, direction)

    if disc_pnt == (0, 0) or cup_pnt == (0, 0):
        d = 0.0
    else:
        d = pnt_distance(disc_pnt, cup_pnt)

    if img is not None:
        cv2.line(img, disc_pnt, cup_pnt, (0, 0, 255), 1)

    # keep return signature: [value, x1, y1, x2, y2]
    return [d, disc_pnt[0], disc_pnt[1], cup_pnt[0], cup_pnt[1]]

def compute_diameter_on_axis(mask, mean_cent, axis, img=None):
    """Compute full diameter (disc or cup) along a single axis-aligned line through center."""
    mask = np.where(mask > 0, 1, 0)
    if axis == 'v':
        pnt1 = _ray_endpoint(mask, mean_cent, 0)  # up
        pnt2 = _ray_endpoint(mask, mean_cent, 1)  # down
    else:
        pnt1 = _ray_endpoint(mask, mean_cent, 2)  # left
        pnt2 = _ray_endpoint(mask, mean_cent, 3)  # right
    d = 0.0 if pnt1 == (0, 0) or pnt2 == (0, 0) else pnt_distance(pnt1, pnt2)

    if img is not None:
        cv2.line(img, pnt1, pnt2, (0, 0, 255), 1)

    return d



def compute_edge_width_from_sector(disc, cup, mean_cent, sector, direction, img=None):
    """
    argx:
    sector: (start_angle, finsih_angle)
    """

    cx, cy = mean_cent
    start_angle, finish_angle = sector
    sample_num = 20
    step = 1.0*(finish_angle-start_angle) / sample_num
    widths = []
    pnt_disc = []
    for i in range(sample_num):
        angle = start_angle + step * i
        pnts, values = extract_line_at_specific_angle(disc, mean_cent, angle)
        intersect_pnt1 = get_intersect_pnts_from_line(mean_cent, pnts, values, direction)
        pnts, values = extract_line_at_specific_angle(cup, mean_cent, angle)
        intersect_pnt2 = get_intersect_pnts_from_line(mean_cent, pnts, values, direction)
        d = pnt_distance(intersect_pnt1, intersect_pnt2)
        if d > 0:
            widths.append(d)
            pnt_disc.append(intersect_pnt1)
        if img is not None:
            cv2.line(img, intersect_pnt1, intersect_pnt2, (0, 0, 255), 1)
    if len(widths)==0:
        return [0.0, 0, 0, 0, 0]
    else:
        widths = np.array(widths)
        return [np.sum(widths)/len(widths), pnt_disc[0][0], pnt_disc[0][1], pnt_disc[-1][0], pnt_disc[-1][1]] # value, x1, y1, x2, y2

def compute_edge_wid_by_eyeside(disc, cup, mean_cent, direction, eyeside, img=None):
    # For axis-based measurement, direction simply picks one of the 4 axes around center.
    return compute_edge_width_on_axis(disc, cup, mean_cent, direction, img=img)

def compute_cup_diameter_from_sector(cup, mean_cent, sector, img=None):
    """
    argx:
    sector: (start_angle, finsih_angle)
    """

    cup = np.where(cup>0, 1, 0)
    cx, cy = mean_cent
    start_angle, finish_angle = sector
    sample_num = 20
    step = 1.0*(finish_angle-start_angle) / sample_num
    widths = []
    for i in range(sample_num):
        angle = start_angle + step * i
        pnts, values = extract_line_at_specific_angle(cup, mean_cent, angle)
        pnt1, pnt2 = get_intersect_pnts_from_line(mean_cent, pnts, values, 4)
        #print('cup edge: {}, {}'.format(pnt1, pnt2))
        d = pnt_distance(pnt1, pnt2)
        if d > 0:
            widths.append(d)
        if img is not None:
            cv2.line(img, pnt1, pnt2, (0, 0, 255), 1)
    if len(widths)==0:
        return 0.0
    else:
        return np.sort(widths)[-1]

def get_edge_wid_from_range(disc_edge, cup_edge, mean_cent, eyeside, img=None, disc_full=None, cup_full=None, disc_cent=None, cup_cent=None):
    """
    Compute ISNT rim widths (using edges) and cup/disc diameters (using full masks when provided).
    - disc_edge, cup_edge: edge masks (1 where edge exists)
    - disc_full, cup_full: optional full masks; fallback to edge masks if not provided
    - disc_cent, cup_cent: centers for disc/cup diameter measurements (fallback to mean_cent)
    """
    # ISNT: rim thickness along four axes using edges
    hdl = compute_edge_wid_by_eyeside(disc_edge, cup_edge, mean_cent, 2, eyeside, img=None)
    hdr = compute_edge_wid_by_eyeside(disc_edge, cup_edge, mean_cent, 3, eyeside, img=None)
    vdu = compute_edge_wid_by_eyeside(disc_edge, cup_edge, mean_cent, 0, eyeside, img=None)
    vdd = compute_edge_wid_by_eyeside(disc_edge, cup_edge, mean_cent, 1, eyeside, img=None)

    # Diameters: prefer filled masks; if provided masks look too thin (e.g., only edges), fall back to hull/edge
    disc_mask_for_d = disc_full if disc_full is not None else disc_edge
    cup_mask_for_d = cup_full if cup_full is not None else cup_edge
    if disc_full is not None and disc_full.sum() <= disc_edge.sum():  # guard against edge-only inputs
        disc_mask_for_d, _, _, _ = filter_mask(disc_full.astype(np.uint8))
    if cup_full is not None and cup_full.sum() <= cup_edge.sum():
        cup_mask_for_d, _, _, _ = filter_mask(cup_full.astype(np.uint8))

    # Use respective centers for cup and disc diameters to avoid offsets
    cup_center = cup_cent if cup_cent is not None else mean_cent
    disc_center = disc_cent if disc_cent is not None else mean_cent

    hori_cup_d = compute_diameter_on_axis(cup_mask_for_d, cup_center, 'h', img=None)
    vert_cup_d = compute_diameter_on_axis(cup_mask_for_d, cup_center, 'v', img=None)

    hori_dd = compute_diameter_on_axis(disc_mask_for_d, disc_center, 'h', img=None)
    vert_dd = compute_diameter_on_axis(disc_mask_for_d, disc_center, 'v', img=None)
    return hdl, hdr, vdu, vdd, hori_cup_d, vert_cup_d, hori_dd, vert_dd


def get_isnt(img, disc, cup, mean_cent, visual_type, eyeside=0, disc_full=None, cup_full=None, disc_cent=None):
    # Use raw masks as edges to avoid losing thin structures
    disc_edge = np.where(disc > 0, 1, 0).astype(np.uint8)
    cup_edge = np.where(cup > 0, 1, 0).astype(np.uint8)

    if visual_type == 1:
        # draw edge
        rows, cols= np.where(disc_edge==1)
        img[rows, cols, 0] = 2
        img[rows, cols, 1] = 207
        img[rows, cols, 2] = 112

        rows, cols= np.where(cup_edge==1)
        img[rows, cols, 0] = 255
        img[rows, cols, 1] = 145
        img[rows, cols, 2] = 0

        # draw center point
        cv2.circle(img, (int(mean_cent[0]), int(mean_cent[1])), 1, (255,255,0), 3) # add
    
        # only for showing the pnts
        img = get_edge_width(img, disc_edge, cup_edge, mean_cent)

    elif visual_type == 2:
        # '''
        kernel_1 = np.ones((3,3), np.uint8)
        kernel_2 = np.ones((7,7), np.uint8)
        disc_dilate_ = cv2.dilate(disc, kernel_1, iterations=1)
        disc_erosion_ = cv2.erode(disc_dilate_, kernel_2, iterations=1)
        disc_edge_ = disc_dilate_ - disc_erosion_

        cup_dilate_ = cv2.dilate(cup, kernel_1, iterations=1)
        cup_erosion_ = cv2.erode(cup_dilate_, kernel_2, iterations=1)
        cup_edge_ = cup_dilate_ - cup_erosion_

        disc_edge_ = cv2.medianBlur(disc_edge_, 3)
        cup_edge_ = cv2.medianBlur(cup_edge_, 3)
        disc_edge_[disc_edge_>0] = 1
        cup_edge_[cup_edge_>0] = 1

        rows_, cols_ = np.where(disc_edge_==1)
        img[rows_, cols_, 0] = 2
        img[rows_, cols_, 1] = 207
        img[rows_, cols_, 2] = 112

        rows_, cols_ = np.where(cup_edge_==1)
        img[rows_, cols_, 0] = 255
        img[rows_, cols_, 1] = 145
        img[rows_, cols_, 2] = 0
        '''
        disc_rgb_mask = np.zeros(img.shape, dtype=np.uint8)
        rows_, cols_ = np.where(cup==1)
        disc_rgb_mask[rows_, cols_, 0] = 0
        disc_rgb_mask[rows_, cols_, 1] = 0
        disc_rgb_mask[rows_, cols_, 2] = 255
        rows_, cols_ = np.where(disc-cup==1)
        disc_rgb_mask[rows_, cols_, 0] = 255
        disc_rgb_mask[rows_, cols_, 1] = 0
        disc_rgb_mask[rows_, cols_, 2] = 0
        img_add = cv2.addWeighted(img, 0.7, disc_rgb_mask, 0.3, 0)
        rows_, cols_ = np.where(disc==1)
        for i in range(len(rows_)):
            img[rows_[i], cols_[i], :] = img_add[rows_[i], cols_[i], :] 
        '''
    else:
        pass

        
    # Use full masks for diameters if provided; fallback to edge masks
    disc_full_mask = disc_full if disc_full is not None else disc
    cup_full_mask = cup_full if cup_full is not None else cup
    hdl, hdr, vdu, vdd, hori_cd, vert_cd, hori_dd, vert_dd = get_edge_wid_from_range(
        disc_edge, cup_edge, mean_cent, eyeside, img=None,
        disc_full=disc_full_mask, cup_full=cup_full_mask,
        disc_cent=disc_cent
    )

    # Nasal/temporal depend on eye side.
    # Map nasal/temporal per eye side:
    # - Left eye (OS): nasal on image left, temporal on image right
    # - Right eye (OD): nasal on image right, temporal on image left
    if eyeside: # right eye
        interior = vdd
        superior = vdu
        nasal = hdr   # image right
        temporal = hdl  # image left
    else: # left eye
        interior = vdd
        superior = vdu
        nasal = hdl   # image left
        temporal = hdr  # image right

    return img, [interior, superior, nasal, temporal], [hori_cd, vert_cd, hori_dd, vert_dd]


def calculate_rect(mask):
    mask[mask>0] = 1
    rows, cols = np.where(mask == 1)
    edge_minx = np.min(cols)
    edge_maxx = np.max(cols)
    edge_miny = np.min(rows)
    edge_maxy = np.max(rows)
    rect = [edge_minx, edge_maxx, edge_miny, edge_maxy]
    return rect

def draw_edge_cent(disc_img, disc_mask, cup_mask, disc_cent, cup_cent):
    kernel = np.ones((3,3), np.uint8)
    disc_erosion = cv2.erode(disc_mask, kernel, iterations=1)
    disc_edge = disc_mask - disc_erosion

    cup_erosion = cv2.erode(cup_mask, kernel, iterations=1)
    cup_edge = cup_mask - cup_erosion

    rows, cols= np.where(disc_edge==1)
    disc_img[rows, cols, 0] = 255
    disc_img[rows, cols, 1] = 0
    disc_img[rows, cols, 2] = 0

    rows, cols= np.where(cup_edge==1)
    disc_img[rows, cols, 0] = 255
    disc_img[rows, cols, 1] = 0 
    disc_img[rows, cols, 2] = 0

    # draw center point
    cv2.circle(disc_img, (int(disc_cent[0]), int(disc_cent[1])), 1, (255,255,0), 3) # add
    cv2.circle(disc_img, (int(cup_cent[0]), int(cup_cent[1])), 1, (255,255,0), 3) # add

    return disc_img

def get_intersect_pnt_from_angle(mask, cent, angle, rect):
    """
    args:
        input:
            mask: pixel value = 1, channel = 2
            cent: centroid point [x,y]
            angle: np.tan(angle)
            rect: mask edge point
        output:
            pnts: [left, right, distance]
    """
    new_mask = copy.deepcopy(mask)
    edge_minx,edge_maxx,edge_miny,edge_maxy = rect
    cx,cy = cent
    pnt = []
    if angle < np.pi/4:
        k = np.tan(angle)
        for x in range(edge_minx, edge_maxx+1, 1):
            if x < cx:
                y = int(cy + k * (cx - x))
            else:
                y = int(cy - k * (x - cx))
            if y <= edge_maxy and y >= edge_miny:
                pnt.append([x,y])

    elif angle >= np.pi/4 and angle < np.pi/2:
        k = np.tan(np.pi/2-angle)
        for y in range(edge_miny, edge_maxy+1, 1):
            if y > cy:
                x = int(cx - k * (y-cy))
            else:
                x = int(cx + k * (cy - y))
            if x <= edge_maxx and x >= edge_minx:
                pnt.append([x,y])

    elif angle >= np.pi/2 and angle < 3*np.pi/4:
        k = np.tan(angle-np.pi/2)
        for y in range(edge_miny, edge_maxy+1, 1):
            if y > cy:
                x = int(k * (y-cy) + cx)
            else:
                x = int(cx - k * (cy - y))
            if x <= edge_maxx and x >= edge_minx:
                pnt.append([x,y])

    else:
        k = np.tan(np.pi-angle)
        for x in range(edge_minx, edge_maxx+1, 1):
            if x > cx:
                y = int(k * (x-cx) + cy)
            else:
                y = int(cy - k * (cx - x))
            if y <= edge_maxy and y >= edge_miny:
                pnt.append([x,y])
    
    # calculate coordinate of intersect
    left = [0,0]
    right = [0,0]
    angle_ = angle%np.pi/np.pi*180
    for d in pnt:
        if (angle_ >= 45 and angle_ < 135) or (angle_-180 >= 45 and angle_-180 < 135):
            if mask[d[1],d[0]] == 1:
                if d[1] < cy:
                    if left == [0,0]:
                        left = [d[0], d[1]]
                    else:
                        pass
                else:
                    if right == [0,0]:
                        right = [d[0], d[1]]
                    else:
                        pass
        else:
            if mask[d[1],d[0]] == 1:
                if d[0] < cx:
                    if left == [0,0]:
                        left = [d[0], d[1]]
                    else:
                        pass
                else:
                    if right == [0,0]:
                        right = [d[0], d[1]]
                    else:
                        pass

    if left == [0,0] or right == [0,0]:
        dist = 0.0
    else:
        dist = pnt_distance(left, right)
    
    return [left,right, dist]


def compute_longest_diam(disc, mean_cent, sector):
    new_disc = copy.deepcopy(disc)
    """
    argx:
    sector: (start_angle, finsih_angle)
    """
    rect = calculate_rect(disc)
    
    start_angle, finish_angle = sector
    sample_num = 40
    step = 1.0*(finish_angle-start_angle) / sample_num
    max_dist = [0,0,0,0] # distance , left-point, right-point, angle
    for i in range(sample_num):
        angle = start_angle + step * i   
        coord = []
        left,right, dist = get_intersect_pnt_from_angle(new_disc, mean_cent, angle, rect)
        if dist > max_dist[0]:
            max_dist = [dist, left, right, angle]
            
    # # draw for show
    # img = np.zeros((new_disc.shape[0], new_disc.shape[1], 3),np.uint8)
    # cv2.line(img, tuple(max_dist[1]), tuple(max_dist[2]), (0, 0, 255), 1)
    # print(max_dist[3])
    
    return max_dist

def computer_ratio(disc_boundary, disc_centroid, cup_boundary, cup_centroid):
    sector = (np.pi/4, 3*np.pi/4)
    # dvert_list = [distance , left-point, right-point, angle]
    dvert_list = compute_longest_diam(disc_boundary, disc_centroid, sector)
    c_rect = calculate_rect(cup_boundary)
    # cvert_list = [left,right, distance]
    cvert_list = get_intersect_pnt_from_angle(cup_boundary, cup_centroid, dvert_list[3], c_rect)

    # calculate disc hori intersect 
    d_rect = calculate_rect(disc_boundary)
    # dhori_list = [left,right, distance]
    dhori_list = get_intersect_pnt_from_angle(disc_boundary, disc_centroid, (dvert_list[3]+np.pi/2), d_rect)

    return cvert_list, dvert_list, dhori_list


def draw_text(img_gt_show, mean_cent, cd_list, eyeside=0):

    gt_hori_cd, gt_vert_cd, gt_hori_dd, gt_vert_dd = cd_list

    cv2.putText(img_gt_show, 'i', (mean_cent[0], int(mean_cent[1]+gt_vert_dd/2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1) # interior
    cv2.putText(img_gt_show, 's', (mean_cent[0], int(mean_cent[1]-gt_vert_dd/2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1) # superior

    # nasal/temporal swap depending on eye side
    if eyeside==0:
        # left eye: nasal at image left, temporal at image right
        cv2.putText(img_gt_show, 'n', (int(mean_cent[0]-gt_hori_dd/2) - 30, mean_cent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1)
        cv2.putText(img_gt_show, 't', (int(mean_cent[0]+gt_hori_dd/2) + 20, mean_cent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1)
    else:
        # right eye: nasal at image right, temporal at image left
        cv2.putText(img_gt_show, 'n', (int(mean_cent[0]+gt_hori_dd/2) + 20, mean_cent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1)
        cv2.putText(img_gt_show, 't', (int(mean_cent[0]-gt_hori_dd/2) - 30, mean_cent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1)

    return img_gt_show

def draw_isnt(img_gt_show, mean_cent, isnt_list, eyeside=1):
    interior, superior, nasal, temporal = isnt_list
    if eyeside==0:
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((interior[1] + nasal[3])/2),int((interior[2] + nasal[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((nasal[1] + superior[3])/2),int((nasal[2] + superior[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((temporal[1] + interior[3])/2),int((temporal[2] + interior[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((superior[1] + temporal[3])/2),int((superior[2] + temporal[4])/2)), (255, 255, 255), 1)
    else:
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((interior[1] + temporal[3])/2),int((interior[2] + temporal[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((temporal[1] + superior[3])/2),int((temporal[2] + superior[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((nasal[1] + interior[3])/2),int((nasal[2] + interior[4])/2)), (255, 255, 255), 1)
        cv2.line(img_gt_show, (mean_cent[0],mean_cent[1]), (int((superior[1] + nasal[3])/2),int((superior[2] + nasal[4])/2)), (255, 255, 255), 1)
    return img_gt_show

def filter_mask(img):
    if cv2.__version__.startswith('3'):
        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    point_num_list = []
    for i in range(len(contours)):
        point_num_list.append(contours[i].shape[0])
    if len(point_num_list)==0:
        return img, img, (0,0), 0
    contours = [contours[np.argmax(point_num_list)]]
    # print (contours[0].shape)
    
    new_img = np.zeros_like(img)
    hull = cv2.convexHull(contours[0])
    cv2.fillPoly(new_img, [hull], 255)

    # get the boundary
    kernel = np.ones((7,7), np.uint8)
    new_mask = np.zeros((new_img.shape[0]+10, new_img.shape[1]+10), dtype=np.uint8)
    new_mask[5:-5,5:-5] = new_img
    erosion = cv2.erode(new_mask, kernel, iterations=1)
    boundary = new_mask - erosion
    boundary_ = np.zeros(new_img.shape, dtype=np.uint8)
    boundary_[:,:] = boundary[5:-5,5:-5]
    boundary_[boundary_>0] = 1
    #cv2.imwrite('boundary.png', boundary)
    rows, cols = np.where(boundary_==1)
    points = np.zeros((len(rows), 2), int)
    points[:,0] = cols
    points[:,1] = rows

    num = points.shape[0]
    cent_x = int(points[:, 0].sum() / num)
    cent_y = int(points[:, 1].sum() / num)
    centroid = (cent_x, cent_y)
    new_img[new_img>0] = 1
    assert(new_img.shape == boundary_.shape)
 
    area_px = int(new_img.sum())
    return new_img, boundary_, centroid, area_px

def draw_isnt_and_ratio(show_img, eyeside, isnt_list, ratio, vh_ratio, angle, cd_list, flag_isnt, disc_cent):
    interior, superior, nasal, temporal = isnt_list
    hori_cd, vert_cd, hori_dd, vert_dd = cd_list

    # --------------- text ---------------------
    sides = ['OS', 'OD']
    put_text1 = '[{}]'.format(sides[eyeside])
    put_text2 = 'i {}, s {}, n {}, t {}'.format(int(interior[0]), int(superior[0]), int(nasal[0]), int(temporal[0]))
    put_text3 = 'A: {},  B: {}'.format(int(hori_cd), int(vert_cd))
    put_text4 = 'Eccent(Dh/Dv): {}'.format(vh_ratio)
    put_text5 = 'angle: {}'.format(angle)
    # put_text6 = 'ratio(C/D): {:6.4f}'.format(ratio - 0.015)
    put_text6 = 'ratio(C/D): {:6.4f}'.format(ratio)
    #-------------------------------------------
    text_xmin = disc_cent[0] - 50
    # text_ymin = 90
    text_ymin = 60
    text_ystep = 40

    #if flag_isnt is True -> draw isnt and position
    if flag_isnt:  
        cv2.putText(show_img, put_text1, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
        cv2.putText(show_img, put_text2, (30, show_img.shape[0]-200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(show_img, put_text3, (30, show_img.shape[0]-160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(show_img, put_text4, (30, show_img.shape[0]-120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(show_img, put_text5, (30, show_img.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(show_img, put_text6, (30, show_img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    else:
        # cv2.putText(show_img, put_text3, (text_xmin, show_img.shape[0]-text_ymin-4*text_ystep), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # cv2.putText(show_img, put_text4, (text_xmin, show_img.shape[0]-text_ymin-3*text_ystep), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # cv2.putText(show_img, put_text5, (text_xmin, show_img.shape[0]-text_ymin-2*text_ystep), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(show_img, put_text6, (text_xmin, show_img.shape[0]-text_ymin-1*text_ystep), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    return show_img

def auto_detect_eyeside(image, disc_centroid, cup_centroid, macula_center=None):
    """
    Automatically detect eye side based on the relative position of optic disc and macula.
    
    Args:
        image: Input fundus image (numpy array, shape: H x W x 3)
        disc_centroid: Disc center coordinates (tuple: (x, y))
        cup_centroid: Cup center coordinates (tuple: (x, y))
        macula_center: Macula center coordinates (tuple: (x, y)), optional
    
    Returns:
        int: 0 for left eye (OS), 1 for right eye (OD)
    
    Logic:
        - If macula_center is provided: Use relative position of disc and macula
          - In left eye: macula is temporal to disc (macula_x > disc_x)
          - In right eye: macula is temporal to disc (macula_x < disc_x)
        - If macula_center is not provided: Fall back to image position method
    """
    if disc_centroid == (0, 0) or cup_centroid == (0, 0):
        # Default to right eye if centroids are invalid
        return 1
    
    # If macula center is provided, use anatomical relationship
    if macula_center is not None and macula_center != (0, 0):
        disc_x = disc_centroid[0]
        macula_x = macula_center[0]
        
        # Anatomical relationship:
        # - In left eye: macula is temporal to disc (macula_x > disc_x)
        # - In right eye: macula is temporal to disc (macula_x < disc_x)
        if macula_x > disc_x:
            return 0  # Left eye (OS)
        else:
            return 1  # Right eye (OD)
    
    # Fallback to image position method if macula center is not available
    # Calculate mean center of disc and cup
    mean_center_x = (disc_centroid[0] + cup_centroid[0]) / 2
    
    # Get image width
    image_width = image.shape[1]
    
    # Determine eye side based on horizontal position
    # If the optic disc/cup center is in the left half of the image, it's a left eye
    if mean_center_x < image_width / 2:
        return 0  # Left eye (OS)
    else:
        return 1  # Right eye (OD)

def visualize_glaucoma(image, mask, eyeside_value=None, flag_isnt=True, visual_type=1, macula_center=None):
    """
    Analyze glaucoma features from fundus image and segmentation mask.
    
    Args:
        image: RGB fundus image (numpy array, shape: H x W x 3)
        mask: Segmentation mask (numpy array, shape: H x W, values: 0=background, 1=disc, 2=cup)
        eyeside_value: Eye side indicator (int, 0 for left eye, 1 for right eye, None for auto-detection)
        flag_isnt: Whether to draw ISNT regions (bool, default: True)
        visual_type: Visualization type (int, 0=no visualization, 1=basic, 2=detailed, default: 1)
        macula_center: Macula center coordinates (tuple: (x, y)), optional for improved eye side detection
    
    Returns:
        dict: Analysis results with keys:
            - 'image': Processed image with annotations
            - 'cd_ratio': Cup-to-disc ratio
            - 'vh_ratio': Vertical-to-horizontal ratio
            - 'angle_str': Angle measurement
            - 'isnt_list': ISNT measurements
            - 'cd_list': Cup and disc diameter measurements
            - 'eyeside_detected': Auto-detected eye side
    """
    # Extract disc and cup masks from the combined mask
    # Disc: all pixels > 0 (includes both disc and cup regions)
    # Cup: only pixels == 2
    disc_mask = np.zeros_like(mask)
    disc_mask[mask > 0] = 1
    
    cup_mask = np.zeros_like(mask)
    cup_mask[mask == 2] = 1
    
    disc_hull, disc_boundary, disc_centroid, disc_area_px = filter_mask(disc_mask)  # disc mask convex 
    cup_hull, cup_boundary, cup_centroid, cup_area_px = filter_mask(cup_mask)  # cup mask convex 
    rows_, cols_ = np.where((disc_hull-cup_hull)==255)

    if len(rows_) > 0:
        return {
            'image': None,
            'cd_ratio': 0,
            'vh_ratio': 0,
            'angle_str': '',
            'isnt_list': [],
            'cd_list': [],
            'eyeside_detected': None
        }
    if disc_centroid == (0,0) or cup_centroid == (0,0):
        return {
            'image': None,
            'cd_ratio': 0,
            'vh_ratio': 0,
            'angle_str': '',
            'isnt_list': [],
            'cd_list': [],
            'eyeside_detected': None
        }

    if eyeside_value is None:
        eyeside_value = auto_detect_eyeside(image, disc_centroid, cup_centroid, macula_center)

    # ISNT/CD uses disc center as reference
    mean_cent = (int(disc_centroid[0]), int(disc_centroid[1]))

    # calculate isnt
    # output:
    #   - image : had draw disc and cup edge, also AB line
    #   - isnt_list = [interior, superior, nasal, temporal]
    #   - cd_list = [hori_cd, vert_cd, hori_dd, vert_dd]
    image, isnt_list, cd_list = get_isnt(
        image,
        disc_hull,
        cup_hull,
        mean_cent,
        visual_type,
        eyeside=eyeside_value,
        disc_full=disc_mask,
        cup_full=cup_mask,
        disc_cent=disc_centroid
    )

    # Recompute cup diameters using cup center to reduce offset bias
    cup_hori_d = compute_diameter_on_axis(cup_hull, cup_centroid, 'h') if cup_centroid != (0, 0) else 0.0
    cup_vert_d = compute_diameter_on_axis(cup_hull, cup_centroid, 'v') if cup_centroid != (0, 0) else 0.0
    cd_list = [cup_hori_d, cup_vert_d, cd_list[2], cd_list[3]]

    cvert_list, dvert_list, dhori_list = computer_ratio(disc_boundary, disc_centroid, cup_boundary, cup_centroid)
    if dvert_list[0] > 0 and cvert_list[2] > 0 and dvert_list[0] > cvert_list[2]:
        cd_ratio = round(max(float(cvert_list[2])/dvert_list[0], 0.0), 4)
        vh_ratio = round(float(dhori_list[2])/dvert_list[0], 2)
    else:
        cd_ratio = 0.0
        vh_ratio = 0.0
    angle = 90 - float(round(dvert_list[3]/np.pi*180,2))
    angle_str = '+'+str(angle) if angle > 0 else str(angle)

    # draw ratio and isnt text and line
    if visual_type > 0:
        image = draw_isnt_and_ratio(image, eyeside_value, isnt_list, cd_ratio, vh_ratio, angle_str, cd_list, flag_isnt, disc_centroid) # draw ratio text
        if flag_isnt:
            image = draw_isnt(image, mean_cent, isnt_list, eyeside=eyeside_value) # draw i, s, n, t region
            image = draw_text(image, mean_cent, cd_list, eyeside=eyeside_value) # draw isnt text
    
    return {
        'image': image,
        'cd_ratio': cd_ratio,
        'vh_ratio': vh_ratio,
        'angle_str': angle_str,
        'isnt_list': isnt_list,
        'cd_list': cd_list,
        'disc_area_px': disc_area_px,
        'cup_area_px': cup_area_px,
        'eyeside_detected': eyeside_value
    }
