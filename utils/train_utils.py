import math
import os
import shutil

import cv2
import numpy as np
import torch
import torch.optim as optim
import copy
from skimage import morphology
import scipy
from skimage.measure import label
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import time


def save_checkpoint(
    state, is_best, fold, savename, epoch, filename="model_checkpoint.pth.tar"
):
    dirname = "{}".format(savename)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        print("Saving checkpoint {} as the best model...".format(epoch))
        shutil.copyfile(
            os.path.join(dirname, filename),
            "{}/model_best_{}.pth.tar".format(savename, str(fold)),
        )


def adjust_learning_rate(optimizer, epoch, epochs, lr, cos=True, schedule=None):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_alpha(epoch, epochs, min_alpha=0.2):
    step = (1 - min_alpha) / epochs
    return 1 - epoch * step


def make_optimizer(args):
    """make optimizer"""
    # optimizer
    kwargs_optimizer = {"lr": args.lr}

    if args.optimizer == "sgd":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = args.momentum
    elif args.optimizer == "adam":
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8
    elif args.optimizer == "rmsprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = args.epsilon
    elif args.optimizer == "adamw":
        optimizer_class = optim.AdamW
        kwargs_optimizer["lr"] = args.lr
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8
        kwargs_optimizer["weight_decay"] = args.weight_decay
    else:
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8

    return optimizer_class, kwargs_optimizer


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_soft_label(input_tensor, num_class):
    """
    convert a label tensor to soft label
    input_tensor: tensor with shape [N, C, H, W]
    output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of a binary image

    Parameters:
    ---

    bw_img: ndarray
        binary image

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, background=0, return_num=True)

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    lcc = scipy.ndimage.binary_fill_holes(lcc).astype(int)

    return lcc


def make_skeleton(mask):
    # foreground mask
    vessel_mask = np.zeros_like(mask)
    vessel_mask[mask > 0] = 1

    # skeleton
    skeleton = morphology.skeletonize(vessel_mask)
    skeleton = skeleton.astype(np.uint8)
    skeleton[skeleton > 0] = 1

    return skeleton


def create_skeleton(mask):
    """
    function: create skeleton and split it's branches.

    input:
        mask: 2-D numpy array. arteriovenous mask. pixel value: 1 -> artery, 2 -> vein

    output:
        skeleton: 2-D numpy array. skeleton of vessel mask
        skeleton_branch: 2-D numpy array. split by intersection point and get single branch line.
    """
    skeleton = make_skeleton(mask)

    skeleton_branch = copy.deepcopy(skeleton)

    # filter kernel
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


def filter_process(mask, thres):
    """
    filter out regions less than the threshold.
    """
    filter_mask = np.zeros_like(mask)
    ret, markers = cv2.connectedComponents(mask)
    if ret > 1:
        for i in range(1, ret):
            tmp_mask = np.zeros_like(mask)
            tmp_mask[markers == i] = 1
            if np.sum(tmp_mask) < thres:
                filter_mask[markers == i] = 1
    return filter_mask


def filter_noise_process(orig_mask, mask, thres):
    """
    filter out regions less than the threshold.
    """
    assert(orig_mask.shape == mask.shape)
    filter_mask = np.zeros_like(mask)
    ret, markers = cv2.connectedComponents(mask)

    # get the boundary
    kernel = np.ones((3, 3), np.uint8)
    if ret > 1:
        for i in range(1, ret):
            tmp_mask = np.zeros_like(mask)
            tmp_mask[markers == i] = 1
            if np.sum(tmp_mask) < thres:
                dilation = cv2.dilate(tmp_mask, kernel, iterations=1)
                boundary = dilation - tmp_mask
                assert(np.max(boundary) == 1)
                rows, cols = np.where(boundary == 1)
                if 0 in orig_mask[rows, cols].tolist():
                    filter_mask[markers == i] = 1
    return filter_mask


def filter_small_region(mask, isolated_thres=500, remove_thres=50):
    """
    filter out small region of artery and vein respectively.
    input:
        mask: 2-D numpy array. arteriovenous mask. pixel value: 1 -> artery, 2 -> vein
    output:
        isolated_mask: 2-D numpy array. isolated region both of artery and vein.
        remove_mask: 2-D numpy array. small region which to remove.
    """
    a_mask = np.zeros_like(mask)
    v_mask = np.zeros_like(mask)

    a_mask[mask == 1] = 1
    v_mask[mask == 2] = 1

    a_isolated = filter_process(a_mask, isolated_thres)
    v_isolated = filter_process(v_mask, isolated_thres)

    isolated_mask = a_isolated + v_isolated
    isolated_mask[isolated_mask > 0] = 1

    remove_mask = filter_noise_process(mask, isolated_mask, remove_thres)

    isolated_mask[remove_mask > 0] = 0

    return isolated_mask, remove_mask


def adjust_region(artven_mask, skeleton, isolated_mask):
    """postprocess isolated region"""

    """find skeleton endpoints"""
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0
    corner_mask = cv2.filter2D(skeleton, -1, kernel)

    new_prob_mask = np.zeros_like(isolated_mask)
    ret_reg, reg_markers = cv2.connectedComponents(isolated_mask)
    ret_ske, ske_markers = cv2.connectedComponents(skeleton)
    if ret_reg > 1:
        """if existed foreground regions"""
        for i in range(1, ret_reg):
            tmp_mask = np.zeros_like(isolated_mask)
            tmp_mask[reg_markers == i] = 1

            tmp_prob_mask = tmp_mask * artven_mask

            pixel_1_num = np.sum(tmp_prob_mask == 1)
            pixel_2_num = np.sum(tmp_prob_mask == 2)

            rows_ske, cols_ske = np.where(tmp_mask * skeleton)

            ske_pixel_list = list(set(ske_markers[rows_ske, cols_ske].tolist()))

            assert (0 not in ske_pixel_list)

            if len(ske_pixel_list) == 1:
                """normal situation"""
                ret_ske_value = ske_pixel_list[0]
                tmp_ske_mask = np.zeros_like(isolated_mask)
                tmp_ske_mask[ske_markers == ret_ske_value] = 1

                rows_comp_ske, cols_comp_ske = np.where(tmp_ske_mask)
                skeleton_set = list(set(artven_mask[rows_comp_ske, cols_comp_ske].tolist()))

                # situation 1: {pixel_set = 1, skeleton_set = 1}
                if pixel_1_num * pixel_2_num == 0 and len(skeleton_set) == 1:

                    # stay unchange
                    new_prob_mask += tmp_prob_mask

                # situation 2: {pixel_set = 2, skeleton_set = 1}
                elif pixel_1_num * pixel_2_num > 0 and len(skeleton_set) == 1:
                    max_value = skeleton_set[0]
                    new_prob_mask[tmp_mask == 1] = max_value

                # situation 3: {pixel_set = 1, skeleton_set = 2}
                elif pixel_1_num * pixel_2_num == 0 and len(skeleton_set) == 2:
                    rate = 1 - float(len(rows_ske)) / len(rows_comp_ske)

                    # the expand length less than 10% -> stay unchange
                    if rate < 0.1:
                        new_prob_mask[tmp_mask == 1] = 1 if pixel_1_num > 0 else 2

                    # the expand length more than 10% -> reverse
                    else:
                        new_prob_mask[tmp_mask == 1] = 2 if pixel_1_num > 0 else 1

                # situation 4: {pixel_set = 2, skeleton_set = 2}
                elif pixel_1_num * pixel_2_num > 0 and len(skeleton_set) == 2:
                    endponits_list = list(set(artven_mask[np.where(corner_mask * tmp_ske_mask == 1)].tolist()))

                    # endpoints is unique
                    if len(endponits_list) == 1:
                        new_prob_mask[tmp_mask == 1] = endponits_list[0]
                    else:
                        ske_comp_list = artven_mask[rows_comp_ske, cols_comp_ske].tolist()
                        new_prob_mask[tmp_mask == 1] = 1 if ske_comp_list.count(1) > ske_comp_list.count(2) else 2
            else:
                """abnormal situation"""
                new_prob_mask[tmp_mask == 1] = 1 if np.sum(tmp_prob_mask == 1) > np.sum(tmp_prob_mask == 2) else 2

    return new_prob_mask


def create_gaussian_importance_map(patch_size, sigma_scale=1./8):
    """
    Create a Gaussian importance map for patch-based inference.
    Similar to nnU-Net's implementation.
    
    Args:
        patch_size (int or tuple): Size of the patch (assumed square if int)
        sigma_scale (float): Scale factor for sigma calculation
        
    Returns:
        numpy.ndarray: Gaussian importance map of shape (patch_size, patch_size)
    """
    if isinstance(patch_size, int):
        tmp = np.zeros((patch_size, patch_size))
        center_coords = [patch_size // 2, patch_size // 2]
        sigmas = [patch_size * sigma_scale, patch_size * sigma_scale]
    else:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
    
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = scipy.ndimage.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    
    # Ensure minimum weight to avoid division by zero
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])
    
    return gaussian_importance_map


def sliding_window_evaluation(model, image, patch_size=256, stride_factor=0.5, 
                            num_classes=2, device='cuda', normalize_params=None):
    """
    Perform sliding window evaluation on a full-resolution image.
    Based on nnU-Net's sliding window inference strategy.
    
    Args:
        model: The trained model for inference
        image (numpy.ndarray): Input image of shape (H, W, C) 
        patch_size (int): Size of sliding window patches
        stride_factor (float): Stride as a fraction of patch_size (0.5 = 50% overlap)
        num_classes (int): Number of output classes (including background)
        device (str): Device to run inference on
        normalize_params (dict): Normalization parameters with 'mean' and 'std' keys
        
    Returns:
        numpy.ndarray: Predicted segmentation mask of shape (H, W, num_classes)
    """
    model.eval()
    
    # Default normalization parameters if not provided
    if normalize_params is None:
        normalize_params = {
            'mean': [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
            'std': [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
        }
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate stride
    stride = int(patch_size * stride_factor)
    
    # Calculate number of patches needed in each dimension
    num_patches_h = int(np.ceil((img_height - patch_size) / stride)) + 1 if img_height > patch_size else 1
    num_patches_w = int(np.ceil((img_width - patch_size) / stride)) + 1 if img_width > patch_size else 1
    
    # Initialize output arrays
    prediction_sum = np.zeros((img_height, img_width, num_classes), dtype=np.float32)
    weight_sum = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Create Gaussian importance map
    gaussian_map = create_gaussian_importance_map(patch_size)
    
    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_params['mean'], std=normalize_params['std'])
    ])
    
    print(f"Processing {num_patches_h}x{num_patches_w} = {num_patches_h * num_patches_w} patches...")
    
    with torch.no_grad():
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                start_h = i * stride
                start_w = j * stride
                
                # Ensure patch doesn't exceed image boundaries
                end_h = min(start_h + patch_size, img_height)
                end_w = min(start_w + patch_size, img_width)
                start_h = max(end_h - patch_size, 0)
                start_w = max(end_w - patch_size, 0)
                
                # Extract patch
                patch = image[start_h:end_h, start_w:end_w]
                
                # Handle edge cases where patch is smaller than expected
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    # Pad patch to required size
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                
                # Convert to PIL and preprocess
                patch_pil = Image.fromarray(patch.astype(np.uint8))
                patch_tensor = preprocess(patch_pil).unsqueeze(0).to(device)
                
                # Model inference
                with torch.cuda.amp.autocast():
                    prediction = model(patch_tensor)
                
                # Handle multi-task output (assume we want the first task for vessel segmentation)
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]  # Take first task output
                
                # Apply softmax to get probabilities
                prediction = F.softmax(prediction, dim=1)
                prediction = prediction.squeeze(0).cpu().numpy()  # Shape: (num_classes, H, W)
                prediction = prediction.transpose(1, 2, 0)  # Shape: (H, W, num_classes)
                
                # Remove padding if it was added
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    original_h = end_h - start_h
                    original_w = end_w - start_w
                    prediction = prediction[:original_h, :original_w]
                    current_gaussian_map = gaussian_map[:original_h, :original_w]
                else:
                    current_gaussian_map = gaussian_map
                
                # Add weighted prediction to output
                for c in range(num_classes):
                    prediction_sum[start_h:end_h, start_w:end_w, c] += (
                        prediction[:, :, c] * current_gaussian_map
                    )
                
                # Add weights
                weight_sum[start_h:end_h, start_w:end_w] += current_gaussian_map
    
    # Normalize by weights to get final prediction
    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1
    
    final_prediction = np.zeros_like(prediction_sum)
    for c in range(num_classes):
        final_prediction[:, :, c] = prediction_sum[:, :, c] / weight_sum
    
    # print("Sliding window evaluation completed.")
    
    return final_prediction


def sliding_window_validation(model, image, ground_truth, patch_size=256, stride_factor=0.5,
                            num_classes=2, device='cuda', normalize_params=None):
    """
    Perform sliding window validation with metrics calculation.
    
    Args:
        model: The trained model for inference
        image (numpy.ndarray): Input image of shape (H, W, C)
        ground_truth (numpy.ndarray): Ground truth mask of shape (H, W) or (H, W, C)
        patch_size (int): Size of sliding window patches
        stride_factor (float): Stride as a fraction of patch_size
        num_classes (int): Number of output classes
        device (str): Device to run inference on
        normalize_params (dict): Normalization parameters
        
    Returns:
        dict: Dictionary containing prediction, ground_truth, and calculated metrics
    """
    # Get prediction using sliding window
    prediction = sliding_window_evaluation(
        model, image, patch_size, stride_factor, num_classes, device, normalize_params
    )
    
    # Convert prediction to class labels
    pred_labels = np.argmax(prediction, axis=-1)
    
    # Prepare ground truth
    if len(ground_truth.shape) == 3:
        # If ground truth has channels, take argmax or appropriate conversion
        if ground_truth.shape[-1] > 1:
            gt_labels = np.argmax(ground_truth, axis=-1)
        else:
            gt_labels = ground_truth.squeeze(-1)
    else:
        gt_labels = ground_truth
    
    # Calculate basic metrics
    results = {
        'prediction': prediction,
        'pred_labels': pred_labels,
        'ground_truth': gt_labels,
        'image_shape': image.shape,
        'num_patches': prediction.shape
    }
    
    return results


def create_gaussian_importance_map(patch_size):
    """
    Create a 2D Gaussian importance map for patch weighting.
    Center pixels have higher weight, edge pixels have lower weight.
    """
    # Create coordinate grids
    center = patch_size // 2
    sigma = patch_size / 6  # Smaller sigma for smoother falloff
    
    y, x = np.ogrid[:patch_size, :patch_size]
    
    # Calculate Gaussian weights
    gaussian_map = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Normalize to [0.1, 1] to avoid zero weights at edges
    gaussian_map = gaussian_map / gaussian_map.max()
    gaussian_map = 0.1 + 0.9 * gaussian_map  # Scale to [0.1, 1]
    
    return gaussian_map


def process_precut_patches_validation(model, patches_tensor, targets_tensor, metadata, device='cuda'):
    """
    Process pre-cut sliding window patches for validation.
    
    Args:
        model: The trained model for inference
        patches_tensor (torch.Tensor): Pre-cut patches of shape (N_patches, C, H, W)
        targets_tensor (torch.Tensor): Target masks of shape (N_patches, 2, H, W)
        metadata (dict): Contains positions, original_shape, patch_size, stride_factor, name
        device (str): Device to run inference on
        
    Returns:
        dict: Dictionary containing reconstructed prediction, ground_truth, and metadata
    """
    model.eval()
    
    # Move patches to device
    patches_tensor = patches_tensor.to(device)
    targets_tensor = targets_tensor.to(device)
    
    # Create Gaussian importance map
    patch_size = metadata['patch_size']
    gaussian_map = create_gaussian_importance_map(patch_size)
    
    # Optional: Save Gaussian map for visualization (first time only)
    # if not hasattr(process_precut_patches_validation, '_gaussian_saved'):
    #     try:
    #         import matplotlib.pyplot as plt
    #         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #         im = ax.imshow(gaussian_map, cmap='viridis')
    #         ax.set_title(f'Gaussian Importance Map ({patch_size}x{patch_size})')
    #         plt.colorbar(im)
    #         plt.savefig('/tmp/gaussian_importance_map.png', dpi=150, bbox_inches='tight')
    #         plt.close()
    #         print(f"Gaussian importance map saved to /tmp/gaussian_importance_map.png")
    #         process_precut_patches_validation._gaussian_saved = True
    #     except:
    #         pass  # Ignore if matplotlib not available
    
    # Run inference on all patches
    with torch.no_grad():
        # Process patches in batches to avoid memory issues
        batch_size = 8  # Adjust based on GPU memory
        patch_predictions = []
        
        for i in range(0, patches_tensor.shape[0], batch_size):
            batch_patches = patches_tensor[i:i+batch_size]
            batch_preds = model(batch_patches)
            
            # Convert to probabilities
            if batch_preds.shape[1] == 2:  # Binary case
                batch_probs = torch.sigmoid(batch_preds)
            else:  # Multi-class case
                batch_probs = torch.softmax(batch_preds, dim=1)
            
            patch_predictions.append(batch_probs.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(patch_predictions, dim=0)  # (N_patches, C, H, W)
    
    # Reconstruct full image from patches
    positions = metadata['positions']
    original_shape = metadata['original_shape']
    patch_size = metadata['patch_size']
    
    h, w = original_shape
    num_classes = all_predictions.shape[1]
    
    # Initialize prediction accumulation arrays
    prediction_sum = np.zeros((h, w, num_classes), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    # Accumulate predictions with Gaussian weights
    for i, (top, left) in enumerate(positions):
        bottom = top + patch_size
        right = left + patch_size
        
        # Handle boundary cases - ensure we don't go outside image bounds
        img_top = max(0, top)
        img_bottom = min(h, bottom)
        img_left = max(0, left)
        img_right = min(w, right)
        
        # Calculate corresponding patch coordinates
        patch_top = img_top - top
        patch_bottom = patch_top + (img_bottom - img_top)
        patch_left = img_left - left
        patch_right = patch_left + (img_right - img_left)
        
        # Extract the valid region from patch prediction
        patch_pred = all_predictions[i].numpy().transpose(1, 2, 0)  # (H, W, C)
        valid_patch_pred = patch_pred[patch_top:patch_bottom, patch_left:patch_right]
        valid_gaussian_map = gaussian_map[patch_top:patch_bottom, patch_left:patch_right]
        
        # Apply Gaussian weighting to the valid patch prediction
        gaussian_weighted_pred = valid_patch_pred * valid_gaussian_map[..., np.newaxis]
        
        # Add to accumulation with Gaussian weights (only valid regions)
        prediction_sum[img_top:img_bottom, img_left:img_right] += gaussian_weighted_pred
        weight_sum[img_top:img_bottom, img_left:img_right] += valid_gaussian_map
    
    # Average overlapping regions with Gaussian weights
    prediction_sum = prediction_sum / (weight_sum[..., np.newaxis] + 1e-8)
    
    # Debug: Check weight coverage
    uncovered_pixels = np.sum(weight_sum < 0.01)  # Pixels with very low weight
    total_pixels = h * w
    if uncovered_pixels > 0:
        print(f"Warning: {uncovered_pixels}/{total_pixels} pixels have low weight coverage")
    
    # Reconstruct ground truth from patches (just take the first occurrence)
    gt_reconstruction = np.zeros((h, w, 2), dtype=np.uint8)
    gt_mask = np.zeros((h, w), dtype=bool)
    
    for i, (top, left) in enumerate(positions):
        bottom = top + patch_size
        right = left + patch_size
        
        # Handle boundary cases for ground truth
        img_top = max(0, top)
        img_bottom = min(h, bottom)
        img_left = max(0, left)
        img_right = min(w, right)
        
        # Calculate corresponding patch coordinates
        patch_top = img_top - top
        patch_bottom = patch_top + (img_bottom - img_top)
        patch_left = img_left - left
        patch_right = patch_left + (img_right - img_left)
        
        # Only fill areas that haven't been filled yet (first occurrence)
        patch_target = targets_tensor[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
        valid_patch_target = patch_target[patch_top:patch_bottom, patch_left:patch_right]
        
        mask = ~gt_mask[img_top:img_bottom, img_left:img_right]
        
        gt_reconstruction[img_top:img_bottom, img_left:img_right][mask] = valid_patch_target[mask]
        gt_mask[img_top:img_bottom, img_left:img_right] = True
    
    # For binary format: apply threshold to get binary masks
    # prediction_sum is already averaged probabilities
    pred_artery = (prediction_sum[:, :, 0] > 0.5).astype(np.uint8)  # Channel 0: artery
    pred_vein = (prediction_sum[:, :, 1] > 0.5).astype(np.uint8)    # Channel 1: vein
    
    # Ground truth is already binary (0 or 1)
    gt_artery = gt_reconstruction[:, :, 0]  # Channel 0: artery  
    gt_vein = gt_reconstruction[:, :, 1]     # Channel 1: vein
    
    results = {
        'prediction': prediction_sum,
        'pred_artery': pred_artery,
        'pred_vein': pred_vein,
        'gt_artery': gt_artery,
        'gt_vein': gt_vein,
        'original_shape': original_shape,
        'name': metadata['name'],
        'num_patches': len(positions)
    }
    
    return results


# ============================== Loss Functions ==============================

class clDiceLoss(nn.Module):
    """
    Centerline Dice Loss for vessel segmentation
    Based on the official implementation: https://github.com/jocpae/clDice
    """
    def __init__(self, iter_=3, smooth=1.0):
        super(clDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def soft_erode(self, img):
        """Soft erosion using max pooling inversion"""
        if len(img.shape) == 3:
            img = img.unsqueeze(1)  # Add channel dimension if needed
            
        # Erosion is the negation of dilation of the negated image
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

    def soft_dilate(self, img):
        """Soft dilation using max pooling"""
        if len(img.shape) == 3:
            img = img.unsqueeze(1)  # Add channel dimension if needed
            
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

    def soft_open(self, img):
        """Soft morphological opening: erosion followed by dilation"""
        eroded = self.soft_erode(img)
        return self.soft_dilate(eroded)

    def soft_skel(self, img, iter_):
        """
        Compute soft skeleton using iterative morphological operations
        Based on the official clDice implementation
        """
        img1 = self.soft_open(img)
        skel = img - img1
        
        for j in range(iter_):
            delta = img - self.soft_dilate(img1)
            img1 = self.soft_open(delta)
            skel = skel + img - img1
            
        return torch.clamp(skel, 0, 1)

    def soft_dice(self, y_true, y_pred):
        """
        Calculate soft Dice coefficient
        """
        smooth = 1e-5
        intersection = torch.sum(y_true * y_pred)
        return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

    def forward(self, y_pred, y_true):
        """
        Calculate clDice loss
        Args:
            y_pred: predicted probabilities [B, C, H, W] (after softmax/sigmoid)
            y_true: ground truth [B, C, H, W] (one-hot format) or [B, H, W] (class indices)
        """
        # Handle different input formats
        if len(y_true.shape) == 3:
            # Convert class indices to one-hot
            y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1])
            y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
        else:
            y_true_one_hot = y_true.float()
        
        # Apply softmax if needed (assuming input is logits)
        if not (y_pred.min() >= 0 and y_pred.max() <= 1):
            y_pred_soft = F.softmax(y_pred, dim=1)
        else:
            y_pred_soft = y_pred
            
        total_loss = 0
        num_classes = 0
        
        # Calculate loss for each class (excluding background)
        for i in range(1, y_pred.shape[1]):
            # Extract single class
            y_true_i = y_true_one_hot[:, i]
            y_pred_i = y_pred_soft[:, i]
            
            # Skip if no positive pixels in ground truth
            if y_true_i.sum() == 0:
                continue
                
            # Compute soft skeletons
            skel_pred = self.soft_skel(y_pred_i, self.iter)
            skel_true = self.soft_skel(y_true_i, self.iter)
            
            # Calculate topology preserving Dice components
            tprec = self.soft_dice(skel_pred, y_true_i)
            tsens = self.soft_dice(skel_true, y_pred_i)
            
            # Calculate clDice
            cl_dice = 2.0 * (tprec * tsens) / (torch.clamp(tprec + tsens, min=1e-5))
            
            # Pure clDice loss (1 - score to convert to loss)
            loss = 1.0 - cl_dice
            
            total_loss += loss
            num_classes += 1
            
        # Average over classes
        if num_classes > 0:
            return total_loss / num_classes
        else:
            # Return zero loss with gradient
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Combined loss function for vessel segmentation
    Includes CE, Dice, and clDice losses
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.5, cldice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(mode='multiclass')
        self.cldice_loss = clDiceLoss(iter_=3, smooth=1.0)
    
    def forward(self, pred, target):
        """
        Calculate combined loss
        
        Args:
            pred: predicted logits [B, C, H, W]
            target: ground truth [B, H, W]
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        
        total_loss = (self.ce_weight * ce + 
                     self.dice_weight * dice + 
                     self.cldice_weight * cldice)
        
        return total_loss, ce, dice, cldice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Useful for vessel segmentation where background >> vessel pixels
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth [B, H, W]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice loss
    Can control false positive and false negative weights
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Calculate Tversky loss
        
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth [B, H, W] or [B, C, H, W]
        """
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot if needed
        if len(targets.shape) == 3:
            targets = F.one_hot(targets.long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Flatten
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        
        # Calculate Tversky index
        intersection = (inputs * targets).sum(dim=2)
        fps = (inputs * (1 - targets)).sum(dim=2)
        fns = ((1 - inputs) * targets).sum(dim=2)
        
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        
        # Average over classes and batch
        return 1 - tversky.mean()


# Utility function to get loss function by name
def get_loss_function(loss_name, **kwargs):
    """
    Get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    loss_dict = {
        'ce': nn.CrossEntropyLoss,
        'dice': lambda: DiceLoss(mode='multiclass'),
        'cldice': clDiceLoss,
        'combined': CombinedLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss
    }
    
    if loss_name not in loss_dict:
        raise ValueError(f"Loss function {loss_name} not found. Available: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name](**kwargs)


# ============================== Evaluation Metrics ==============================

def calculate_metrics(pred, target, eps=1e-8):
    """
    Calculate sensitivity, specificity, and accuracy for binary segmentation
    
    Args:
        pred: predicted binary mask (numpy array or torch tensor)
        target: ground truth binary mask (numpy array or torch tensor)
        eps: small value to avoid division by zero
        
    Returns:
        dict: Dictionary containing sensitivity, specificity, and accuracy
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Ensure binary
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    # Calculate confusion matrix elements
    tp = np.sum(pred * target)
    tn = np.sum((1 - pred) * (1 - target))
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)
    
    # Calculate metrics
    sensitivity = tp / (tp + fn + eps)  # True Positive Rate (Recall)
    specificity = tn / (tn + fp + eps)  # True Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }


def calculate_path_metrics(pred_skeleton, target_skeleton, threshold=3):
    """
    Calculate infeasible (INF) and correct (COR) path fractions
    Based on skeleton matching for vessel connectivity evaluation
    
    Args:
        pred_skeleton: predicted vessel skeleton (binary numpy array)
        target_skeleton: ground truth vessel skeleton (binary numpy array)
        threshold: maximum distance for a path to be considered correct (pixels)
        
    Returns:
        dict: Dictionary containing INF and COR values in 0-1 range
    """
    # Ensure binary skeletons
    pred_skeleton = (pred_skeleton > 0).astype(np.uint8)
    target_skeleton = (target_skeleton > 0).astype(np.uint8)
    
    # Get skeleton pixels
    pred_pixels = np.column_stack(np.where(pred_skeleton))
    target_pixels = np.column_stack(np.where(target_skeleton))
    
    if len(pred_pixels) == 0:
        return {'INF': 1.0, 'COR': 0.0}
    
    if len(target_pixels) == 0:
        return {'INF': 1.0, 'COR': 0.0}
    
    # Calculate distances from each predicted pixel to nearest target pixel
    from scipy.spatial import distance_matrix
    
    # Compute distance matrix
    dist_matrix = distance_matrix(pred_pixels, target_pixels)
    
    # Find minimum distance for each predicted pixel
    min_distances = np.min(dist_matrix, axis=1)
    
    # Calculate correct and infeasible paths
    correct_pixels = np.sum(min_distances <= threshold)
    total_pixels = len(pred_pixels)
    
    # Calculate fractions (0-1 range)
    COR = correct_pixels / total_pixels
    INF = 1.0 - COR
    
    return {
        'INF': INF,
        'COR': COR
    }


def calculate_all_metrics(pred, target, calculate_path=True):
    """
    Calculate all metrics: Sensitivity, Specificity, Accuracy, INF, and COR
    
    Args:
        pred: predicted binary mask
        target: ground truth binary mask
        calculate_path: whether to calculate path metrics (requires skeletonization)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Basic metrics
    basic_metrics = calculate_metrics(pred, target)
    
    # Path metrics (optional as skeletonization can be slow)
    if calculate_path:
        # Create skeletons
        pred_skel = make_skeleton(pred)
        target_skel = make_skeleton(target)
        
        # Calculate path metrics
        path_metrics = calculate_path_metrics(pred_skel, target_skel)
    else:
        path_metrics = {'INF': None, 'COR': None}
    
    # Combine all metrics
    all_metrics = {
        'sensitivity': basic_metrics['sensitivity'],
        'specificity': basic_metrics['specificity'],
        'accuracy': basic_metrics['accuracy'],
        'INF': path_metrics['INF'],
        'COR': path_metrics['COR']
    }
    
    return all_metrics


from skimage.morphology import skeletonize
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))
    return 2 * tprec * tsens / (tprec + tsens)


def centerline_dice(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
    nan_for_nonexisting=True
):
    if mask is not None:
        ground_truth = np.where(mask, ground_truth, 0)
        prediction = np.where(mask, prediction, 0)
    
    gt_bin = (ground_truth > 0.5).astype(np.uint8)
    pred_bin = (prediction > threshold).astype(np.uint8)
    
    if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return float("NaN") if nan_for_nonexisting else 0.0
    
    try:
        skeleton_gt = skeletonize(gt_bin)
        skeleton_pred = skeletonize(pred_bin)
    except ValueError:
        return float("NaN")

    intersection = np.sum(skeleton_gt * skeleton_pred)
    return (2.0 * intersection) / (np.sum(skeleton_gt) + np.sum(skeleton_pred) + 1e-7)
