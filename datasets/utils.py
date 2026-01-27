import os
from torchvision import transforms
import numpy as np
import scipy
from skimage.measure import label
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


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


# mean: [0.3365757, 0.32790676, 0.00656315], std: [0.15169601, 0.1455664, 0.03449817]
def build_transform(args, train=False, use_albumentations=True, task_type='vessel'):
    """
    Build transform pipeline - supports both torchvision and albumentations.
    For fundus images, albumentations is recommended for better augmentations.
    
    Args:
        args: Arguments containing size and other parameters
        train: Whether this is for training or validation
        use_albumentations: Whether to use albumentations (recommended) or torchvision
        task_type: Type of segmentation task ('vessel', 'lesion', 'od_oc', etc.)
    """
    if use_albumentations:
        return build_albumentations_transform(args, train, task_type=task_type), None
        
    return build_torchvision_transform(args, train)


def build_torchvision_transform(args, train=False):
    if train:
        img_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                # transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                # transforms.RandomAutocontrast(p=0.5),
                # transforms.RandomInvert(p=0.5),
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181], std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114]),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Resize(args.size, interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    else:
        img_transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181], std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114]),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.Resize(args.size, interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    return img_transform, label_transform


def crop_center(input_data):
    """
    Crop the central region of the input data from 0.25 to 0.75 of its dimensions.
    The input can be either an image (2D or 3D) or a label (2D).

    Args:
        input_data (numpy.ndarray): The input data, can be:
            - 2D array (H,W) for grayscale image or label
            - 3D array (H,W,C) for color image

    Returns:
        numpy.ndarray: The cropped central region of the input data.
    """
    # Calculate the crop boundaries
    height, width = input_data.shape[:2]
    start_y = int(height * 0.25)
    end_y = int(height * 0.75)
    start_x = int(width * 0.25)
    end_x = int(width * 0.75)

    # Crop the input data
    cropped_data = input_data[start_y:end_y, start_x:end_x]
    
    return cropped_data


def build_albumentations_transform(args, train=False, task_type='vessel'):
    """
    Build albumentations transform pipeline for fundus images.
    Includes specialized augmentations for retinal segmentation tasks.
    Properly handles synchronized image and mask transformations.
    
    Args:
        args: Arguments containing size and other parameters
        train: Whether this is for training or validation
        task_type: Type of segmentation task ('vessel', 'lesion', 'od_oc', etc.)
    """
    # Define additional targets based on task type
    additional_targets = {}
    if task_type == 'vessel':
        additional_targets = {'artery_mask': 'mask', 'vein_mask': 'mask'}
    # For lesion and other single-mask tasks, no additional targets needed
    
    if train:
        transform = A.Compose([
            # Spatial transforms (applied to both image and masks)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                p=0.4,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.1, 
                rotate_limit=45, 
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Elastic and grid distortions (useful for all structures)
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.OpticalDistortion(
                    distort_limit=0.05,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.Perspective(
                    scale=(0.05, 0.1),
                    keep_size=True,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=0,
                    mask_pad_val=0,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.2,
                ),
                A.PiecewiseAffine(
                    scale=(0.01, 0.03),
                    p=0.2,
                ),
            ], p=0.3),
            
            # Crop and resize (applied to both image and masks)
            A.RandomResizedCrop(
                size=(args.size, args.size),
                scale=(0.4, 1.0),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=0.5
            ),
            A.PadIfNeeded(min_height=args.size, min_width=args.size, border_mode=cv2.BORDER_CONSTANT, p=0.2),
            
            # Final resize to ensure consistent size (applied to both)
            A.Resize(args.size, args.size, interpolation=cv2.INTER_LINEAR),
            
            # Color augmentations (IMAGE ONLY - not applied to masks)
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            A.ChannelShuffle(p=0.05),
            A.Equalize(mode="cv", by_channels=True, p=0.2),
            A.RandomToneCurve(scale=0.2, p=0.2),
            
            # Fundus-specific augmentations (IMAGE ONLY)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.5
                ),
            ], p=0.5),
            
            # Noise augmentations (IMAGE ONLY - simulate camera noise)
            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
                A.GaussNoise(p=0.5),
            ], p=0.2),
            
            # Blur augmentations (IMAGE ONLY - simulate focus issues)
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            
            # Sharpening (IMAGE ONLY - enhance edges)
            A.Sharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1.0),
                p=0.3
            ),
            A.UnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.6), threshold=5, p=0.2),

            # Image compression and resolution degradations (IMAGE ONLY)
            # Simplify compression/downscale to avoid version warnings
            A.ImageCompression(p=0.2),
            A.Downscale(p=0.2),

            # Dropouts/occlusions (IMAGE ONLY)
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(args.size * 0.08),
                    max_width=int(args.size * 0.08),
                    min_holes=1,
                    min_height=int(args.size * 0.02),
                    min_width=int(args.size * 0.02),
                    fill_value=0,
                    p=0.7,
                ),
                A.GridDropout(ratio=0.5, p=0.3),
            ], p=0.15),

            # Normalization and tensor conversion (IMAGE ONLY)
            A.Normalize(
                mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181],
                std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets=additional_targets)
    else:
        # Validation/test transform
        transform = A.Compose([
            A.Resize(args.size, args.size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181],
                std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets=additional_targets)
    
    return transform


def crop_center(input_data):
    """
    Crop the central region of the input data from 0.25 to 0.75 of its dimensions.
    The input can be either an image (2D or 3D) or a label (2D).

    Args:
        input_data (numpy.ndarray): The input data, can be:
            - 2D array (H,W) for grayscale image or label
            - 3D array (H,W,C) for color image

    Returns:
        numpy.ndarray: The cropped central region of the input data.
    """
    # Calculate the crop boundaries
    height, width = input_data.shape[:2]
    start_y = int(height * 0.25)
    end_y = int(height * 0.75)
    start_x = int(width * 0.25)
    end_x = int(width * 0.75)

    # Crop the input data
    cropped_data = input_data[start_y:end_y, start_x:end_x]
    
    return cropped_data


def regularize_cor_cut(ori_image, xmin, ymin, xmax, ymax):
    img_height = ori_image.shape[0]
    img_width = ori_image.shape[1]
    # Mask sure all coordinates in right area

    if xmin > img_width:
        xmin = img_width
    elif xmin < 0:
        xmin = 0

    if xmax > img_width:
        xmax = img_width
    elif xmax < 0:
        xmax = 0

    if ymin > img_height:
        ymin = img_height
    elif ymin < 0:
        ymin = 0

    if ymax > img_height:
        ymax = img_height
    elif ymax < 0:
        ymax = 0

    return xmin, ymin, xmax, ymax


def ave_edge_cal(ori_image):
    gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    ul = gray[:5, :5]
    ur = gray[:5, ori_image.shape[1] - 5:]
    ll = gray[ori_image.shape[0] - 5:, :5]
    lr = gray[ori_image.shape[0] - 5:, ori_image.shape[1] - 5:]
    mean = (np.sum(ul) / 25 + np.sum(ur) / 25 + np.sum(ll) / 25 + np.sum(lr) / 25) / 4
    return mean


def remove_black_edge(ori_image):
    b, g, r = cv2.split(ori_image)
    img_width = ori_image.shape[1]
    img_height = ori_image.shape[0]

    mean_pixel = ave_edge_cal(ori_image)

    if mean_pixel < 127:
        img_mask = np.array(((b < 30) * (g < 30) * (r < 30)) * 1, dtype=np.uint8)
    else:
        img_mask = np.array((((b > 245) * (g > 245) * (r > 245))) * 1, dtype=np.uint8)

    # Cauculate the x-beginning for each line by sum the half mask
    half_start = np.sum(img_mask[:, :img_width // 2], 1).reshape(img_height, 1)
    # Cauculate the x-end by img_width - sum of the right half mask
    half_end = img_width - np.sum(img_mask[:, img_width // 2:], 1).reshape(img_height, 1)

    start_len = img_width // 2 - np.min(half_start)
    end_len = np.max(half_end) - img_width // 2
    half_len = max(start_len, end_len)

    cor_sum = np.sum(img_mask, 1).reshape(img_height, 1)
    cor_sum_binary = np.array((cor_sum > img_width - 10)*1, dtype=np.uint8)

    vertical_start = cor_sum_binary[:len(cor_sum_binary) // 2]
    vertical_end = cor_sum_binary[len(cor_sum_binary) // 2:]

    vertical_start_cor = np.sum(vertical_start)
    vertical_end_cor = np.sum(vertical_end)

    xmin, ymin, xmax, ymax = regularize_cor_cut(ori_image,
                                                int(img_width // 2 - half_len),
                                                int(vertical_start_cor),
                                                int(img_width // 2 + half_len),
                                                int(img_height-vertical_end_cor))
    
    # Check if ymax and xmax are both 0 or if they're equal to ymin and xmin
    if (ymax == 0 and xmax == 0) or (ymax - ymin <= 0 or xmax - xmin <= 0):
        # Return original image with its full dimensions
        return ori_image, 0, img_height, 0, img_width
    
    cut_image = ori_image[ymin: ymax, xmin: xmax]

    # Additional check for empty resulting image
    if cut_image.size == 0:
        return ori_image, 0, img_height, 0, img_width
        
    return cut_image, ymin, ymax, xmin, xmax
