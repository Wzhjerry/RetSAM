import random
from typing import List, Union

import numpy
import numpy as np
from skimage import transform, color
import torch
from scipy import ndimage



def random_affine(npimages):
    h = npimages[0].shape[0]
    w = npimages[0].shape[1]

    dorotate = np.deg2rad(random.uniform(-180, 180))

    zoom = 0.7  # you can modified it

    shear = np.deg2rad(random.uniform(-25, 25))

    center_shift = np.array((h, w)) / 2.0
    tform_center = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(
        rotation=dorotate,
        scale=(1/zoom, 1/zoom),
        shear=shear
    )

    tform = tform_center + tform_aug + tform_uncenter

    transformed_imgs = []
    for i in range(len(npimages)):
        img = npimages[i]
        if i == 1 or i == 2:  
            order = 0
        else:
            order = 3
            
        output_shape = (int(h * zoom), int(w * zoom)) + img.shape[2:]  
        transformed_img = transform.warp(
            img,
            tform,
            output_shape=output_shape,
            order=order,
            preserve_range=True,
            mode='constant',
            cval=0
        )
        transformed_imgs.append(transformed_img)

    return transformed_imgs


def random_hsv(npimages):
    img_full = npimages[0]
    img = img_full[:, :, :3]
    img = color.rgb2hsv(img)
    dh = random.uniform(-0.02, 0.02)
    ds = random.uniform(-0.2, 0.2)
    dv = random.uniform(-0.2, 0.2)
    ms = random.uniform(0.8, 1.2)
    mv = random.uniform(0.8, 1.2)

    img[:, :, 0] += dh
    img[:, :, 0][img[:, :, 0] > 1] -= 1
    img[:, :, 0][img[:, :, 0] < 0] += 1

    img[:, :, 1] = np.clip(ms * (img[:, :, 1] + ds), 0, 1)
    img[:, :, 2] = np.clip(mv * (img[:, :, 2] + dv), 0, 1)

    img_full[:, :, :3] = color.hsv2rgb(img)

    npimages[0] = img_full

    return npimages


def to_torch_tensors(npimages):
    return [torch.from_numpy(img.transpose(2, 0, 1).astype('float32')) for img in npimages]


def random_vertical_flip(npimages):
    if random.random() < 0.5:
        return [np.flip(img, axis=0) for img in npimages]
    else:
        return npimages


def random_horizontal_flip(npimages):
    if random.random() < 0.5:
        return [np.flip(img, axis=1) for img in npimages]
    else:
        return npimages


def random_cutout(npimages, num_cutouts=16, size=(0.04, 0.04)):
    if random.random() < 0.8:
        img = npimages[0].copy()
        cutout_size = (int(size[0] * img.shape[0]), int(size[1] * img.shape[1]))
        for _ in range(num_cutouts):
            y = random.randint(0, img.shape[0] - cutout_size[0])
            x = random.randint(0, img.shape[1] - cutout_size[1])
            img[y:y+cutout_size[0], x:x+cutout_size[1], :] = random.uniform(0.4, 0.6)
        npimages[0] = img
    return npimages


def check_nonzeros_min(npimages, min_nonzeros):
    return all(np.count_nonzero(img) > min_nonzeros for img in npimages)


def get_unet_padding_np(np_image: numpy.ndarray, n_down=4) -> tuple:
    n = 2**n_down
    shape = np_image.shape
    h_pad = n - shape[0] % n
    w_pad = n - shape[1] % n
    h_half_pad = int(h_pad/2)
    w_half_pad = int(w_pad/2)
    if len(shape) == 3:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad), (0, 0)
    else:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad)
    return padding


def pad_images_unet(
    np_images: List[numpy.ndarray],
    return_paddings: bool=False,
) -> Union[tuple, list]:
    padded_images = []
    paddings = []
    for np_image in np_images:
        padding = get_unet_padding_np(np_image)
        paddings.append(padding)
        padded_images.append(np.pad(np_image, padding))
    if return_paddings:
        return padded_images, paddings
    else:
        return padded_images


def rescale_to_width(np_images, target_width=720):
    transformed_images = []
    for img in np_images:
        width = img.shape[1]
        ratio = target_width / width
        transformed_images.append(ndimage.zoom(img, (ratio, ratio, 1), order=1))
    return transformed_images


def rescale(np_images, target_height=576):
    transformed_images = []
    for img in np_images:
        height = img.shape[0]
        ratio = target_height / height
        transformed_images.append(ndimage.zoom(img, (ratio, ratio, 1), order=1))
    return transformed_images
