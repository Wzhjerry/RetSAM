"""
Image processing utilities for inference pipeline.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, Optional

from .config import InferenceConfig
from .utils import Logger


class ImageProcessor:
    """Handles image loading and preprocessing."""
    
    def __init__(self):
        """Initialize image processor."""
        self.config = InferenceConfig()
        self.transform = self._create_transform()
        
    def load_and_preprocess(self, image_path: str) -> Dict:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing:
                - tensor: Preprocessed image tensor
                - original: Original image
                - processed: Processed image
                - original_shape: Original image dimensions
                - crop_coords: Coordinates used for black edge removal
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB
        original_image_rgb = original_image[..., ::-1]
        
        # Store original dimensions
        original_height, original_width = original_image.shape[:2]
        
        # Remove black edges
        processed_image, crop_coords = self.remove_black_edge(original_image_rgb)
        
        # Resize to model input size
        resized_image = cv2.resize(
            processed_image, 
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(resized_image))
        
        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0)
        
        return {
            'tensor': tensor,
            'original': original_image,
            'processed': processed_image,
            'resized': resized_image,
            'original_shape': (original_width, original_height),
            'crop_coords': crop_coords
        }
        
    def remove_black_edge(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove black edges from fundus image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Cropped image and crop coordinates
        """
        # Convert to BGR for processing
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Calculate edge statistics
        mean_edge = self._calculate_edge_mean(bgr_image)
        
        # Create mask for black/white regions
        mask = self._create_edge_mask(bgr_image, mean_edge)
        
        # Find crop coordinates
        crop_coords = self._find_crop_region(mask, bgr_image.shape)
        
        # Crop image
        cropped = bgr_image[
            crop_coords['ymin']:crop_coords['ymax'],
            crop_coords['xmin']:crop_coords['xmax']
        ]
        
        # Convert back to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        return cropped_rgb, crop_coords
        
    def resize_to_original(self, image: np.ndarray, original_shape: Tuple[int, int],
                          interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
        """
        Resize image back to original dimensions.
        
        Args:
            image: Image to resize
            original_shape: Target (width, height)
            interpolation: OpenCV interpolation method
            
        Returns:
            Resized image
        """
        width, height = original_shape
        return cv2.resize(image, (width, height), interpolation=interpolation)
        
    def resize_to_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to standard analysis size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(
            image,
            (self.config.analysis_size, self.config.analysis_size),
            interpolation=cv2.INTER_NEAREST
        )
        
    def _create_transform(self) -> transforms.Compose:
        """Create image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalization_mean,
                std=self.config.normalization_std
            )
        ])
        
    def _calculate_edge_mean(self, image: np.ndarray) -> float:
        """Calculate mean value of image edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Sample corners
        corner_size = 5
        corners = [
            gray[:corner_size, :corner_size],                    # Upper left
            gray[:corner_size, w-corner_size:],                  # Upper right
            gray[h-corner_size:, :corner_size],                  # Lower left
            gray[h-corner_size:, w-corner_size:]                 # Lower right
        ]
        
        # Calculate mean
        corner_means = [np.mean(corner) for corner in corners]
        return np.mean(corner_means)
        
    def _create_edge_mask(self, image: np.ndarray, mean_edge: float) -> np.ndarray:
        """Create mask for edge detection."""
        b, g, r = cv2.split(image)
        
        if mean_edge < 127:
            # Dark edges
            mask = ((b < 30) & (g < 30) & (r < 30)).astype(np.uint8)
        else:
            # Light edges
            mask = ((b > 245) & (g > 245) & (r > 245)).astype(np.uint8)
            
        return mask
        
    def _find_crop_region(self, mask: np.ndarray, shape: Tuple) -> Dict:
        """Find optimal crop region based on mask."""
        h, w = shape[:2]
        
        # Find horizontal bounds
        horizontal_profile = np.sum(mask, axis=0)
        valid_cols = np.where(horizontal_profile < h - 10)[0]
        
        if len(valid_cols) > 0:
            xmin = valid_cols[0]
            xmax = valid_cols[-1] + 1
        else:
            xmin, xmax = w // 4, 3 * w // 4
            
        # Find vertical bounds
        vertical_profile = np.sum(mask, axis=1)
        valid_rows = np.where(vertical_profile < w - 10)[0]
        
        if len(valid_rows) > 0:
            ymin = valid_rows[0]
            ymax = valid_rows[-1] + 1
        else:
            ymin, ymax = h // 4, 3 * h // 4
            
        # Ensure valid bounds
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        
        # Ensure positive dimensions
        if xmax <= xmin:
            xmin, xmax = w // 4, 3 * w // 4
        if ymax <= ymin:
            ymin, ymax = h // 4, 3 * h // 4
            
        return {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax
        }