"""
Utility functions and classes for inference pipeline.
"""

import numpy as np
import cv2
import os
import json
import torch
from typing import Dict, List, Any, Union, Tuple


class Logger:
    """Unified logging utilities."""
    quiet_mode: bool = False
    
    @staticmethod
    def info(message: str):
        """Log information message."""
        if Logger.quiet_mode:
            return
        print(f"ℹ️ {message}")
    
    @staticmethod
    def success(message: str):
        """Log success message."""
        if Logger.quiet_mode:
            return
        print(f"✅ {message}")
    
    @staticmethod
    def warning(message: str):
        """Log warning message."""
        print(f"⚠️ {message}")
    
    @staticmethod
    def error(message: str):
        """Log error message."""
        print(f"❌ {message}")
    
    @staticmethod
    def debug(message: str):
        """Log debug message."""
        if Logger.quiet_mode:
            return
        print(f"🐛 {message}")
    
    @staticmethod
    def progress(current: int, total: int, item_name: str = ""):
        """Log progress message."""
        if Logger.quiet_mode:
            return
        print(f"🔄 Processing [{current}/{total}]{': ' + item_name if item_name else ''}")
    
    @staticmethod
    def saved(description: str, filepath: str = None):
        """Log file save message."""
        if Logger.quiet_mode:
            return
        if filepath:
            print(f"💾 Saved {description}: {filepath}")
        else:
            print(f"💾 Saved {description}")

    @staticmethod
    def set_quiet(quiet: bool):
        """Enable or disable quiet mode for informational logging."""
        Logger.quiet_mode = quiet


class FileManager:
    """File management utilities."""
    
    @staticmethod
    def create_output_paths(base_output_dir: str, image_name: str) -> Dict[str, str]:
        """
        Create output directory structure.
        
        Args:
            base_output_dir: Base output directory
            image_name: Name of the image being processed
            
        Returns:
            Dictionary of created paths
        """
        # Define standard paths (lazily create some to avoid empty folders)
        paths = {
            'base': base_output_dir,
            'masks': os.path.join(base_output_dir, 'masks'),
            # Create individual_lesions lazily only when saving lesion channels
            'individual_lesions': os.path.join(base_output_dir, 'masks', 'individual_lesions'),
            # Top-level individual_channels for visualizations; subfolders created by visualizer as needed
            'individual_channels': os.path.join(base_output_dir, 'individual_channels'),
            'visualizations': os.path.join(base_output_dir, 'visualizations')
        }
        
        # Create only the commonly used base directories upfront
        for key in ['base', 'masks', 'visualizations']:
            os.makedirs(paths[key], exist_ok=True)
            
        return paths
    
    @staticmethod
    def save_image_with_log(image: np.ndarray, filepath: str, 
                           description: str = "", original_size: Tuple[int, int] = None):
        """
        Save image with logging.
        
        Args:
            image: Image to save
            filepath: File path to save to
            description: Description for logging
            original_size: Original image dimensions
        """
        cv2.imwrite(filepath, image)
        
        if description and original_size:
            width, height = original_size
            Logger.saved(f"{description} at original size {width}x{height}", filepath)
        elif description:
            Logger.saved(description, filepath)
    
    @staticmethod
    def save_json(data: Dict, filepath: str, description: str = ""):
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filepath: File path
            description: Description for logging
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(data), f, ensure_ascii=False, indent=2)
        
        if description:
            Logger.saved(description, filepath)


class MaskProcessor:
    """Mask processing utilities."""
    
    @staticmethod
    def apply_threshold(prediction: Union[np.ndarray, torch.Tensor], 
                       threshold: float = 0.5) -> np.ndarray:
        """
        Apply threshold to get binary mask.
        
        Args:
            prediction: Prediction array/tensor
            threshold: Threshold value
            
        Returns:
            Binary mask
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        return (prediction > threshold).astype(np.uint8)
    
    @staticmethod
    def apply_argmax(prediction: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply argmax to get class mask.
        
        Args:
            prediction: Multi-channel prediction
            
        Returns:
            Class mask
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        return np.argmax(prediction, axis=0).astype(np.uint8)
    
    @staticmethod
    def process_multilabel_prediction(prediction: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process multilabel prediction.
        
        Args:
            prediction: Raw multilabel prediction
            
        Returns:
            Tuple of (combined_mask, individual_binary_masks)
        """
        pred_probs = torch.sigmoid(prediction).squeeze().cpu().numpy()
        pred_binary = (pred_probs > 0.5).astype(np.uint8)
        
        # Create combined mask
        if len(pred_binary.shape) == 3:
            combined_mask = np.sum(pred_binary, axis=0)
        else:
            combined_mask = pred_binary
            
        combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        
        return combined_mask, pred_binary
    
    @staticmethod
    def get_largest_component(mask: np.ndarray) -> np.ndarray:
        """
        Get largest connected component from mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask with only largest component
        """
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        if num_labels <= 1:
            return mask
            
        # Find largest component (excluding background)
        largest_size = 0
        largest_label = 0
        
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            if size > largest_size:
                largest_size = size
                largest_label = label
                
        # Create mask with only largest component
        largest_mask = (labels == largest_label).astype(np.uint8)
        return largest_mask
    
    @staticmethod
    def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Clean up mask using morphological operations.
        
        Args:
            mask: Binary mask
            kernel_size: Size of morphological kernel
            
        Returns:
            Cleaned mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening to remove noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return cleaned


class ImageUtils:
    """Image processing utilities."""
    
    @staticmethod
    def create_overlay(base_image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int], alpha: float = 0.3) -> np.ndarray:
        """
        Create colored overlay on base image.
        
        Args:
            base_image: Base image
            mask: Binary mask
            color: RGB color for overlay
            alpha: Transparency level
            
        Returns:
            Image with overlay
        """
        overlay = base_image.copy().astype(np.float32)
        
        if mask.any():
            color_array = np.array(color, dtype=np.float32)
            overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + color_array * alpha
            
        return overlay.astype(np.uint8)
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int],
                                keep_aspect: bool = True, 
                                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image while optionally maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            keep_aspect: Whether to maintain aspect ratio
            interpolation: OpenCV interpolation method
            
        Returns:
            Resized image
        """
        if not keep_aspect:
            return cv2.resize(image, target_size, interpolation=interpolation)
            
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Pad to target size if needed
        if new_w != target_w or new_h != target_h:
            # Create padded image
            padded = np.zeros((target_h, target_w) + image.shape[2:], dtype=image.dtype)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return padded
            
        return resized


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)) or \
         str(type(obj)) == "<class 'numpy.bool_'>" or \
         (hasattr(obj, 'dtype') and 'bool' in str(obj.dtype)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    elif obj is None:
        return None
    else:
        return obj


def parse_output_channels(output_channels_str: str) -> Union[Tuple[int, ...], int]:
    """
    Parse output channels string configuration.
    
    Args:
        output_channels_str: String representation of output channels
        
    Returns:
        Parsed output channels
    """
    if output_channels_str is None:
        return None
        
    try:
        import ast
        parsed = ast.literal_eval(output_channels_str)
        return parsed
    except (ValueError, SyntaxError):
        try:
            return int(output_channels_str)
        except ValueError:
            raise ValueError(f"Invalid output_channels format: {output_channels_str}")


def find_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find image files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of valid extensions
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
        
    image_files = []
    
    for ext in extensions:
        # Check both lowercase and uppercase
        pattern_lower = os.path.join(directory, f'*.{ext.lower()}')
        pattern_upper = os.path.join(directory, f'*.{ext.upper()}')
        
        import glob
        image_files.extend(glob.glob(pattern_lower))
        image_files.extend(glob.glob(pattern_upper))
        
    return sorted(list(set(image_files)))  # Remove duplicates and sort