"""
Visualization module for creating and saving segmentation visualizations.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import os

from .config import InferenceConfig
from .utils import Logger, FileManager, MaskProcessor
from .image_processor import ImageProcessor


TASK_COLOR_MAP = {
    'artery_vein': {
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0, 255, 0),  # overlap: artery & vein
    },
    'od_oc': {
        1: (255, 255, 0),
        2: (255, 0, 255),
    },
    'tessellation': {
        1: (0, 255, 255),
        2: (128, 0, 128),
        3: (255, 192, 203),
    },
    'myopia': {
        1: (0, 128, 255),
        2: (138, 43, 226),
        3: (255, 215, 0),
    },
    'lesion': {
        1: (46, 139, 87),
        2: (128, 0, 128),
        3: (100, 149, 237),
        4: (173, 216, 230),
    },
    'lesion_s1': {
        1: (46, 139, 87),
        2: (128, 0, 128),
        3: (100, 149, 237),
        4: (173, 216, 230),
        5: (205, 133, 63),
    },
    'lesion_all': {
        1: (0, 128, 128),
        2: (32, 178, 170),
        5: (148, 0, 211),
        9: (64, 224, 208),
    },
    'lesion_s2': {
        1: (0, 128, 128),
        2: (32, 178, 170),
        5: (148, 0, 211),
        9: (64, 224, 208),
    },
    'lesion_hard': {
        1: (72, 209, 204),
        2: (148, 0, 211),
        3: (30, 144, 255),
        4: (0, 255, 127),
    },
    'lesion_s3': {
        1: (72, 209, 204),
        2: (205, 92, 92),
        3: (30, 144, 255),
        4: (0, 255, 127),
        5: (255, 140, 0),
    },
    'lesion_dr': {
        1: (46, 139, 87),
        2: (128, 0, 128),
        3: (100, 149, 237),
        4: (173, 216, 230),
    },
    'lesion_amd': {
        1: (173, 216, 230),
        2: (135, 206, 235),
    },
    'lesion_others': {
        1: (0, 128, 128),
        2: (148, 0, 211),
        3: (0, 255, 127),
        4: (32, 178, 170),  # macular hole original color
    },
    'possible_lesions': {
        1: (255, 255, 255),  # other_lesions overlay in white as requested
        2: (64, 224, 208),
        3: (72, 209, 204),
    },
}


class Visualizer:
    """Handles visualization of segmentation results."""
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.image_processor = ImageProcessor()
        self.mask_processor = MaskProcessor()
    
    def _is_multilabel_task(self, task_name: str, task_idx: int = -1) -> bool:
        """
        Determine if a task should be treated as multilabel for mask export.
        """
        if task_idx == 5 or self.config.get_display_name(task_name) == 'lesion_s2':
            return True
        return self.config.get_output_type(task_name) == 'multilabel'
        
    def _place_on_original_canvas(self, mask: np.ndarray, image_data: Dict) -> np.ndarray:
        """
        Place a cropped-region mask back onto the original image canvas using crop coordinates.
        - mask: mask sized to the cropped region (or any size; will be resized)
        - image_data: includes 'original_shape' (w,h) and 'crop_coords' with xmin,xmax,ymin,ymax
        """
        original_width, original_height = image_data['original_shape']
        crop = image_data.get('crop_coords', None)
        if not crop:
            # Fallback: direct resize to original if cropping was not applied
            return self.image_processor.resize_to_original(
                mask.astype(np.uint8), (original_width, original_height), cv2.INTER_NEAREST
            )
        xmin, xmax = int(crop['xmin']), int(crop['xmax'])
        ymin, ymax = int(crop['ymin']), int(crop['ymax'])
        cropped_w = max(0, xmax - xmin)
        cropped_h = max(0, ymax - ymin)
        if cropped_w == 0 or cropped_h == 0:
            return np.zeros((original_height, original_width), dtype=np.uint8)
        # Ensure mask matches cropped region size
        resized_cropped = cv2.resize(
            mask.astype(np.uint8), (cropped_w, cropped_h), interpolation=cv2.INTER_NEAREST
        )
        # Paste into full-size canvas
        canvas = np.zeros((original_height, original_width), dtype=np.uint8)
        canvas[ymin:ymax, xmin:xmax] = resized_cropped
        return canvas
    
    def save_binary_masks_per_class(self, prediction: torch.Tensor, task_name: str,
                                    output_dir: str, image_data: Dict, task_idx: int = -1) -> Dict[str, str]:
        """
        Save per-class binary masks (0 or 255) for a task.

        Returns:
            Dictionary mapping class names to saved file paths.
        """
        paths = FileManager.create_output_paths(output_dir, task_name)
        mask_dir = paths['masks']
        display_name = self.config.get_display_name(task_name)
        class_mapping = self.config.get_task_class_mapping(task_name)
        saved_paths: Dict[str, str] = {}

        is_multilabel = self._is_multilabel_task(task_name, task_idx)

        if is_multilabel:
            _, pred_binary = self.mask_processor.process_multilabel_prediction(prediction)
            if pred_binary.ndim == 2:
                pred_binary = np.expand_dims(pred_binary, axis=0)

            if class_mapping:
                class_ids = sorted(class_mapping.keys())
            else:
                class_ids = list(range(1, pred_binary.shape[0]))

            # Ensure we include any additional channels beyond mapping
            for idx in range(1, pred_binary.shape[0]):
                if idx not in class_ids:
                    class_ids.append(idx)

            for class_id in sorted(set(class_ids)):
                if class_id <= 0 or class_id >= pred_binary.shape[0]:
                    continue
                binary_cropped = pred_binary[class_id].astype(np.uint8)
                mask_on_canvas = self._place_on_original_canvas(binary_cropped, image_data)
                mask_binary = (mask_on_canvas > 0).astype(np.uint8) * 255
                class_name = class_mapping.get(class_id, f'class_{class_id}')
                filename = f'{display_name}_{class_name}.png'
                save_path = os.path.join(mask_dir, filename)
                cv2.imwrite(save_path, mask_binary)
                saved_paths[class_name] = save_path

            return saved_paths

        # Multiclass or binary case
        pred_np = prediction.squeeze().cpu().numpy()

        if pred_np.ndim == 3:
            num_classes = pred_np.shape[0]
            label_map = np.argmax(pred_np, axis=0).astype(np.uint8)
        else:
            num_classes = 2  # background + foreground
            probs = 1.0 / (1.0 + np.exp(-pred_np))
            label_map = (probs > 0.5).astype(np.uint8)

        if class_mapping:
            class_ids = sorted(class_mapping.keys())
        else:
            class_ids = list(range(1, num_classes))

        # Ensure we cover all possible classes
        for cid in range(1, num_classes):
            if cid not in class_ids:
                class_ids.append(cid)

        for class_id in sorted(set(class_ids)):
            if class_id <= 0:
                continue
            binary_cropped = (label_map == class_id).astype(np.uint8)
            mask_on_canvas = self._place_on_original_canvas(binary_cropped, image_data)
            mask_binary = (mask_on_canvas > 0).astype(np.uint8) * 255
            class_name = class_mapping.get(class_id, f'class_{class_id}')
            filename = f'{display_name}_{class_name}.png'
            save_path = os.path.join(mask_dir, filename)
            cv2.imwrite(save_path, mask_binary)
            saved_paths[class_name] = save_path

        return saved_paths
        
    def save_task_masks(self, prediction: torch.Tensor, task_name: str, 
                       output_dir: str, image_data: Dict, task_idx: int = -1) -> Optional[str]:
        """
        Save masks for a specific task.
        
        Args:
            prediction: Model prediction for this task
            task_name: Name of the task
            output_dir: Directory to save masks
            image_data: Image metadata
            
        Returns:
            Path to saved mask
        """
        # For lesion_s1/s2/s3, skip saving per-task masks; regrouped masks will be saved later
        if task_name in ['lesion_s1', 'lesion_s2', 'lesion_s3']:
            # Return None to indicate no individual mask should be saved
            return None

        # Create output paths
        paths = FileManager.create_output_paths(output_dir, task_name)
        
        # Process prediction based on task type
        output_type = self.config.get_output_type(task_name)
        
        # Force lesion_s2 (task index 5 / task name 'lesion_all') to use multilabel saving
        # so we can restrict saved channels to [1, 2, 5, 9]
        is_lesion_multilabel = (
            (output_type == 'multilabel' and self.config.is_lesion_task(task_name))
            or task_idx == 5
            or self.config.get_display_name(task_name) == 'lesion_s2'
        )
        
        # Special-case handling for artery_vein multilabel (2-channel) output
        if task_name == 'artery_vein' and self.config.get_output_type(task_name) == 'multilabel':
            mask_path = self._save_artery_vein_mask(
                prediction, task_name, paths, image_data
            )
        elif is_lesion_multilabel:
            mask_path = self._save_multilabel_lesion_masks(
                prediction, task_name, paths, image_data, task_idx=task_idx
            )
        else:
            mask_path = self._save_multiclass_mask(
                prediction, task_name, paths, image_data
            )
            
        return mask_path
        
    def save_grouped_lesion_masks(self, lesion_task_masks: Dict[str, np.ndarray], 
                                 output_dir: str, image_data: Dict, save_visuals: bool = True,
                                 save_masks: bool = True) -> Dict[str, Union[str, np.ndarray]]:
        """
        Save grouped lesion masks and return their paths for analysis.
        
        Args:
            lesion_task_masks: Dictionary of lesion task masks
            output_dir: Output directory
            image_data: Image metadata
            
        Returns:
            Dictionary of grouped lesion mask paths
        """
        masks_dir = os.path.join(output_dir, 'masks')
        if save_masks:
            os.makedirs(masks_dir, exist_ok=True)
        
        grouped_paths = {}
        allowed_groups = set(self.config.get_grouped_lesion_names())
        if not allowed_groups:
            return grouped_paths
        
        lesion_s1_mask = lesion_task_masks.get('lesion_s1')
        lesion_s2_mask = lesion_task_masks.get('lesion_s2')
        lesion_s3_mask = lesion_task_masks.get('lesion_s3')

        # lesion_dr from lesion_s1: channels 1,2,3 -> 1..3
        if lesion_s1_mask is not None and 'lesion_dr' in allowed_groups:
            lesion_dr = np.zeros_like(lesion_s1_mask, dtype=np.uint8)
            lesion_dr[lesion_s1_mask == 1] = 1  # hemorrhage
            lesion_dr[lesion_s1_mask == 2] = 2  # exudate
            lesion_dr[lesion_s1_mask == 3] = 3  # cotton wool spot
            if save_masks:
                mask_path = os.path.join(masks_dir, 'lesion_dr_mask.png')
                cv2.imwrite(mask_path, lesion_dr.astype(np.uint8))
                grouped_paths['lesion_dr'] = mask_path
                # Also save per-category masks for analyzer
                for class_id, name in self.config.get_grouped_lesion_types('lesion_dr').items():
                    bin_mask = (lesion_dr == int(class_id)).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(masks_dir, f'lesion_dr_{name}.png'), bin_mask)
            else:
                grouped_paths['lesion_dr'] = lesion_dr

        # lesion_amd: s1 ch4 ->1 (drusen only in the public default setup)
        if ((lesion_s1_mask is not None) or (lesion_s3_mask is not None)) and 'lesion_amd' in allowed_groups:
            base = lesion_s1_mask if lesion_s1_mask is not None else lesion_s3_mask
            lesion_amd = np.zeros_like(base, dtype=np.uint8)
            if lesion_s1_mask is not None:
                lesion_amd[lesion_s1_mask == 4] = 1  # drusen
            if lesion_s3_mask is not None:
                lesion_amd[lesion_s3_mask == 5] = 2  # patch hemorrhage
            if save_masks:
                mask_path = os.path.join(masks_dir, 'lesion_amd_mask.png')
                cv2.imwrite(mask_path, lesion_amd.astype(np.uint8))
                grouped_paths['lesion_amd'] = mask_path
                # Also save per-category masks for analyzer
                for class_id, name in self.config.get_grouped_lesion_types('lesion_amd').items():
                    bin_mask = (lesion_amd == int(class_id)).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(masks_dir, f'lesion_amd_{name}.png'), bin_mask)
            else:
                grouped_paths['lesion_amd'] = lesion_amd

        # lesion_others: s2 ch1->1, s2 ch5->2, s3 ch4->3, s2 ch2->4 (macular hole)
        if ((lesion_s2_mask is not None) or (lesion_s3_mask is not None)) and 'lesion_others' in allowed_groups:
            base = lesion_s2_mask if lesion_s2_mask is not None else lesion_s3_mask
            lesion_others = np.zeros_like(base, dtype=np.uint8)
            if lesion_s2_mask is not None:
                lesion_others[lesion_s2_mask == 1] = 1  # epiretinal membrane
                lesion_others[lesion_s2_mask == 5] = 2  # artifact
                lesion_others[lesion_s2_mask == 2] = 4  # macular hole moved from AMD
            if lesion_s3_mask is not None:
                lesion_others[lesion_s3_mask == 4] = 3  # retinal scar
            if save_masks:
                mask_path = os.path.join(masks_dir, 'lesion_others_mask.png')
                cv2.imwrite(mask_path, lesion_others.astype(np.uint8))
                grouped_paths['lesion_others'] = mask_path
                # Also save per-category masks for analyzer
                for class_id, name in self.config.get_grouped_lesion_types('lesion_others').items():
                    bin_mask = (lesion_others == int(class_id)).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(masks_dir, f'lesion_others_{name}.png'), bin_mask)
            else:
                grouped_paths['lesion_others'] = lesion_others

        # Build possible_lesions from dedicated head + remapped classes (myelinated nerve fiber, venous tortuosity)
        poss = lesion_task_masks.get('possible_lesions')
        base_mask = poss if poss is not None else (lesion_s1_mask if lesion_s1_mask is not None else (lesion_s2_mask if lesion_s2_mask is not None else lesion_s3_mask))
        if base_mask is not None and 'possible_lesions' in allowed_groups:
            possible_mask = np.zeros_like(base_mask, dtype=np.uint8)
            if poss is not None:
                possible_mask[poss == 1] = 1  # other lesions
            if lesion_s2_mask is not None:
                possible_mask[lesion_s2_mask == 9] = 2  # myelinated nerve fiber
            if lesion_s3_mask is not None:
                possible_mask[lesion_s3_mask == 1] = 3  # venous tortuosity
            if np.any(possible_mask):
                if save_masks:
                    mask_path = os.path.join(masks_dir, 'possible_lesions_mask.png')
                    cv2.imwrite(mask_path, possible_mask.astype(np.uint8))
                    grouped_paths['possible_lesions'] = mask_path
                    for class_id, name in self.config.get_grouped_lesion_types('possible_lesions').items():
                        bin_mask = (possible_mask == int(class_id)).astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(masks_dir, f'possible_lesions_{name}.png'), bin_mask)
                else:
                    grouped_paths['possible_lesions'] = possible_mask

        return grouped_paths
        
    def create_combined_visualization(self, predictions: List[torch.Tensor],
                                    original_image_path: str,
                                    output_dir: str,
                                    image_data: Dict,
                                    save_visuals: bool = True) -> Optional[str]:
        """
        Create combined visualization with all predictions overlaid on one image.
        
        Args:
            predictions: List of model predictions
            original_image_path: Path to original image
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization
        """
        # Load original image
        original = cv2.imread(original_image_path)
        height, width = original.shape[:2]
        
        # Create directories for visualizations and masks
        vis_dir = os.path.join(output_dir, 'visualizations')
        if save_visuals:
            os.makedirs(vis_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # Start with original image for combined overlay
        combined = original.copy()

        # If last element is a coordinate tensor (shape [B, C] or [C] with even C), keep aside
        coord_tensor = None
        if len(predictions) > 0 and torch.is_tensor(predictions[-1]) and predictions[-1].dim() in (1, 2):
            if (predictions[-1].shape[-1] % 2) == 0:
                coord_tensor = predictions[-1]
                predictions = predictions[:-1]
        
        # Collect lesion task masks for regrouped visualization
        lesion_task_masks: Dict[str, np.ndarray] = {}
        grouped_paths: Dict[str, Union[str, np.ndarray]] = {}
        lesion_base_tasks = {'lesion_s1', 'lesion_s2', 'lesion_s3'}
        for idx, prediction in enumerate(predictions):
            # Get task name and process prediction
            task_name = self.config.get_task_name(idx)
            is_legacy_lesion_task = task_name in lesion_base_tasks
            
            if not is_legacy_lesion_task and save_visuals:
                # Create individual visualization for saving
                individual_viz = self._create_task_visualization(
                    prediction, task_name, original, (width, height), image_data, task_idx=idx
                )
                task_display_name = self.config.get_display_name(task_name)
                vis_path = os.path.join(vis_dir, f'{task_display_name}_overlay.png')
                cv2.imwrite(vis_path, individual_viz)
            
            # Add this task's prediction to combined overlay
            crop = image_data.get('crop_coords', None)
            if crop:
                cropped_h = int(crop['ymax']) - int(crop['ymin'])
                cropped_w = int(crop['xmax']) - int(crop['xmin'])
                task_mask_cropped = self._get_task_mask(
                    prediction, task_name, (cropped_h, cropped_w), task_idx=idx
                )
                task_mask = self._place_on_original_canvas(task_mask_cropped, image_data)
            else:
                task_mask = self._get_task_mask(prediction, task_name, (height, width), task_idx=idx)
            # Collect lesion task masks for regrouping
            if task_name in ['lesion_s1', 'lesion_s2', 'lesion_s3', 'possible_lesions']:
                lesion_task_masks[task_name] = task_mask.copy()
            # Skip adding raw lesion sub-tasks to combined; regrouped overlays will be added later
            if is_legacy_lesion_task:
                continue
            if save_visuals:
                # Determine alpha from config using internal task name
                alpha = self.config.get_overlay_alpha(task_name)
                combined = self._add_task_to_combined(combined, task_mask, task_name, alpha)

        # After per-task processing, create regrouped lesion masks and overlays
        if any(k in lesion_task_masks for k in ['lesion_s1', 'lesion_s2', 'lesion_s3', 'possible_lesions']):
            lesion_s1_mask = lesion_task_masks.get('lesion_s1', None)
            lesion_s2_mask = lesion_task_masks.get('lesion_s2', None)
            lesion_s3_mask = lesion_task_masks.get('lesion_s3', None)
            possible_lesion_head = lesion_task_masks.get('possible_lesions', None)

            def save_group(name: str, mask_int: np.ndarray):
                nonlocal combined, grouped_paths
                # Mask saving for grouped results is handled in save_grouped_lesion_masks; here we only handle overlays
                grouped_paths[name] = mask_int
                if not save_visuals:
                    return
                # Save overlay
                overlay = original.copy()
                colors = TASK_COLOR_MAP.get(name, {})
                for class_id, color in colors.items():
                    if class_id == 0:
                        continue
                    m = (mask_int == class_id)
                    if np.any(m):
                        overlay[m] = (
                            overlay[m] * (1 - self.config.overlay_alpha_default)
                            + np.array((color[2], color[1], color[0])) * self.config.overlay_alpha_default
                        )
                overlay_path = os.path.join(vis_dir, f'{name}_overlay.png')
                cv2.imwrite(overlay_path, overlay)
                # Also push regrouped result into combined visualization
                alpha = self.config.get_overlay_alpha(name)
                combined = self._add_task_to_combined(combined, mask_int, name, alpha)

        # lesion_dr from lesion_s1: channels 1,2,3 -> 1..3
        if lesion_s1_mask is not None:
            lesion_dr = np.zeros_like(lesion_s1_mask, dtype=np.uint8)
            lesion_dr[lesion_s1_mask == 1] = 1  # hemorrhage
            lesion_dr[lesion_s1_mask == 2] = 2  # exudate
            lesion_dr[lesion_s1_mask == 3] = 3  # cotton wool spot
            save_group('lesion_dr', lesion_dr)

        # lesion_amd: s1 ch4 ->1 (drusen only in the public default setup)
        if (lesion_s1_mask is not None) or (lesion_s3_mask is not None):
            base = lesion_s1_mask if lesion_s1_mask is not None else lesion_s3_mask
            lesion_amd = np.zeros_like(base, dtype=np.uint8)
            if lesion_s1_mask is not None:
                lesion_amd[lesion_s1_mask == 4] = 1  # drusen
            if lesion_s3_mask is not None:
                lesion_amd[lesion_s3_mask == 5] = 2  # patch hemorrhage
            save_group('lesion_amd', lesion_amd)

        # lesion_others: s2 ch1->1, s2 ch5->2, s3 ch4->3, s2 ch2->4 (macular hole moved here)
        if (lesion_s2_mask is not None) or (lesion_s3_mask is not None):
            base = lesion_s2_mask if lesion_s2_mask is not None else lesion_s3_mask
            lesion_others = np.zeros_like(base, dtype=np.uint8)
            if lesion_s2_mask is not None:
                lesion_others[lesion_s2_mask == 1] = 1  # epiretinal membrane
                lesion_others[lesion_s2_mask == 5] = 2  # artifact
                lesion_others[lesion_s2_mask == 2] = 4  # macular hole moved from AMD
            if lesion_s3_mask is not None:
                lesion_others[lesion_s3_mask == 4] = 3  # retinal scar
            save_group('lesion_others', lesion_others)

            # possible_lesions: dedicated head + remapped classes (myelinated nerve fiber, venous tortuosity)
            base_for_possible = possible_lesion_head
            if base_for_possible is None:
                base_for_possible = lesion_s1_mask if lesion_s1_mask is not None else (lesion_s2_mask if lesion_s2_mask is not None else lesion_s3_mask)
            if base_for_possible is not None:
                possible_mask = np.zeros_like(base_for_possible, dtype=np.uint8)
                if possible_lesion_head is not None:
                    possible_mask[possible_lesion_head == 1] = 1  # other lesions
                if lesion_s2_mask is not None:
                    possible_mask[lesion_s2_mask == 9] = 2  # myelinated nerve fiber
                if lesion_s3_mask is not None:
                    possible_mask[lesion_s3_mask == 1] = 3  # venous tortuosity
                if np.any(possible_mask):
                    save_group('possible_lesions', possible_mask)

        if not save_visuals:
            return None
        # Save combined visualization after regrouped overlays are merged
        output_path = os.path.join(output_dir, 'combined_predictions.png')
        cv2.imwrite(output_path, combined)
        Logger.saved("combined overlay visualization", output_path)
        Logger.saved("task overlays", vis_dir)

        # Save coordinate overlay if provided
        if coord_tensor is not None:
            coord_overlay = original.copy()
            coords_np = coord_tensor.squeeze().detach().cpu().numpy()
            # Ensure 1D
            if coords_np.ndim > 1:
                coords_np = coords_np[0]
            # If normalized [0,1], map to pixel coords (heuristic)
            is_normalized = coords_np.max() <= 2.0
            for i in range(0, len(coords_np), 2):
                if is_normalized:
                    x = int(coords_np[i] * width)
                    y = int(coords_np[i+1] * height)
                else:
                    x = int(coords_np[i])
                    y = int(coords_np[i+1])
                self._draw_red_cross(coord_overlay, (x, y))

            coord_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(coord_dir, exist_ok=True)
            coord_path = os.path.join(coord_dir, 'coordinates_overlay.png')
            cv2.imwrite(coord_path, coord_overlay)
            Logger.saved("coordinate overlay", coord_path)
        

        return output_path
        
    def create_overlay_visualization(self, mask: np.ndarray, original: np.ndarray,
                                   task_name: str, alpha: float = None) -> np.ndarray:
        """
        Create overlay visualization of mask on original image.
        
        Args:
            mask: Segmentation mask
            original: Original image
            task_name: Task name for color selection
            alpha: Transparency for overlay
            
        Returns:
            Overlay visualization
        """
        overlay = original.copy()
        # If alpha not provided, read from config
        if alpha is None:
            alpha = self.config.get_overlay_alpha(task_name)
        
        # Use per-class color mapping (same scheme as combined overlay)
        colors = TASK_COLOR_MAP.get(task_name, {1: (255, 255, 255)})
        for class_id, color in colors.items():
            if class_id == 0:
                continue
            pixels = (mask == class_id)
            if np.any(pixels):
                # Convert RGB -> BGR for OpenCV image
                bgr = (color[2], color[1], color[0])
                overlay[pixels] = overlay[pixels] * (1 - alpha) + np.array(bgr) * alpha
        
        return overlay.astype(np.uint8)
        
    def _save_multiclass_mask(self, prediction: torch.Tensor, task_name: str,
                             paths: Dict, image_data: Dict) -> str:
        """Save multiclass segmentation mask."""
        # Convert prediction to numpy
        pred_np = prediction.squeeze().cpu().numpy()
        
        # Apply argmax if multi-channel
        if len(pred_np.shape) == 3:
            mask = np.argmax(pred_np, axis=0)
            num_channels = pred_np.shape[0]
        else:
            mask = pred_np
            num_channels = 1
            
        # Map back to original canvas using crop coordinates
        mask_resized = self._place_on_original_canvas(mask.astype(np.uint8), image_data)
        
        # Save mask
        display_name = self.config.get_display_name(task_name)
        mask_path = os.path.join(paths['masks'], f'{display_name}_mask.png')
        cv2.imwrite(mask_path, mask_resized)
        
        # Do not save individual channels anymore
        
        Logger.saved(f"{task_name} mask", mask_path)
        
        return mask_path

    def _save_artery_vein_mask(self, prediction: torch.Tensor, task_name: str,
                              paths: Dict, image_data: Dict) -> str:
        """Save multilabel artery/vein mask with overlap marked as 3."""
        # Decode multilabel AV mask: round two channels independently
        pred_sig = torch.sigmoid(prediction.squeeze())
        pred_bin = torch.round(pred_sig).detach().cpu().numpy()
        if pred_bin.ndim == 2:
            artery_bin = pred_bin
            vein_bin = np.zeros_like(artery_bin, dtype=np.uint8)
        else:
            artery_bin = pred_bin[0].astype(np.uint8)
            vein_bin = pred_bin[1].astype(np.uint8) if pred_bin.shape[0] > 1 else np.zeros_like(artery_bin, dtype=np.uint8)

        mask = np.zeros_like(artery_bin, dtype=np.uint8)
        mask[artery_bin == 1] = 1
        mask[vein_bin == 1] = 2
        mask[(artery_bin == 1) & (vein_bin == 1)] = 3  # overlap = 3

        # Map back to original canvas using crop coordinates
        mask_resized = self._place_on_original_canvas(mask.astype(np.uint8), image_data)

        display_name = self.config.get_display_name(task_name)
        mask_path = os.path.join(paths['masks'], f'{display_name}_mask.png')
        cv2.imwrite(mask_path, mask_resized)
        Logger.saved(f"{task_name} mask", mask_path)
        return mask_path
        
    def _save_multilabel_lesion_masks(self, prediction: torch.Tensor, task_name: str,
                                     paths: Dict, image_data: Dict, task_idx: int = -1) -> str:
        """Save multilabel lesion masks with individual channels."""
        
        # For task 5 (s2, 11-channel), filter prediction BEFORE processing
        if task_idx == 5:  # This is the 11-channel lesion_s2 task
            # Clone and filter the prediction tensor
            filtered_pred = torch.zeros_like(prediction)
            # Save only selected channels for lesion_all (exclude background): 1,2,5,9
            keep_channels = [1, 2, 5, 9]
            for ch in keep_channels:
                if ch < prediction.shape[1]:  # Check channel dimension
                    filtered_pred[:, ch, :, :] = prediction[:, ch, :, :]
            prediction = filtered_pred
        
        # Process multilabel prediction
        pred_mask, pred_binary = self.mask_processor.process_multilabel_prediction(prediction)
        
        # Map back to original canvas using crop coordinates
        mask_resized = self._place_on_original_canvas(pred_mask.astype(np.uint8), image_data)
        
        # Save combined mask
        display_name = self.config.get_display_name(task_name)
        mask_path = os.path.join(paths['masks'], f'{display_name}_mask.png')
        cv2.imwrite(mask_path, mask_resized)
        
        # Do not save individual lesion channels anymore
                
        Logger.saved(f"{task_name} mask", mask_path)
        
        return mask_path
        
    def _create_task_visualization(self, prediction: torch.Tensor, task_name: str,
                                 original: np.ndarray, target_size: Tuple[int, int], image_data: Dict, task_idx: int = -1) -> np.ndarray:
        """Create visualization for a single task."""
        # For task 5 (s2, 11-channel), zero out unwanted channels before processing
        if task_idx == 5:  # This is the 11-channel lesion_s2 task
            # Clone prediction to avoid modifying original
            prediction = prediction.clone()
            # Only keep channels 1, 2, 5, 9 (set others to 0)
            keep_channels = [1, 2, 5, 9]
            for i in range(prediction.shape[1] if len(prediction.shape) > 3 else prediction.shape[0]):
                if i not in keep_channels:
                    if len(prediction.shape) > 3:
                        prediction[:, i, :, :] = 0
                    else:
                        prediction[i, :, :] = 0
        
        # Build mask using the same logic as combined overlay to keep consistency with saved masks
        crop = image_data.get('crop_coords', None)
        if crop:
            cropped_h = int(crop['ymax']) - int(crop['ymin'])
            cropped_w = int(crop['xmax']) - int(crop['xmin'])
            task_mask_cropped = self._get_task_mask(
                prediction, task_name, (cropped_h, cropped_w), task_idx=task_idx
            )
            task_mask = self._place_on_original_canvas(task_mask_cropped, image_data)
        else:
            task_mask = self._get_task_mask(
                prediction, task_name, (original.shape[0], original.shape[1]), task_idx=task_idx
            )
        # Create overlay
        overlay = self.create_overlay_visualization(task_mask, original, task_name)
        
        # Resize to target size
        viz_resized = cv2.resize(overlay, target_size)

        # Return without drawing task label text
        return viz_resized
        
    def _get_task_mask(self, prediction: torch.Tensor, task_name: str, 
                      target_size: Tuple[int, int], task_idx: int = -1) -> np.ndarray:
        """
        Extract mask from prediction for a specific task.
        
        Args:
            prediction: Model prediction tensor
            task_name: Name of the task
            target_size: Target (height, width)
            task_idx: Task index for special handling
            
        Returns:
            Processed mask as numpy array
        """
        # Special handling: artery_vein is now 2-channel multilabel (artery, vein)
        if task_name == 'artery_vein':
            pred_sig = torch.sigmoid(prediction.squeeze())
            pred_bin = torch.round(pred_sig).detach().cpu().numpy()
            if pred_bin.ndim == 2:
                artery_bin = pred_bin
                vein_bin = np.zeros_like(artery_bin, dtype=np.uint8)
            else:
                artery_bin = pred_bin[0].astype(np.uint8)
                vein_bin = pred_bin[1].astype(np.uint8) if pred_bin.shape[0] > 1 else np.zeros_like(artery_bin, dtype=np.uint8)

            mask = np.zeros_like(artery_bin, dtype=np.uint8)
            mask[artery_bin == 1] = 1
            mask[vein_bin == 1] = 2
            mask[(artery_bin == 1) & (vein_bin == 1)] = 3  # overlap

            height, width = target_size
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
            return mask_resized

        # For task 5 (s2), filter channels before processing
        if task_idx == 5:  # This is the 11-channel lesion_s2 task
            # Clone prediction to avoid modifying original
            prediction = prediction.clone()
            # Only keep channels 1, 2, 5, 9 (set others to 0)
            keep_channels = [1, 2, 5, 9]
            for i in range(prediction.shape[1] if len(prediction.shape) > 3 else prediction.shape[0]):
                if i not in keep_channels:
                    if len(prediction.shape) > 3:
                        prediction[:, i, :, :] = 0
                    else:
                        prediction[i, :, :] = 0
        
        # Convert prediction to numpy
        pred_np = prediction.squeeze().cpu().numpy()
        
        # Process based on prediction shape
        if len(pred_np.shape) == 3:
            # Decide output type with override for lesion_s2 (task_idx 5)
            output_type = self.config.get_output_type(task_name)
            if task_idx == 5 or self.config.get_display_name(task_name) == 'lesion_s2':
                output_type = 'multilabel'
            
            if output_type == 'multilabel':
                # For multilabel tasks, threshold probabilities at 0.5 (sigmoid) to match saved masks
                mask = np.zeros(pred_np.shape[1:], dtype=np.uint8)
                for i in range(pred_np.shape[0]):
                    if i == 0:
                        continue  # background
                    probs = 1.0 / (1.0 + np.exp(-pred_np[i]))
                    channel_mask = (probs > 0.5).astype(np.uint8)
                    mask = np.maximum(mask, (channel_mask * i).astype(np.uint8))
            else:
                # For multiclass tasks, use argmax
                mask = np.argmax(pred_np, axis=0).astype(np.uint8)
        else:
            # Single channel (treat as probability map): sigmoid > 0.5
            probs = 1.0 / (1.0 + np.exp(-pred_np))
            mask = (probs > 0.5).astype(np.uint8)
            
        # Resize to target size
        height, width = target_size
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )
        
        return mask_resized
        
    def _add_task_to_combined(self, combined_image: np.ndarray, task_mask: np.ndarray,
                             task_name: str, alpha: float) -> np.ndarray:
        """
        Add a task's segmentation results to the combined visualization.
        
        Args:
            combined_image: Current combined image
            task_mask: Mask for this task
            task_name: Name of the task
            alpha: Alpha value for blending
            
        Returns:
            Updated combined image
        """
        # Create a copy to avoid modifying input
        result = combined_image.copy()
        
        # Get colors for this task
        colors = TASK_COLOR_MAP.get(task_name, {1: (255, 255, 255)})  # Default white
        
        # Apply colors based on mask values
        for class_id, color in colors.items():
            if class_id == 0:  # Skip background class
                continue
            mask_pixels = (task_mask == class_id)
            if np.any(mask_pixels):
                # Convert RGB -> BGR for OpenCV image
                bgr = (color[2], color[1], color[0])
                result[mask_pixels] = result[mask_pixels] * (1 - alpha) + np.array(bgr) * alpha
                                    
        return result.astype(np.uint8)

    @staticmethod
    def _draw_red_cross(image: np.ndarray, center: Tuple[int, int], size: int = 12, thickness: int = 2):
        x, y = center
        color = (0, 0, 255)
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)
