"""Post-processing utilities for inference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .config import InferenceConfig
from .utils import Logger


VERY_POSITIVE = 10.0
VERY_NEGATIVE = -10.0


@dataclass
class ComponentSizeResolver:
    """Resolve per-task minimum component sizes for post-processing."""

    config: InferenceConfig
    default_lesion_size: int = 5
    default_vessel_size: int = 10
    default_tessellation_size: int = 20
    default_myopia_size: int = 15

    def resolve(self, task_name: str) -> int:
        """Return the minimum connected-component size for the task."""

        try:
            return int(self.config.get_analysis_min_component_size(task_name))
        except Exception:
            if task_name == "artery_vein":
                return self.default_vessel_size
            if task_name == "tessellation":
                return self.default_tessellation_size
            if task_name == "myopia":
                return self.default_myopia_size
            if self.config.is_lesion_task(task_name):
                return self.default_lesion_size
            return 0


class ProbabilityGate:
    """Apply probability thresholds to filtered logits."""

    def __init__(self, config: InferenceConfig):
        self._config = config

    def apply(self, raw_pred: torch.Tensor, filtered_pred: torch.Tensor, task_name: str) -> torch.Tensor:
        """Apply probability gating using raw logits and filtered prediction."""

        threshold = float(self._config.get_prob_threshold(task_name))
        if threshold <= 0.0:
            return filtered_pred

        output_type = self._config.get_output_type(task_name)
        device = filtered_pred.device

        if output_type == "multilabel":
            probs = torch.sigmoid(raw_pred)
            gate = (probs > threshold).float()
            return filtered_pred * gate

        probs = torch.softmax(raw_pred, dim=1)
        max_prob, _ = torch.max(probs, dim=1, keepdim=True)
        keep = (max_prob > threshold).float().to(device)
        drop_mask = 1.0 - keep

        if torch.any(drop_mask > 0):
            gated = filtered_pred.clone()
            gated = gated * keep + VERY_NEGATIVE * drop_mask
            gated[:, 0, :, :] = gated[:, 0, :, :] * keep + VERY_POSITIVE * drop_mask
            return gated

        return filtered_pred


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` from a binary mask."""

    if min_size <= 0:
        return mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == label] = 1

    return filtered_mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in ``mask``."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    largest_label = 0
    largest_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    if largest_label > 0:
        filtered_mask[labels == largest_label] = 1

    return filtered_mask


def build_logits_from_class_mask(template: torch.Tensor, class_mask: np.ndarray) -> torch.Tensor:
    """Construct logits by assigning positives to pixels in ``class_mask``."""

    device = template.device
    filtered = template.clone()
    zeros = torch.zeros_like(template[0, 0])
    num_classes = template.shape[1]

    for channel in range(num_classes):
        filtered[0, channel] = zeros

    background_mask = torch.as_tensor(class_mask == 0, device=device)
    if background_mask.any():
        filtered[0, 0][background_mask] = VERY_POSITIVE

    for class_id in range(1, num_classes):
        class_bool = torch.as_tensor(class_mask == class_id, device=device)
        if class_bool.any():
            filtered[0, class_id][class_bool] = VERY_POSITIVE

    return filtered


class NoiseFilter:
    """Filter small noise from segmentation predictions."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._size_resolver = ComponentSizeResolver(config)
        self._probability_gate = ProbabilityGate(config)

    def filter_prediction(self, prediction: torch.Tensor, task_name: str,
                           context_predictions: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Apply noise filtering and probability gating to a prediction tensor."""

        raw_pred = prediction.clone()
        filtered = self._filter_by_task(prediction, task_name)

        if self.config.is_lesion_task(task_name) and context_predictions:
            od_mask = context_predictions.get("od_oc")
            if isinstance(od_mask, torch.Tensor):
                filtered = self._suppress_with_od_oc(filtered, od_mask, task_name)

        return self._probability_gate.apply(raw_pred, filtered, task_name)

    # ---------------------------------------------------------------------
    # Task routing helpers
    # ---------------------------------------------------------------------
    def _filter_by_task(self, prediction: torch.Tensor, task_name: str) -> torch.Tensor:
        if self.config.is_lesion_task(task_name):
            return self._filter_lesion_prediction(prediction, task_name)
        if task_name == "artery_vein":
            return self._filter_vessel_prediction(prediction)
        if task_name == "od_oc":
            return self._filter_disc_cup_prediction(prediction)
        if task_name in ("tessellation", "myopia"):
            return self._filter_regional_prediction(prediction, task_name)
        return prediction

    # ------------------------------------------------------------------
    # Lesion-specific filtering
    # ------------------------------------------------------------------
    def _filter_lesion_prediction(self, prediction: torch.Tensor, task_name: str) -> torch.Tensor:
        output_type = self.config.get_output_type(task_name)
        min_size = self._size_resolver.resolve(task_name)

        if output_type == "multilabel":
            filtered = self._filter_multilabel_lesion(prediction, min_size)
            Logger.debug(f"Filtered {task_name} multilabel lesions with min size {min_size}")
            return filtered

        filtered = self._filter_multiclass_by_min_size(prediction, min_size)
        Logger.debug(f"Filtered {task_name} multiclass lesions with min size {min_size}")
        return filtered

    def _filter_multilabel_lesion(self, prediction: torch.Tensor, min_size: int) -> torch.Tensor:
        device = prediction.device
        filtered = prediction.clone()
        probs = torch.sigmoid(prediction)

        for channel_idx in range(probs.shape[1]):
            channel_probs = probs[0, channel_idx]
            channel_mask = (channel_probs > 0.5).cpu().numpy().astype(np.uint8)
            cleaned_mask = remove_small_components(channel_mask, min_size)
            removed = channel_mask - cleaned_mask
            if np.any(removed):
                removal_mask = torch.as_tensor(removed.astype(bool), device=device)
                filtered[0, channel_idx][removal_mask] = VERY_NEGATIVE

        return filtered

    def _filter_multiclass_by_min_size(self, prediction: torch.Tensor, min_size: int) -> torch.Tensor:
        class_map = prediction[0].argmax(dim=0).cpu().numpy()
        filtered_mask = np.zeros_like(class_map, dtype=np.uint8)

        unique_classes = np.unique(class_map)
        unique_classes = unique_classes[unique_classes > 0]

        for class_id in unique_classes:
            class_mask = (class_map == class_id).astype(np.uint8)
            cleaned = remove_small_components(class_mask, min_size)
            if np.any(cleaned):
                filtered_mask[cleaned > 0] = class_id

        return build_logits_from_class_mask(prediction, filtered_mask)

    # ------------------------------------------------------------------
    # Other task filters
    # ------------------------------------------------------------------
    def _filter_vessel_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        # Artery/vein is multilabel (2 channels); remove tiny blobs per channel but keep overlaps.
        device = prediction.device
        filtered = prediction.clone()
        probs = torch.sigmoid(prediction)
        min_size = self._size_resolver.resolve("artery_vein")

        for channel_idx in range(min(probs.shape[1], 2)):
            channel_probs = probs[0, channel_idx]
            channel_mask = (channel_probs > 0.5).cpu().numpy().astype(np.uint8)
            cleaned_mask = remove_small_components(channel_mask, min_size=min_size)
            removed = channel_mask - cleaned_mask
            if np.any(removed):
                removal_mask = torch.as_tensor(removed.astype(bool), device=device)
                filtered[0, channel_idx][removal_mask] = VERY_NEGATIVE

        Logger.debug(f"Filtered vessels (multilabel) with min size {min_size}")
        return filtered

    def _filter_disc_cup_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        class_map = prediction[0].argmax(dim=0).cpu().numpy()
        filtered_mask = np.zeros_like(class_map, dtype=np.uint8)

        for class_id in (1, 2):
            class_mask = (class_map == class_id).astype(np.uint8)
            largest = keep_largest_component(class_mask)
            if np.any(largest):
                filtered_mask[largest > 0] = class_id

        Logger.debug("Filtered disc/cup keeping only largest components")
        return build_logits_from_class_mask(prediction, filtered_mask)

    def _filter_regional_prediction(self, prediction: torch.Tensor, task_name: str) -> torch.Tensor:
        min_size = self._size_resolver.resolve(task_name)
        filtered = self._filter_multiclass_by_min_size(prediction, min_size)
        Logger.debug(f"Filtered {task_name} with min size {min_size}")
        return filtered

    def _suppress_with_od_oc(self, prediction: torch.Tensor, od_prediction: torch.Tensor, task_name: str) -> torch.Tensor:
        """Remove lesion logits overlapping optic disc/cup segmentation."""

        od_classes = od_prediction.argmax(dim=1, keepdim=True)
        overlap = (od_classes > 0)
        if not overlap.any():
            return prediction

        overlap = overlap.to(prediction.device)
        if not overlap.any():
            return prediction

        cleaned = prediction.clone()
        mask = overlap.expand_as(cleaned)
        cleaned = cleaned.masked_fill(mask, VERY_NEGATIVE)

        if self.config.get_output_type(task_name) != "multilabel":
            cleaned[:, 0].masked_fill_(overlap[:, 0], VERY_POSITIVE)

        return cleaned

