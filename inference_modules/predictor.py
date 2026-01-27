"""
Prediction module for running model inference.
"""

import torch
from typing import Dict, List, Union, Optional
import torch.nn as nn

from .utils import Logger
from .postprocessing import NoiseFilter
from .config import InferenceConfig


class Predictor:
    """Handles model inference and prediction."""
    
    def __init__(self, device: str = 'cuda', enable_noise_filter: bool = True):
        """
        Initialize predictor.
        
        Args:
            device: Device to run predictions on
            enable_noise_filter: Whether to enable noise filtering for predictions
        """
        self.device = device
        self.enable_noise_filter = enable_noise_filter
        if enable_noise_filter:
            self.config = InferenceConfig()
            self.noise_filter = NoiseFilter(self.config)
        
    def predict(self, model: nn.Module, image_tensor: torch.Tensor, 
                task_names: Optional[List[str]] = None) -> List[torch.Tensor]:
        """
        Run model inference on image tensor with optional noise filtering.
        
        Args:
            model: Loaded model
            image_tensor: Preprocessed image tensor
            task_names: List of task names for filtering (required if noise filter is enabled)
            
        Returns:
            List of prediction tensors (one per task for multitask models)
        """
        model.eval()
        
        with torch.no_grad():
            # Move tensor to device
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            outputs = model(image_tensor)
            
            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                # Handle possible (seg_list, coordinates)
                if len(outputs) == 2 and isinstance(outputs[0], (list, tuple)) and torch.is_tensor(outputs[1]):
                    seg_predictions = list(outputs[0])
                    coordinates = outputs[1]
                    predictions = seg_predictions + [coordinates]
                else:
                    predictions = list(outputs)
            else:
                # Single task model - wrap in list
                predictions = [outputs]
            
            # Apply noise filtering if enabled
            if self.enable_noise_filter and task_names is not None:
                filtered_predictions = []
                lesion_context: Dict[str, torch.Tensor] = {}
                if task_names and 'od_oc' in task_names:
                    od_index = task_names.index('od_oc')
                    if od_index < len(predictions):
                        lesion_context['od_oc'] = predictions[od_index]
                # Only apply filtering to segmentation tasks (exclude trailing coordinates tensor if present)
                num_tasks = len(task_names)
                for i, pred in enumerate(predictions):
                    if i < num_tasks:
                        task_name = task_names[i]
                        context = lesion_context if self.config.is_lesion_task(task_name) else None
                        filtered_pred = self.noise_filter.filter_prediction(pred, task_name, context)
                        filtered_predictions.append(filtered_pred)
                    else:
                        # coordinates tensor
                        filtered_predictions.append(pred)
                    
                    # Log filtering info for lesion tasks
                    if self.config.is_lesion_task(task_name):
                        Logger.debug(f"Applied noise filtering to {task_name} (task {i})")
                        
                predictions = filtered_predictions
                
        return predictions
        
    def predict_batch(self, model: nn.Module, image_tensors: torch.Tensor,
                     batch_size: int = 8) -> List[List[torch.Tensor]]:
        """
        Run batch inference on multiple images.
        
        Args:
            model: Loaded model
            image_tensors: Batch of preprocessed image tensors
            batch_size: Batch size for processing
            
        Returns:
            List of predictions for each image
        """
        model.eval()
        all_predictions = []
        
        # Process in batches
        num_images = image_tensors.shape[0]
        num_batches = (num_images + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_images)
                
                batch = image_tensors[start_idx:end_idx].to(self.device)
                
                # Run inference
                outputs = model(batch)
                
                # Handle outputs
                if isinstance(outputs, (list, tuple)):
                    # Multitask - transpose to get per-image predictions
                    batch_size_actual = batch.shape[0]
                    for img_idx in range(batch_size_actual):
                        img_predictions = [task_output[img_idx] for task_output in outputs]
                        all_predictions.append(img_predictions)
                else:
                    # Single task
                    for img_idx in range(outputs.shape[0]):
                        all_predictions.append([outputs[img_idx]])
                        
        return all_predictions
        
    @staticmethod
    def apply_activation(predictions: torch.Tensor, activation_type: str) -> torch.Tensor:
        """
        Apply appropriate activation function to predictions.
        
        Args:
            predictions: Raw model outputs
            activation_type: Type of activation ('sigmoid', 'softmax', or 'none')
            
        Returns:
            Activated predictions
        """
        if activation_type == 'sigmoid':
            return torch.sigmoid(predictions)
        elif activation_type == 'softmax':
            return torch.softmax(predictions, dim=0)
        else:
            return predictions