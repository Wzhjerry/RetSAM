"""
Model loading utilities for inference pipeline.
"""

import torch
import warnings
from typing import Optional, Tuple, Union
import sys
import os

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.swin_multitask import Swin_Multitask
from models.swin_multitask_final import Swin_Multitask_Final
from .config import InferenceConfig
from .utils import Logger


class ModelLoader:
    """Handles model loading and initialization."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize model loader.
        
        Args:
            device: Device to load model on
        """
        self.device = device
        self.config = InferenceConfig()
        self.last_loaded_path = None
        
        # Suppress PyTorch warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch._classes")
        warnings.filterwarnings("ignore", message=".*__path__._path.*")
        
    def load(self, model_path: str, multitask: bool = True, 
             output_channels: Optional[Tuple[int, ...]] = None,
             has_coordinate_head: Optional[bool] = None,
             num_coordinates: Optional[int] = None) -> torch.nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            multitask: Whether to load multitask model
            output_channels: Output channel configuration
            
        Returns:
            Loaded model ready for inference
        """
        self.last_loaded_path = model_path
        
        # Load checkpoint
        checkpoint = self._load_checkpoint(model_path)
        
        # Clean state dict
        state_dict = self._clean_state_dict(checkpoint)
        # Detect coordinate head (can be overridden by caller)
        detected_has_coordinate = any('coordinate_head' in k for k in state_dict.keys())
        self.has_coordinate = detected_has_coordinate if has_coordinate_head is None else has_coordinate_head
        # Try to read num_coordinates from hyper_parameters if present
        # Determine num_coordinates (override > checkpoint > None)
        self.num_coordinates = None
        if num_coordinates is not None:
            self.num_coordinates = int(num_coordinates)
        elif isinstance(checkpoint, dict) and 'hyper_parameters' in checkpoint:
            try:
                self.num_coordinates = checkpoint['hyper_parameters'].get('num_coordinates', None)
            except Exception:
                self.num_coordinates = None

        # Do NOT override feature_size; encoder feature_size is shared and fixed by config
        
        # Determine output channels
        if output_channels is None:
            output_channels = self.config.default_output_channels
            
        # Create model
        model = self._create_model(multitask, output_channels)
        
        # Load weights
        self._load_weights(model, state_dict)
        
        # Prepare for inference
        model = model.to(self.device)
        model.eval()
        
        Logger.success("Model loaded successfully")
        return model
        
    def _load_checkpoint(self, model_path: str) -> dict:
        """Load checkpoint with fallback methods."""
        try:
            # Try loading with weights_only=True first
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to standard loading
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
        except Exception as e:
            Logger.warning(f"Standard loading failed: {e}")
            # Try safe unpickling
            checkpoint = self._safe_load_checkpoint(model_path)
            
        return checkpoint
        
    def _safe_load_checkpoint(self, model_path: str) -> dict:
        """Safe loading method for problematic checkpoints."""
        import pickle
        
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle problematic path classes
                if name == '__path__._path':
                    return str
                return super().find_class(module, name)
                
        try:
            with open(model_path, 'rb') as f:
                return SafeUnpickler(f).load()
        except Exception as e:
            Logger.error(f"Safe loading failed: {e}")
            # Final fallback
            return torch.load(model_path, map_location=self.device)
            
    def _clean_state_dict(self, checkpoint: dict) -> dict:
        """Clean and prepare state dictionary."""
        # Extract state dict if nested
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove module prefixes
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('model.', '')
            cleaned[new_key] = v
            
        return cleaned
        
    def _create_model(self, multitask: bool, output_channels: Union[Tuple[int, ...], int]) -> torch.nn.Module:
        """Create model architecture."""
        params = self.config.model_params.copy()
        params['out_channels'] = output_channels
        # Keep feature_size from config to share encoder across tasks consistently
        
        if not multitask:
            raise NotImplementedError("Single-task inference is disabled in this build; use multitask checkpoints.")
        
        if getattr(self, 'has_coordinate', False):
            # Use final model with optional coordinate prediction
            final_params = params.copy()
            final_params['enable_coordinate_prediction'] = True
            if self.num_coordinates is not None:
                final_params['num_coordinates'] = int(self.num_coordinates)
            model = Swin_Multitask_Final(**final_params)
        else:
            model = Swin_Multitask(**params)
            
        return model
        
    def _load_weights(self, model: torch.nn.Module, state_dict: dict) -> None:
        """Load weights into model."""
        # Proactively drop coordinate_head weights if shape mismatch with current model
        try:
            model_dict = model.state_dict()
            head_key = 'coordinate_head.coordinate_regressor.0.weight'
            if head_key in state_dict and head_key in model_dict:
                if tuple(state_dict[head_key].shape) != tuple(model_dict[head_key].shape):
                    # Remove all coordinate_head params to avoid size mismatch
                    keys_to_remove = [k for k in list(state_dict.keys()) if k.startswith('coordinate_head.') or k.startswith('model.coordinate_head.')]
                    for k in keys_to_remove:
                        state_dict.pop(k, None)
                    Logger.warning("Coordinate head shapes mismatch; dropped coordinate_head weights and will use randomly initialized head.")
        except Exception:
            pass

        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            Logger.warning(f"Model loading partially failed: {e}")
            # Load what we can
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)