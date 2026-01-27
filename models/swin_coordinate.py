#!/usr/bin/env python3
"""
Coordinate prediction model based on SwinTransformer.

This model is designed to predict keypoint coordinates in images, such as the
optic disc center and macula center in fundus images.
Architecture: SwinTransformer encoder + coordinate prediction head.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing_extensions import Final

from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.nets.swin_unetr import SwinTransformer

__all__ = ["SwinCoordinate", "CoordinateHead"]


class CoordinateHead(nn.Module):
    """
    Coordinate prediction head for regressing points from feature maps.
    """
    def __init__(
        self,
        feature_size: int = 768,  # Feature dimension of the last SwinTransformer stage
        hidden_size: int = 512,
        num_coordinates: int = 2,  # Predict x,y by default
        dropout: float = 0.1,
        use_attention: bool = False  # Whether to use attention
    ):
        super().__init__()
        
        self.num_coordinates = num_coordinates
        self.use_attention = use_attention
        
        if use_attention:
            # Coordinate prediction with attention
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            # Learnable query vectors
            self.coord_queries = nn.Parameter(torch.randn(num_coordinates // 2, feature_size))
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Coordinate regression network
        self.coordinate_regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_coordinates),
            nn.Sigmoid()  # Output normalized coordinates in [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature maps [B, C, H, W]
        Returns:
            coordinates: Predicted coordinates [B, num_coordinates] in [0, 1]
        """
        batch_size = x.size(0)
        
        if self.use_attention:
            # Use attention
            # Reshape feature maps: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
            
            # Expand query vectors
            queries = self.coord_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_points, C]
            
            # Attention computation
            attended_features, _ = self.attention(queries, x_flat, x_flat)  # [B, num_points, C]
            
            # Predict coordinates for each query vector
            coordinates_list = []
            for i in range(attended_features.size(1)):
                point_feature = attended_features[:, i, :]  # [B, C]
                point_coords = self.coordinate_regressor(point_feature)  # [B, 2]
                coordinates_list.append(point_coords)
            
            coordinates = torch.cat(coordinates_list, dim=1)  # [B, num_coordinates]
        else:
            # Use global average pooling
            # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
            pooled = self.global_pool(x)
            # Flatten: [B, C, 1, 1] -> [B, C]
            flattened = pooled.view(batch_size, -1)
            # Coordinate regression: [B, C] -> [B, num_coordinates]
            coordinates = self.coordinate_regressor(flattened)
        
        return coordinates


class SwinCoordinate(nn.Module):
    """
    SwinTransformer-based coordinate prediction model.
    """

    patch_size: Final[int] = 2

    def __init__(
        self,
        in_channels: int,
        num_coordinates: int = 2,  # Number of coordinates to predict
        patch_size: int = 2,
        window_size: int = 7,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample: str = "merging",
        use_v2: bool = False,
        # Coordinate head parameters
        coordinate_hidden_size: int = 512,
        coordinate_dropout: float = 0.1,
        use_attention_head: bool = False,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels
            num_coordinates: Number of coordinates to predict (even number, e.g. 2 for one x,y point)
            patch_size: Patch size
            window_size: Window size
            depths: Number of blocks per stage
            num_heads: Number of attention heads
            feature_size: Feature dimension
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate
            normalize: Whether to normalize intermediate features
            use_checkpoint: Whether to use gradient checkpointing
            spatial_dims: Number of spatial dimensions
            downsample: Downsampling method
            use_v2: Whether to use swinunetr_v2
            coordinate_hidden_size: Hidden size of the coordinate head
            coordinate_dropout: Dropout rate of the coordinate head
            use_attention_head: Whether to use attention in the coordinate head
        """
        super().__init__()

        # Parameter validation
        if num_coordinates % 2 != 0:
            raise ValueError("num_coordinates should be even (x,y pairs)")
        
        patch_sizes = ensure_tuple_rep(patch_size, spatial_dims)
        window_sizes = ensure_tuple_rep(window_size, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")
        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.normalize = normalize
        self.num_coordinates = num_coordinates
        
        # SwinTransformer encoder
        self.swin_encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_sizes,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=self._get_downsample_layer(downsample),
            use_v2=use_v2,
        )
        
        # Compute the final feature dimension
        final_feature_dim = feature_size * (2 ** (len(depths) - 1))
        
        # Coordinate prediction head
        self.coordinate_head = CoordinateHead(
            feature_size=final_feature_dim,
            hidden_size=coordinate_hidden_size,
            num_coordinates=num_coordinates,
            dropout=coordinate_dropout,
            use_attention=use_attention_head
        )

    def _get_downsample_layer(self, downsample):
        """Get the downsampling layer."""
        from monai.networks.nets.swin_unetr import PatchMerging, PatchMergingV2
        MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
        
        if isinstance(downsample, str):
            return look_up_option(downsample, MERGING_MODE)
        else:
            return downsample

    @torch.jit.unused

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            coordinates: Predicted coordinates [B, num_coordinates] in [0, 1]
        """
        # SwinTransformer encoding
        hidden_states_out = self.swin_encoder(x, self.normalize)
        
        # Predict coordinates from the deepest features
        deepest_features = hidden_states_out[-1]  # Last stage features [B, C, H, W]
        
        # Coordinate prediction
        coordinates = self.coordinate_head(deepest_features)
        
        return coordinates

    def get_num_parameters(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze encoder parameters and train only the coordinate head."""
        for param in self.swin_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.swin_encoder.parameters():
            param.requires_grad = True


def create_swin_coordinate_model(config_name="default"):
    """
    Create a coordinate prediction model with a predefined configuration.
    
    Args:
        config_name: Configuration name, supports "default", "large", "tiny"
    
    Returns:
        SwinCoordinate model instance
    """
    configs = {
        "default": {
            "in_channels": 3,
            "num_coordinates": 4,  # 2 points (optic disc and macula)
            "patch_size": 4,
            "window_size": 10,
            "feature_size": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "coordinate_hidden_size": 512,
            "use_attention_head": False,
        },
        "large": {
            "in_channels": 3,
            "num_coordinates": 6,  # 3 points
            "patch_size": 4,
            "window_size": 10,
            "feature_size": 192,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "coordinate_hidden_size": 1024,
            "use_attention_head": True,
        },
        "tiny": {
            "in_channels": 3,
            "num_coordinates": 2,  # 1 point
            "patch_size": 4,
            "window_size": 7,
            "feature_size": 48,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "coordinate_hidden_size": 256,
            "use_attention_head": False,
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    return SwinCoordinate(**config)


# Loss function
class CoordinateLoss(nn.Module):
    """Loss function for coordinate prediction."""
    
    def __init__(self, loss_type="smooth_l1", weight_coords=None):
        super().__init__()
        self.loss_type = loss_type
        self.weight_coords = weight_coords  # Weights for each coordinate
        
    def forward(self, pred_coords, target_coords):
        """
        Args:
            pred_coords: [B, num_coordinates] predicted coordinates
            target_coords: [B, num_coordinates] ground-truth coordinates
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred_coords, target_coords, reduction='none')
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred_coords, target_coords, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_coords, target_coords, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Apply weights
        if self.weight_coords is not None:
            weight_tensor = torch.tensor(self.weight_coords, device=loss.device)
            loss = loss * weight_tensor.unsqueeze(0)
        
        return loss.mean()


if __name__ == "__main__":
    print("🔬 SwinCoordinate coordinate prediction model test")
    print("=" * 50)
    
    # Test different configs
    configs_to_test = ["tiny", "default", "large"]
    
    for config_name in configs_to_test:
        print(f"\n📋 Test config: {config_name}")
        try:
            model = create_swin_coordinate_model(config_name)
            model.eval()
            
            # Test input
            x = torch.randn(2, 3, 640, 640)
            
            # Forward pass
            with torch.no_grad():
                coordinates = model(x)
            
            # Output info
            print("  ✅ Model created successfully")
            print(f"  📥 Input shape: {x.shape}")
            print(f"  📤 Output shape: {coordinates.shape}")
            print(f"  📊 Parameter count: {model.get_num_parameters():,}")
            print(f"  🎯 Number of predicted coordinates: {model.num_coordinates}")
            print(f"  📍 Coordinate range: [{coordinates.min().item():.3f}, {coordinates.max().item():.3f}]")
            
            # Parse coordinates
            coords = coordinates[0].cpu().numpy()
            num_points = len(coords) // 2
            print(f"  🔍 Predicted {num_points} keypoints:")
            for i in range(num_points):
                x_coord, y_coord = coords[i*2], coords[i*2+1]
                print(f"    Point {i+1}: ({x_coord:.3f}, {y_coord:.3f})")
                
        except Exception as e:
            print(f"  ❌ Config {config_name} test failed: {e}")
    
    # Test loss function
    print("\n🎯 Testing loss function:")
    loss_fn = CoordinateLoss(loss_type="smooth_l1")
    pred = torch.rand(4, 4)  # 4 samples, 4 coordinates
    target = torch.rand(4, 4)
    loss = loss_fn(pred, target)
    print(f"  Coordinate loss: {loss.item():.6f}")
    
    print("\n📋 Usage notes:")
    print("1. Use create_swin_coordinate_model() to create a predefined config")
    print("2. num_coordinates must be even (x,y pairs)")
    print("3. Output coordinates are in [0,1]; multiply by image size to get pixels")
    print("4. Use freeze_encoder() to train only the coordinate head")
    print("5. Attention-based coordinate head is supported (use_attention_head=True)")