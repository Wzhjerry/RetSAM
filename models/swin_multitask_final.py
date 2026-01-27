# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.nets.swin_unetr import SwinTransformer, SwinTransformerBlock

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "Swin_Multitask_Final",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "CoordinateHead",
]


class CoordinateHead(nn.Module):
    """Coordinate prediction head to regress point coordinates from feature maps."""
    def __init__(
        self,
        feature_size: int = 768,  # feature dim of last SwinTransformer layer
        hidden_size: int = 512,
        num_coordinates: int = 2,  # predict x,y by default
        dropout: float = 0.1,
        use_attention: bool = False  # whether to use attention head
    ):
        super().__init__()
        
        self.num_coordinates = num_coordinates
        self.use_attention = use_attention
        
        if use_attention:
            # Attention-based coordinate regression
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            # Learnable queries
            self.coord_queries = nn.Parameter(torch.randn(num_coordinates // 2, feature_size))
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Coordinate regressor
        self.coordinate_regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_coordinates),
            nn.Sigmoid()  # relative coords in [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: feature map [B, C, H, W]
        Returns:
            coordinates: predicted coords [B, num_coordinates] in [0,1]
        """
        batch_size = x.size(0)
        
        if self.use_attention:
            # Attention path
            # Flatten features: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
            
            # Expand queries
            queries = self.coord_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_points, C]
            
            # Attention
            attended_features, _ = self.attention(queries, x_flat, x_flat)  # [B, num_points, C]
            
            # Predict coordinates per query
            coordinates_list = []
            for i in range(attended_features.size(1)):
                point_feature = attended_features[:, i, :]  # [B, C]
                point_coords = self.coordinate_regressor(point_feature)  # [B, 2]
                coordinates_list.append(point_coords)
            
            coordinates = torch.cat(coordinates_list, dim=1)  # [B, num_coordinates]
        else:
            # Global average pooling path
            pooled = self.global_pool(x)
            flattened = pooled.view(batch_size, -1)
            coordinates = self.coordinate_regressor(flattened)
        
        return coordinates


class Swin_Multitask_Final(nn.Module):
    """
    Swin UNETR based multitask model with optional coordinate prediction.
    
    This model combines:
    1. Multiple segmentation tasks (like Swin_Multitask)
    2. Optional coordinate prediction (like SwinCoordinate)
    
    Based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        patch_size: int = 2,
        window_size: int = 7,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample="merging",
        use_v2=False,
        img_size: int = 640,  # Optional parameter for decoder kernel calculation
        # Coordinate prediction parameters
        enable_coordinate_prediction: bool = False,  # enable coordinate head
        num_coordinates: int = 2,  # number of coordinates to predict
        coordinate_hidden_size: int = 512,  # hidden size for coordinate head
        coordinate_dropout: float = 0.1,  # dropout for coordinate head
        use_attention_head: bool = False,  # whether to use attention in coord head
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels for each segmentation task.
            patch_size: patch size.
            window_size: local window size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            feature_size: dimension of network feature size.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling.
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
            img_size: Optional parameter for decoder kernel calculation.
            
            # Coordinate prediction specific parameters:
            enable_coordinate_prediction: Whether to enable coordinate prediction head.
            num_coordinates: Number of coordinates to predict (should be even for x,y pairs).
            coordinate_hidden_size: Hidden size for coordinate prediction head.
            coordinate_dropout: Dropout rate for coordinate prediction head.
            use_attention_head: Whether to use attention mechanism in coordinate head.

        Examples::

            # Basic multitask model without coordinate prediction
            >>> net = Swin_Multitask_Final(
            ...     in_channels=3, 
            ...     out_channels=(3, 3, 5), 
            ...     feature_size=48
            ... )

            # Multitask model with coordinate prediction
            >>> net = Swin_Multitask_Final(
            ...     in_channels=3, 
            ...     out_channels=(3, 3, 5), 
            ...     feature_size=48,
            ...     enable_coordinate_prediction=True,
            ...     num_coordinates=4,  # 2 points (x,y pairs)
            ...     use_attention_head=True
            ... )

        """

        super().__init__()

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

        # Coordinate prediction validation
        if enable_coordinate_prediction and num_coordinates % 2 != 0:
            raise ValueError("num_coordinates should be even (x,y pairs) when coordinate prediction is enabled")

        self.normalize = normalize
        self.tasks = len(out_channels)
        self.enable_coordinate_prediction = enable_coordinate_prediction
        self.num_coordinates = num_coordinates if enable_coordinate_prediction else 0
        
        # SwinTransformer backbone
        self.swinViT = SwinTransformer(
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
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        # Segmentation decoder modules for each task
        self.decoder_modules = nn.ModuleList()
        for i in range(self.tasks):
            self.decoder_modules.append(
                Decoder_Module(
                    in_channels=in_channels,
                    out_channels=out_channels[i] if out_channels[i] > 0 else 1,
                    patch_size=patch_size,
                    window_size=window_size,
                    feature_size=feature_size,
                    norm_name=norm_name,
                    spatial_dims=spatial_dims,
                    img_size=img_size,  # Pass img_size for kernel calculation
                )
            )

        # Coordinate prediction head (optional)
        if self.enable_coordinate_prediction:
            # Compute last feature dimension
            # For SwinViT: feature_size * (2 ** (len(depths) - 1)) * 2
            # Example: feature_size=128, depths=(2,2,18,2) -> 128 * 8 * 2 = 2048
            final_feature_dim = feature_size * (2 ** (len(depths) - 1)) * 2
            
            self.coordinate_head = CoordinateHead(
                feature_size=final_feature_dim,
                hidden_size=coordinate_hidden_size,
                num_coordinates=num_coordinates,
                dropout=coordinate_dropout,
                use_attention=use_attention_head
            )
        else:
            self.coordinate_head = None

    def load_from(self, weights):
        """Load pretrained weights from SwinUNETR checkpoint"""
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
        """
        Forward pass
        
        Args:
            x_in: Input tensor [B, C, H, W]
            
        Returns:
            If coordinate prediction is enabled:
                (segmentation_logits, coordinates)
                - segmentation_logits: List of segmentation outputs for each task
                - coordinates: Predicted coordinates [B, num_coordinates]
            If coordinate prediction is disabled:
                segmentation_logits: List of segmentation outputs for each task
        """
        # Get features from SwinTransformer backbone
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # Segmentation outputs for each task
        segmentation_logits = []
        for i in range(self.tasks):
            segmentation_logits.append(self.decoder_modules[i](x_in, hidden_states_out))

        # Coordinate prediction (optional)
        if self.enable_coordinate_prediction:
            # Use the deepest features for coordinate prediction
            deepest_features = hidden_states_out[-1]  # Last layer features [B, C, H, W]
            coordinates = self.coordinate_head(deepest_features)
            return segmentation_logits, coordinates
        else:
            return segmentation_logits

    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze SwinTransformer encoder parameters"""
        for param in self.swinViT.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze SwinTransformer encoder parameters"""
        for param in self.swinViT.parameters():
            param.requires_grad = True

    def freeze_coordinate_head(self):
        """Freeze coordinate prediction head parameters"""
        if self.coordinate_head is not None:
            for param in self.coordinate_head.parameters():
                param.requires_grad = False

    def unfreeze_coordinate_head(self):
        """Unfreeze coordinate prediction head parameters"""
        if self.coordinate_head is not None:
            for param in self.coordinate_head.parameters():
                param.requires_grad = True


class Decoder_Module(nn.Module):
    """Decoder module for segmentation tasks"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            patch_size: int = 2,
            window_size: int = 7,
            feature_size: int = 24,
            norm_name: tuple | str = "instance",
            spatial_dims: int = 2,
            img_size: int = 640,  # Add optional img_size for kernel calculation
    ):

        super().__init__()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Dynamic kernel size calculation based on img_size
        up_kernel_size = 2 if img_size // patch_size % 16 == window_size else 4
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=up_kernel_size,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in, hidden_states_out):
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        return logits


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def create_swin_multitask_final_model(config_name="default"):
    """
    Create a Swin_Multitask_Final model from predefined configs.
    
    Args:
        config_name: one of "default", "large", "tiny", "coordinate"
    
    Returns:
        Swin_Multitask_Final instance
    """
    configs = {
        "default": {
            "in_channels": 3,
            "out_channels": (3, 3, 5, 4, 2, 11, 5, 2),  # 8 segmentation heads
            "patch_size": 4,
            "window_size": 10,
            "feature_size": 48,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "enable_coordinate_prediction": False,
        },
        "large": {
            "in_channels": 3,
            "out_channels": (3, 3, 5, 4, 2, 11, 5, 2),
            "patch_size": 4,
            "window_size": 10,
            "feature_size": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "enable_coordinate_prediction": False,
        },
        "tiny": {
            "in_channels": 3,
            "out_channels": (3, 3, 5),
            "patch_size": 4,
            "window_size": 7,
            "feature_size": 24,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "enable_coordinate_prediction": False,
        },
        "coordinate": {
            "in_channels": 3,
            "out_channels": (3, 3, 5, 4, 2, 11, 5, 2),
            "patch_size": 4,
            "window_size": 10,
            "feature_size": 48,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "enable_coordinate_prediction": True,
            "num_coordinates": 4,  # 2 keypoints
            "coordinate_hidden_size": 512,
            "use_attention_head": True,
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    return Swin_Multitask_Final(**config)


if __name__ == "__main__":
    print("🔬 Swin_Multitask_Final model test")
    print("=" * 50)
    
    # Test different configs
    configs_to_test = ["tiny", "default", "coordinate"]
    
    for config_name in configs_to_test:
        print(f"\n📋 Testing config: {config_name}")
        try:
            model = create_swin_multitask_final_model(config_name)
            model.eval()
            
            # Test input
            x = torch.randn(2, 3, 640, 640)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(x)
            
            # Output info
            print(f"  ✅ Model created")
            print(f"  📥 Input shape: {x.shape}")
            print(f"  📊 Num parameters: {model.get_num_parameters():,}")
            print(f"  🎯 Num tasks: {model.tasks}")
            print(f"  📍 Coordinate head: {'enabled' if model.enable_coordinate_prediction else 'disabled'}")
            
            if model.enable_coordinate_prediction:
                segmentation_logits, coordinates = outputs
                print(f"  📤 Num segmentation outputs: {len(segmentation_logits)}")
                for i, logit in enumerate(segmentation_logits):
                    print(f"    Task{i}: {logit.shape}")
                print(f"  📤 Coordinate output shape: {coordinates.shape}")
                print(f"  📍 Coordinate range: [{coordinates.min().item():.3f}, {coordinates.max().item():.3f}]")
                
                # Inspect coordinates
                coords = coordinates[0].cpu().numpy()
                num_points = len(coords) // 2
                print(f"  🔍 Predicted {num_points} keypoints:")
                for i in range(num_points):
                    x_coord, y_coord = coords[i*2], coords[i*2+1]
                    print(f"    Point{i+1}: ({x_coord:.3f}, {y_coord:.3f})")
            else:
                segmentation_logits = outputs
                print(f"  📤 Num segmentation outputs: {len(segmentation_logits)}")
                for i, logit in enumerate(segmentation_logits):
                    print(f"    Task{i}: {logit.shape}")
                
        except Exception as e:
            print(f"  ❌ Config {config_name} test failed: {e}")
    
    print(f"\n📋 Usage:")
    print(f"1. Use create_swin_multitask_final_model() for predefined configs")
    print(f"2. Set enable_coordinate_prediction=True to enable coordinate head")
    print(f"3. num_coordinates must be even (x,y pairs)")
    print(f"4. Coordinate outputs are in [0,1]; multiply by image size for pixels")
    print(f"5. Supports freezing/unfreezing encoder and coordinate head")
    print(f"6. Supports multitask segmentation plus coordinate prediction")
