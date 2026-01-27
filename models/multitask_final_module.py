import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from medpy.metric import dc, jc
from segmentation_models_pytorch.losses import DiceLoss
from models.base_module import Base_Module
from models.model_factory import init_model
from models.swin_coordinate import CoordinateLoss
import colorsys
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class MultiTask_Final_Module(Base_Module):
    """
    Lightning Module for Swin_Multitask_Final model.
    
    This module combines:
    1. Multiple segmentation tasks (like MultiTask_Module)
    2. Optional coordinate prediction (like Coordinate_Module)
    
    Key features:
    - Supports both segmentation and coordinate prediction
    - Flexible loss weighting between tasks
    - Comprehensive validation and testing metrics
    - Visualization support for both outputs
    """
    
    def __init__(self, args):
        super(MultiTask_Final_Module, self).__init__(args)

        self.model = init_model(self.args)
        self.init_weights(self.args.pretrained)

        # Segmentation losses
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_dc = DiceLoss(mode='multiclass')

        # Coordinate prediction losses (if enabled)
        if getattr(args, 'enable_coordinate_prediction', False):
            self.loss_smooth_l1 = nn.SmoothL1Loss()
            self.loss_coordinate = CoordinateLoss(loss_type='smooth_l1')
            self.coordinate_loss_weight = getattr(args, 'coordinate_loss_weight', 1.0)
        else:
            self.loss_smooth_l1 = None
            self.loss_coordinate = None
            self.coordinate_loss_weight = 0.0

        # Loss weights
        self.segmentation_loss_weight = getattr(args, 'segmentation_loss_weight', 1.0)
        self.consistency_loss_weight = getattr(args, 'consistency_loss_weight', 10.0)
        self.fusion_loss_weight = getattr(args, 'fusion_loss_weight', 10.0)

        # Validation metrics
        self.val_overall_dsc = 0.0
        self.val_overall_jac = 0.0
        self.val_coordinate_error = 0.0

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 2:
            # Only segmentation data
            input, target = batch
            target_coords = None
        elif len(batch) == 3:
            # Segmentation + coordinate data
            input, target, target_coords = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Forward pass
        outputs = self(input)
        
        # Handle different output formats
        if getattr(self.args, 'enable_coordinate_prediction', False):
            segmentation_outputs, pred_coords = outputs
        else:
            segmentation_outputs = outputs
            pred_coords = None

        # Calculate segmentation losses
        seg_loss = 0
        ce_loss_total = 0
        dc_loss_total = 0
        
        for i in range(len(self.args.out_channels)):
            if self.args.out_channels[i] > 0:  # Skip inactive tasks
                ce_loss = self.loss_ce(segmentation_outputs[i], target[i])
                dc_loss = self.loss_dc(segmentation_outputs[i], target[i])
                seg_loss += self.args.class_weights[i] * (ce_loss + 1.5 * dc_loss)
                ce_loss_total += ce_loss
                dc_loss_total += dc_loss

        # Specific losses for private av separate task
        if len(segmentation_outputs) >= 3:
            consistency_loss = torch.mean(torch.clamp(segmentation_outputs[1] + segmentation_outputs[2] - segmentation_outputs[0], min=0))
            seg_loss += self.consistency_loss_weight * consistency_loss
            self.log(f"Train/Consistency_Loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            # Fusion loss
            output0_probs = torch.softmax(segmentation_outputs[0], dim=1)
            output1_probs = torch.softmax(segmentation_outputs[1], dim=1)
            output2_probs = torch.softmax(segmentation_outputs[2], dim=1)
            
            vessel_prob_0 = output0_probs[:, 1, :, :]
            vessel_prob_1 = output1_probs[:, 1, :, :]
            vessel_prob_2 = output2_probs[:, 1, :, :]
            
            fusion_target = torch.maximum(vessel_prob_1, vessel_prob_2)
            fusion_loss = torch.mean(torch.abs(vessel_prob_0 - fusion_target))
            seg_loss += self.fusion_loss_weight * fusion_loss
            self.log(f"Train/Fusion_Loss", fusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Calculate coordinate losses (if enabled)
        coord_loss = 0
        if pred_coords is not None and target_coords is not None:
            smooth_l1_loss = self.loss_smooth_l1(pred_coords, target_coords)
            coordinate_loss = self.loss_coordinate(pred_coords, target_coords)
            coord_loss = smooth_l1_loss + 0.5 * coordinate_loss
            
            # Calculate distance error for monitoring
            with torch.no_grad():
                num_points = pred_coords.size(1) // 2
                distance_errors = []
                for i in range(num_points):
                    pred_x, pred_y = pred_coords[:, i*2], pred_coords[:, i*2+1]
                    target_x, target_y = target_coords[:, i*2], target_coords[:, i*2+1]
                    distance_error = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
                    distance_errors.append(distance_error.mean())
                avg_distance_error = torch.stack(distance_errors).mean()

            self.log(f"Train/Coordinate_SmoothL1_Loss", smooth_l1_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"Train/Coordinate_Loss", coordinate_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"Train/Avg_Distance_Error", avg_distance_error, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Total loss
        total_loss = self.segmentation_loss_weight * seg_loss + self.coordinate_loss_weight * coord_loss

        # Log segmentation losses
        self.log(f"Train/CE_Loss", ce_loss_total / len(self.args.out_channels), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"Train/DC_Loss", dc_loss_total / len(self.args.out_channels), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"Train/Segmentation_Loss", seg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"Train/Total_Loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss
    
    def on_validation_start(self):
        self.images_to_log = []
        self.coordinate_predictions = []
    
    def validation_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 3:
            # Only segmentation data
            images, targets, _ = batch
            target_coords = None
        elif len(batch) == 4:
            # Segmentation + coordinate data
            images, targets, target_coords, _ = batch
        else:
            raise ValueError(f"Unexpected validation batch format with {len(batch)} elements")

        outputs = self(images)
        
        # Handle different output formats
        if getattr(self.args, 'enable_coordinate_prediction', False):
            segmentation_outputs, pred_coords = outputs
        else:
            segmentation_outputs = outputs
            pred_coords = None

        # Segmentation validation metrics (same as MultiTask_Module)
        dsc_per_task = {}
        jac_per_task = {}
        valid_counts = {}
        for task_idx in range(len(self.args.out_channels)):
            dsc_per_task[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}
            jac_per_task[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}
            valid_counts[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}

        # Process each image in the batch
        for idx in range(images.size(0)):
            input_image = images[idx].cpu().detach()
            
            # Convert input image to NumPy array and denormalize for visualization
            input_img_np = input_image.permute(1, 2, 0).numpy()
            mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
            std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
            input_img_np = (input_img_np * std + mean)
            input_img_np = (input_img_np * 255).astype(np.uint8)
            
            # Create combined prediction and target masks
            combined_pred_mask = np.zeros((input_image.shape[1], input_image.shape[2]), dtype=np.uint8)
            combined_target_mask = np.zeros((input_image.shape[1], input_image.shape[2]), dtype=np.uint8)
            
            class_offset = 1
            class_mapping = {0: "Background"}
            
            # Process each task
            for task_idx in range(len(self.args.out_channels)):
                if self.args.out_channels[task_idx] == 0:
                    continue
                    
                output_task = segmentation_outputs[task_idx][idx]
                target_task = targets[task_idx][idx]
                
                pred_mask = torch.argmax(output_task, dim=0).cpu().detach().numpy().astype(np.uint8)
                target_mask = target_task.cpu().detach().numpy().astype(np.uint8)
                
                # Calculate metrics for each class
                for class_idx in range(1, self.args.out_channels[task_idx]):
                    output_binary = (pred_mask == class_idx).astype(np.uint8)
                    target_binary = (target_mask == class_idx).astype(np.uint8)
                    
                    if np.any(target_binary):
                        dsc = dc(output_binary, target_binary)
                        jac = jc(output_binary, target_binary)
                        
                        dsc_per_task[task_idx][class_idx] += dsc
                        jac_per_task[task_idx][class_idx] += jac
                        valid_counts[task_idx][class_idx] += 1
                    
                    global_class_idx = class_offset
                    class_mapping[global_class_idx] = f"Task{task_idx}_Class{class_idx}"
                    
                    combined_pred_mask[output_binary > 0] = global_class_idx
                    combined_target_mask[target_binary > 0] = global_class_idx
                    
                    class_offset += 1
            
            # Log images for visualization
            if batch_idx < 2 and idx < 4:
                mask_img = wandb.Image(
                    input_img_np,
                    masks={
                        "predictions": {
                            "mask_data": combined_pred_mask,
                            "class_labels": class_mapping,
                        },
                        "ground_truth": {
                            "mask_data": combined_target_mask,
                            "class_labels": class_mapping,
                        },
                    },
                    caption=f"Validation sample {batch_idx * self.args.batch_size + idx}"
                )
                self.images_to_log.append(mask_img)

        # Log segmentation metrics
        valid_dscs = []
        valid_jacs = []
        for task_idx in range(len(self.args.out_channels)):
            if self.args.out_channels[task_idx] == 0:
                continue
                
            for class_idx in range(1, self.args.out_channels[task_idx]):
                if valid_counts[task_idx][class_idx] > 0:
                    avg_dsc = dsc_per_task[task_idx][class_idx] / valid_counts[task_idx][class_idx]
                    avg_jac = jac_per_task[task_idx][class_idx] / valid_counts[task_idx][class_idx]
                    
                    self.log(f"Val/Dsc/Task_{task_idx}/Class_{class_idx}", avg_dsc, on_epoch=True, sync_dist=True)
                    self.log(f"Val/Jac/Task_{task_idx}/Class_{class_idx}", avg_jac, on_epoch=True, sync_dist=True)
                    
                    valid_dscs.append(avg_dsc)
                    valid_jacs.append(avg_jac)

        if valid_dscs:
            batch_avg_dsc = sum(valid_dscs) / len(valid_dscs)
            self.log("Val/Avg_Dsc", batch_avg_dsc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        if valid_jacs:
            batch_avg_jac = sum(valid_jacs) / len(valid_jacs)
            self.log("Val/Avg_Jac", batch_avg_jac, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # Coordinate validation metrics (if enabled)
        if pred_coords is not None and target_coords is not None:
            batch_size = pred_coords.size(0)
            num_points = pred_coords.size(1) // 2
            
            total_distance_error = 0
            point_distance_errors = []
            
            for i in range(num_points):
                pred_x, pred_y = pred_coords[:, i*2], pred_coords[:, i*2+1]
                target_x, target_y = target_coords[:, i*2], target_coords[:, i*2+1]
                
                distance_error = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
                point_avg_error = distance_error.mean()
                point_distance_errors.append(point_avg_error)
                total_distance_error += point_avg_error

            avg_distance_error = total_distance_error / num_points

            self.log("Val/Avg_Distance_Error", avg_distance_error, on_epoch=True, sync_dist=True, prog_bar=True)
            
            for i, point_error in enumerate(point_distance_errors):
                self.log(f"Val/Point_{i+1}_Distance_Error", point_error, on_epoch=True, sync_dist=True)

            # Store coordinate predictions for visualization
            self.coordinate_predictions.append({
                'pred_coords': pred_coords.cpu().numpy(),
                'target_coords': target_coords.cpu().numpy(),
                'distance_errors': [error.cpu().item() for error in point_distance_errors]
            })

    def on_validation_end(self):
        if not self.args.debug:
            # Log segmentation images to wandb
            if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"Validation Images": self.images_to_log})
                
                # Log coordinate predictions if available
                if self.coordinate_predictions:
                    try:
                        num_samples_to_plot = min(4, len(self.coordinate_predictions))
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        axes = axes.flatten()
                        
                        for i in range(num_samples_to_plot):
                            if i < len(self.coordinate_predictions):
                                pred = self.coordinate_predictions[i]['pred_coords'][0]
                                target = self.coordinate_predictions[i]['target_coords'][0]
                                
                                ax = axes[i]
                                
                                num_points = len(pred) // 2
                                for j in range(num_points):
                                    pred_x, pred_y = pred[j*2], pred[j*2+1]
                                    target_x, target_y = target[j*2], target[j*2+1]
                                    
                                    ax.scatter(pred_x, pred_y, c='red', marker='o', s=100, label=f'Pred Point {j+1}' if j == 0 else "")
                                    ax.scatter(target_x, target_y, c='blue', marker='x', s=100, label=f'Target Point {j+1}' if j == 0 else "")
                                    ax.plot([pred_x, target_x], [pred_y, target_y], 'k--', alpha=0.5)
                                
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.set_title(f'Sample {i+1}')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        self.logger.experiment.log({"Validation_Coordinate_Predictions": wandb.Image(fig)}, commit=True)
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"Warning: Could not create coordinate visualization: {e}")

    def configure_optimizers(self):
        # Different learning rates for different components
        if getattr(self.args, 'enable_coordinate_prediction', False):
            # Separate learning rates for segmentation and coordinate prediction
            seg_lr = getattr(self.args, 'segmentation_lr', self.args.lr)
            coord_lr = getattr(self.args, 'coordinate_lr', self.args.lr)
            encoder_lr = getattr(self.args, 'encoder_lr', self.args.lr * 0.1)
            
            param_groups = [
                {'params': self.model.swinViT.parameters(), 'lr': encoder_lr},
                {'params': [p for module in self.model.decoder_modules for p in module.parameters()], 'lr': seg_lr},
                {'params': self.model.coordinate_head.parameters(), 'lr': coord_lr}
            ]
            opt = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        else:
            # Standard optimization for segmentation only
            opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epoch)
        return [opt], [scheduler]
