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
import colorsys
import imageio


class MultiTask_Module(Base_Module):
    def __init__(self, args):
        super(MultiTask_Module, self).__init__(args)

        self.model = init_model(self.args)
        self.init_weights(self.args.pretrained) 

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_dc = DiceLoss(mode='multiclass')

        self.val_overall_dsc = 0.0
        self.val_overall_jac = 0.0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input, target = batch

        output = self(input)    
        
        loss = 0
        ce_loss_total = 0
        dc_loss_total = 0
        for i in range(len(self.args.out_channels)):
            ce_loss = self.loss_ce(output[i], target[i])
            dc_loss = self.loss_dc(output[i], target[i])
            loss += self.args.class_weights[i] * (ce_loss + 1.5 * dc_loss)
            ce_loss_total += ce_loss
            dc_loss_total += dc_loss
        self.log(f"Train/CE_Loss", ce_loss_total / len(self.args.out_channels), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"Train/DC_Loss", dc_loss_total / len(self.args.out_channels), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Specific loss for private av separate task
        consistency_loss = torch.mean(torch.clamp(output[1] + output[2] - output[0], min=0))
        loss += 10 *consistency_loss
        self.log(f"Train/Consistency_Loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # For fusion loss, we need to handle this differently since we're comparing probabilities
        # Convert outputs to probabilities first
        output0_probs = torch.softmax(output[0], dim=1)
        output1_probs = torch.softmax(output[1], dim=1)
        output2_probs = torch.softmax(output[2], dim=1)
        
        # Get the vessel probability (class 1) for each output
        vessel_prob_0 = output0_probs[:, 1, :, :]  # Vessel probability from output[0]
        vessel_prob_1 = output1_probs[:, 1, :, :]  # Artery probability from output[1]
        vessel_prob_2 = output2_probs[:, 1, :, :]  # Vein probability from output[2]
        
        # Fusion loss: vessel should be the union of artery and vein
        # Use L1 loss to encourage vessel_prob_0 to be close to max(vessel_prob_1, vessel_prob_2)
        fusion_target = torch.maximum(vessel_prob_1, vessel_prob_2)
        fusion_loss = torch.mean(torch.abs(vessel_prob_0 - fusion_target))
        loss += 10 * fusion_loss
        self.log(f"Train/Fusion_Loss", fusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def on_validation_start(self):
        self.images_to_log = []
    
    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        outputs = self(images)

        # Handle target formats (single tensor vs list) similar to test_step
        active_tasks = [i for i in range(len(self.args.out_channels)) if self.args.out_channels[i] > 0]
        num_active_tasks = len(active_tasks)
        if num_active_tasks > 1 and not isinstance(targets, list):
            print(f"Detected single target tensor with {num_active_tasks} active tasks. Replicating target for all tasks.")
            targets = [targets for _ in range(len(self.args.out_channels))]

        # Initialize dictionaries to store task metrics and valid sample counts
        dsc_per_task = {}
        jac_per_task = {}
        valid_counts = {}
        for task_idx in range(len(self.args.out_channels)):
            dsc_per_task[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}
            jac_per_task[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}
            valid_counts[task_idx] = {class_idx: 0 for class_idx in range(1, self.args.out_channels[task_idx])}

        # Process each image in the batch
        for idx in range(images.size(0)):
            # Create normalized version of input image for visualization
            input_image = images[idx].cpu().detach()
            
            # Convert input image to NumPy array and denormalize for visualization
            input_img_np = input_image.permute(1, 2, 0).numpy()
            mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
            std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
            input_img_np = (input_img_np * std + mean)  # Denormalize
            input_img_np = (input_img_np * 255).astype(np.uint8)
            
            # Create combined prediction and target masks as integer arrays for class IDs
            combined_pred_mask = np.zeros((input_image.shape[1], input_image.shape[2]), dtype=np.uint8)
            combined_target_mask = np.zeros((input_image.shape[1], input_image.shape[2]), dtype=np.uint8)
            
            # Create a class mapping dictionary to define unique class IDs across all tasks
            class_offset = 1  # Start from 1 (0 is background)
            class_mapping = {0: "Background"}
            
            # Process each task
            for task_idx in range(len(self.args.out_channels)):
                # Skip tasks with out_channels = 0 (inactive tasks)
                if self.args.out_channels[task_idx] == 0:
                    continue
                    
                output_task = outputs[task_idx][idx]
                target_task = targets[task_idx][idx]
                
                # Get predicted mask and target mask
                pred_mask = torch.argmax(output_task, dim=0).cpu().detach().numpy().astype(np.uint8)
                target_mask = target_task.cpu().detach().numpy().astype(np.uint8)
                
                # Calculate metrics for each class and update the combined masks
                for class_idx in range(1, self.args.out_channels[task_idx]):
                    # Create binary masks for this class
                    output_binary = (pred_mask == class_idx).astype(np.uint8)
                    target_binary = (target_mask == class_idx).astype(np.uint8)
                    
                    # Only calculate metrics if target has labels for this class (following test logic)
                    if np.any(target_binary):
                        # Calculate DSC and JAC
                        dsc = dc(output_binary, target_binary)
                        jac = jc(output_binary, target_binary)
                        
                        # Add to task metrics
                        dsc_per_task[task_idx][class_idx] += dsc
                        jac_per_task[task_idx][class_idx] += jac
                        valid_counts[task_idx][class_idx] += 1
                    
                    # Create a unique global class ID for this task and class
                    global_class_idx = class_offset
                    class_mapping[global_class_idx] = f"Task{task_idx}_Class{class_idx}"
                    
                    # Add this class to combined masks with the global class index (regardless of label presence)
                    combined_pred_mask[output_binary > 0] = global_class_idx
                    combined_target_mask[target_binary > 0] = global_class_idx
                    
                    # Increment class offset for next class
                    class_offset += 1
            
            # Log images for visualization (only a few samples to avoid excessive logging)
            if batch_idx < 2 and idx < 4:  # Log up to 8 images
                # Create wandb image with input image and masks
                mask_img = wandb.Image(
                    input_img_np,  # Original input image
                    masks={
                        "predictions": {
                            "mask_data": combined_pred_mask,  # Integer mask with class IDs
                            "class_labels": class_mapping,    # Mapping from IDs to class names
                        },
                        "ground_truth": {
                            "mask_data": combined_target_mask,
                            "class_labels": class_mapping,
                        },
                    },
                    caption=f"Validation sample {batch_idx * self.args.batch_size + idx}"
                )
                self.images_to_log.append(mask_img)
        
        # Log average metrics for each task and class (only for classes with valid labels)
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
                # Note: Don't log 0.0 for missing classes in validation_step as it would interfere with epoch-level aggregation
        
        # Calculate and log overall average metrics (only from valid calculations in this batch)
        # Use on_step=False, on_epoch=True to let Lightning handle the aggregation across batches
        if valid_dscs:
            batch_avg_dsc = sum(valid_dscs) / len(valid_dscs)
            self.log("Val/Avg_Dsc", batch_avg_dsc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        if valid_jacs:
            batch_avg_jac = sum(valid_jacs) / len(valid_jacs)
            self.log("Val/Avg_Jac", batch_avg_jac, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_validation_end(self):
        if not self.args.debug:
            # Log images to wandb
            if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"Validation Images": self.images_to_log})

    # Note: Due to file size limits, I'm including the essential methods. 
    # The full test_step, on_test_end, and other methods should be extracted 
    # from the original file lines 1187-1709
    
    def on_test_start(self):
        self.names = []
        # Initialize dictionaries to store metrics for each task and class
        self.dscs = {}
        self.jacs = {}
        self.dscs_valued = {}
        self.jacs_valued = {}
        
        # Add dictionaries for overall direct metrics (organized by task)
        self.overall_dscs = {}
        self.overall_jacs = {}
        
        # Add dictionaries to map image names to their direct metrics (by task)
        self.name_to_overall_dsc = {}
        self.name_to_overall_jac = {}
        
        # Create dictionaries for each task
        for task_idx in range(len(self.args.out_channels)):
            # If task is inactive (out_channels=0), create empty dictionaries
            if self.args.out_channels[task_idx] == 0:
                self.dscs[task_idx] = {}
                self.jacs[task_idx] = {}
                self.dscs_valued[task_idx] = {}
                self.jacs_valued[task_idx] = {}
                self.overall_dscs[task_idx] = []
                self.overall_jacs[task_idx] = []
            else:
                self.dscs[task_idx] = {class_idx: [] for class_idx in range(1, self.args.out_channels[task_idx])}
                self.jacs[task_idx] = {class_idx: [] for class_idx in range(1, self.args.out_channels[task_idx])}
                self.dscs_valued[task_idx] = {class_idx: [] for class_idx in range(1, self.args.out_channels[task_idx])}
                self.jacs_valued[task_idx] = {class_idx: [] for class_idx in range(1, self.args.out_channels[task_idx])}
                self.overall_dscs[task_idx] = []
                self.overall_jacs[task_idx] = []
    
    def test_step(self, batch, batch_idx):
        images, targets, names = batch
        outputs = self(images)
        
        # Count active tasks
        active_tasks = [i for i in range(len(self.args.out_channels)) if self.args.out_channels[i] > 0]
        num_active_tasks = len(active_tasks)
        
        # Handle case where target is a single tensor but we have multiple active tasks
        # In this case, replicate the target for all active tasks
        if num_active_tasks > 1 and not isinstance(targets, list):
            print(f"Detected single target tensor with {num_active_tasks} active tasks. Replicating target for all tasks.")
            # Convert single target tensor to list of tensors for each active task
            targets = [targets for _ in range(len(self.args.out_channels))]
            is_single_task = False
            single_task_idx = None
        else:
            # Handle different formats based on number of active tasks
            # If only one task is active, targets is a tensor
            # If multiple tasks are active, targets is a list of tensors
            is_single_task = num_active_tasks == 1
            
            # Determine the index of the single active task if applicable
            single_task_idx = active_tasks[0] if is_single_task else None
        
        # Create base directory for results
        basename = "{}/results_{}_{}".format(self.args.save_name, self.args.dataset, self.args.seed)
        if not os.path.exists(basename):
            os.makedirs(basename, exist_ok=True)
            
        # Create overlap and mask directories
        overlap_dir = os.path.join(basename, "overlap")
        mask_dir = os.path.join(basename, "mask")
        if not os.path.exists(overlap_dir):
            os.makedirs(overlap_dir, exist_ok=True)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir, exist_ok=True)
            
        # Create sub-directories for each task in both overlap and mask directories
        overlap_task_dirs = []
        mask_task_dirs = []
        for i in range(len(self.args.out_channels)):
            # Overlap task directories
            overlap_task_dir = os.path.join(overlap_dir, f"task_{i}")
            if not os.path.exists(overlap_task_dir):
                os.makedirs(overlap_task_dir, exist_ok=True)
            overlap_task_dirs.append(overlap_task_dir)
            
            # Mask task directories  
            mask_task_dir = os.path.join(mask_dir, f"task_{i}")
            if not os.path.exists(mask_task_dir):
                os.makedirs(mask_task_dir, exist_ok=True)
            mask_task_dirs.append(mask_task_dir)
            
        # Create combined directory only in overlap folder
        overlap_combined_dir = os.path.join(overlap_dir, "combined")
        if not os.path.exists(overlap_combined_dir):
            os.makedirs(overlap_combined_dir, exist_ok=True)

        # Define fixed color palettes for specified tasks
        task_color_palettes = []
        
        # task0: red and green
        if len(self.args.out_channels) > 0:
            palette0 = [(0, 0, 0)]  # Background
            if self.args.out_channels[0] > 1:
                palette0.append((255, 0, 0))  # Red
            if self.args.out_channels[0] > 2:
                palette0.append((0, 255, 0))  # Green
            # Add more colors if needed
            for i in range(3, self.args.out_channels[0]):
                h = (i - 3) / max(1, self.args.out_channels[0] - 3)
                s, v = 0.8, 0.9
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                palette0.append((int(r * 255), int(g * 255), int(b * 255)))
            task_color_palettes.append(palette0)
            
        # task1: yellow and dark green
        if len(self.args.out_channels) > 1:
            palette1 = [(0, 0, 0)]  # Background
            if self.args.out_channels[1] > 1:
                palette1.append((255, 255, 0))  # Yellow
            if self.args.out_channels[1] > 2:
                palette1.append((0, 100, 0))  # Dark Green
            # Add more colors if needed
            for i in range(3, self.args.out_channels[1]):
                h = 0.3 + (i - 3) / max(1, self.args.out_channels[1] - 3) * 0.3
                s, v = 0.8, 0.9
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                palette1.append((int(r * 255), int(g * 255), int(b * 255)))
            task_color_palettes.append(palette1)
        
        # Add default colors for remaining tasks (task2, task3, etc.)
        for task_idx in range(2, len(self.args.out_channels)):
            palette = [(0, 0, 0)]  # Background
            hue_offset = (task_idx - 2) / max(1, len(self.args.out_channels) - 2) * 0.6 + 0.3
            for i in range(1, self.args.out_channels[task_idx]):
                h = (hue_offset + i / max(2, self.args.out_channels[task_idx] * 2)) % 1.0
                s, v = 0.8, 0.9
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                palette.append((int(r * 255), int(g * 255), int(b * 255)))
            task_color_palettes.append(palette)
            
        # Process each image in the batch
        for idx in range(images.size(0)):
            name = names[idx]
            
            # Create normalized version of input image for saving
            image_save = images[idx].cpu().detach().numpy()
            mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
            std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
            image_save = np.transpose(image_save, (1, 2, 0))
            image_save = (image_save * std + mean)  # Denormalize
            image_save = (image_save * 255).astype(np.uint8)
            
            # Create a blank canvas for combined segmentation visualization
            combined_output = image_save.copy()
            
            # Create integer masks for class IDs (for metric calculation)
            combined_pred_int = np.zeros((image_save.shape[0], image_save.shape[1]), dtype=np.uint8)
            combined_target_int = np.zeros((image_save.shape[0], image_save.shape[1]), dtype=np.uint8)
            
            # Create a class mapping dictionary
            class_offset = 1  # Start from 1 (0 is background)
            class_mapping = {0: "Background"}
            
            # Create a mask to track regions that have been segmented by any task
            all_segmented = np.zeros((image_save.shape[0], image_save.shape[1]), dtype=bool)
            
            # Define task processing order: task3 first, then task0, task1, task2
            # This controls the overlap order in the combined visualization
            task_order = []
            
            # Add task3 first if it exists and is active
            if 3 in active_tasks:
                task_order.append(3)
                active_tasks_temp = active_tasks.copy()
                active_tasks_temp.remove(3)
                
            # Add task0, task1, task2 if they are active
            for i in range(min(3, len(self.args.out_channels))):
                if i in active_tasks and i != 3:
                    task_order.append(i)
                    if 3 in active_tasks:
                        active_tasks_temp.remove(i)
                
            # Add any remaining tasks
            if 3 in active_tasks:
                task_order.extend(active_tasks_temp)
            
            # Process metrics for active tasks in original order
            for task_idx in range(len(self.args.out_channels)):
                # Skip tasks with out_channels = 0
                if self.args.out_channels[task_idx] == 0:
                    continue
                    
                output_task = outputs[task_idx][idx]
                
                # Handle different target formats
                if is_single_task:
                    # For single task, targets is a tensor directly
                    if task_idx == single_task_idx:
                        target_task = targets[idx]
                    else:
                        # This should not happen - we're only processing the active task
                        continue
                else:
                    # For multiple tasks, targets is a list of tensors
                    target_task = targets[task_idx][idx]
                
                # Get predicted mask as NumPy array
                pred_mask = torch.argmax(output_task, dim=0).cpu().detach().numpy().astype(np.uint8)
                target_mask = target_task.cpu().detach().numpy().astype(np.uint8)

                if np.any(target_mask):
                    overall_dsc = dc(pred_mask, target_mask)
                    overall_jac = jc(pred_mask, target_mask)
                    self.overall_dscs[task_idx].append(overall_dsc)
                    self.overall_jacs[task_idx].append(overall_jac)
                    
                    # Store direct metrics mapped to image name
                    if name not in self.name_to_overall_dsc:
                        self.name_to_overall_dsc[name] = {}
                    if name not in self.name_to_overall_jac:
                        self.name_to_overall_jac[name] = {}
                    
                    # Initialize task dictionary if needed
                    if task_idx not in self.name_to_overall_dsc[name]:
                        self.name_to_overall_dsc[name][task_idx] = []
                    if task_idx not in self.name_to_overall_jac[name]:
                        self.name_to_overall_jac[name][task_idx] = []
                        
                    self.name_to_overall_dsc[name][task_idx].append(overall_dsc)
                    self.name_to_overall_jac[name][task_idx].append(overall_jac)
                    
                    # Log direct overall metrics
                    self.log(f"Test/Direct_Overall_Dsc/Task_{task_idx}", overall_dsc, on_step=False, on_epoch=True, sync_dist=True)
                    self.log(f"Test/Direct_Overall_Jac/Task_{task_idx}", overall_jac, on_step=False, on_epoch=True, sync_dist=True)
                
                # Create task-specific segmentation for individual task images
                task_output = image_save.copy()
                task_colors = task_color_palettes[task_idx]
                
                for class_idx in range(1, self.args.out_channels[task_idx]):
                    # Apply color to segmentation mask for this class
                    mask = (pred_mask == class_idx)
                    if np.any(mask):
                        rgb_color = task_colors[class_idx]
                        task_output[mask] = rgb_color
                    
                    # Calculate metrics for this class
                    output_binary = mask.astype(np.uint8)
                    target_binary = (target_mask == class_idx).astype(np.uint8)
                    
                    # Only calculate and store metrics if target has this class
                    if np.any(target_binary):
                        dsc = dc(output_binary, target_binary)
                        jac = jc(output_binary, target_binary)
                        
                        self.dscs[task_idx][class_idx].append(dsc)
                        self.jacs[task_idx][class_idx].append(jac)
                        self.dscs_valued[task_idx][class_idx].append(dsc)
                        self.jacs_valued[task_idx][class_idx].append(jac)
                        
                        # Log metrics for this class
                        self.log(f"Test/Dice/Task_{task_idx}/Class_{class_idx}", dsc, on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"Test/Jac/Task_{task_idx}/Class_{class_idx}", jac, on_step=False, on_epoch=True, sync_dist=True)
                    else:
                        # If target doesn't have this class, add None to maintain consistency with names list
                        self.dscs[task_idx][class_idx].append(None)
                        self.jacs[task_idx][class_idx].append(None)
                
                # Save task-specific segmentation result if task is active
                if self.args.save_results:
                    # Save overlap result (colored visualization)
                    overlap_filename = os.path.join(overlap_task_dirs[task_idx], f"{name}.png")
                    overlap_dirname = os.path.dirname(overlap_filename)
                    if not os.path.exists(overlap_dirname):
                        os.makedirs(overlap_dirname, exist_ok=True)
                    imageio.imwrite(overlap_filename, task_output)
                    
                    # Save mask result (original segmentation mask)
                    mask_filename = os.path.join(mask_task_dirs[task_idx], f"{name}.png")
                    mask_dirname = os.path.dirname(mask_filename)
                    if not os.path.exists(mask_dirname):
                        os.makedirs(mask_dirname, exist_ok=True)
                    imageio.imwrite(mask_filename, pred_mask)
            
            # Skip combined visualization if only one task is active
            if num_active_tasks <= 1:
                # Store the name for later use in on_test_end
                self.names.append(name)
                continue
                
            # Now process tasks in the specified order for the combined visualization
            for task_idx in task_order:
                # Skip tasks with out_channels = 0
                if self.args.out_channels[task_idx] == 0:
                    continue
                    
                output_task = outputs[task_idx][idx]
                
                # Handle different target formats
                if is_single_task:
                    # For single task, targets is a tensor directly
                    if task_idx == single_task_idx:
                        target_task = targets[idx]
                    else:
                        # This should not happen in single task mode
                        continue
                else:
                    # For multiple tasks, targets is a list of tensors
                    target_task = targets[task_idx][idx]
                
                # Get predicted mask as NumPy array
                pred_mask = torch.argmax(output_task, dim=0).cpu().detach().numpy().astype(np.uint8)
                target_mask = target_task.cpu().detach().numpy().astype(np.uint8)
                
                # Get the color palette for this task
                task_colors = task_color_palettes[task_idx]
                
                # Process classes for this task
                for class_idx in range(1, self.args.out_channels[task_idx]):
                    # Apply color to segmentation mask for this class
                    mask = (pred_mask == class_idx)
                    if np.any(mask):
                        rgb_color = task_colors[class_idx]
                        # Add this class to the combined RGB output with task-specific color
                        combined_output[mask] = rgb_color
                    
                    # Create a unique global class ID for this task and class
                    global_class_idx = class_offset
                    class_mapping[global_class_idx] = f"Task{task_idx}_Class{class_idx}"
                    
                    # Add this class to integer masks with the global class ID
                    combined_pred_int[mask] = global_class_idx
                    target_mask_binary = (target_mask == class_idx)
                    combined_target_int[target_mask_binary] = global_class_idx
                    
                    # Increment class offset for next class
                    class_offset += 1
            
            # Save combined segmentation result if multiple tasks are active
            if self.args.save_results and num_active_tasks > 1:
                combined_filename = os.path.join(overlap_combined_dir, f"{name}.png")
                # Create directory if it doesn't exist
                combined_dirname = os.path.dirname(combined_filename)
                if not os.path.exists(combined_dirname):
                    os.makedirs(combined_dirname, exist_ok=True)
                
                # No boundaries, just colored segmentation
                imageio.imwrite(combined_filename, combined_output)
            
            # Store the name for later use in on_test_end
            self.names.append(name)

    def on_test_end(self):
        print('=> Writing results to local file')
        
        # Create a dataframe for Excel - include all values
        excel_data = {"name": self.names}
        
        # Add columns for DSC and JAC for each task and class
        for task_idx in range(len(self.args.out_channels)):
            # Skip inactive tasks (out_channels=0) in metrics
            if self.args.out_channels[task_idx] == 0:
                # For inactive tasks, add placeholder columns with zeros
                num_samples = len(self.names)
                # Assume at least class 1 would exist if the task was active
                excel_data[f"task_{task_idx}_class_1_dsc"] = [0.0] * num_samples
                excel_data[f"task_{task_idx}_class_1_jac"] = [0.0] * num_samples
                continue
                
            for class_idx in range(1, self.args.out_channels[task_idx]):
                # Filter out None values for Excel
                dscs_filtered = [dsc for dsc in self.dscs[task_idx][class_idx] if dsc is not None]
                jacs_filtered = [jac for jac in self.jacs[task_idx][class_idx] if jac is not None]
                
                # Create arrays of same length as names with None for missing values
                dsc_array = [None] * len(self.names)
                jac_array = [None] * len(self.names)
                
                # Fill in the non-None values
                valid_indices = [i for i, dsc in enumerate(self.dscs[task_idx][class_idx]) if dsc is not None]
                for i, valid_idx in enumerate(valid_indices):
                    dsc_array[valid_idx] = dscs_filtered[i]
                    jac_array[valid_idx] = jacs_filtered[i]
                
                excel_data[f"task_{task_idx}_class_{class_idx}_dsc"] = dsc_array
                excel_data[f"task_{task_idx}_class_{class_idx}_jac"] = jac_array
        
        # Add direct overall DSC metric to Excel for each task
        for task_idx in range(len(self.args.out_channels)):
            # Skip inactive tasks
            if self.args.out_channels[task_idx] == 0:
                continue
                
            # Add per-image direct overall DSC and JAC metrics for this task
            task_dsc_array = []
            task_jac_array = []
            
            # Process each image
            for name in self.names:
                if (name in self.name_to_overall_dsc and 
                    task_idx in self.name_to_overall_dsc[name] and 
                    self.name_to_overall_dsc[name][task_idx]):
                    # Average if multiple values exist for this image and task
                    avg_dsc = np.average(self.name_to_overall_dsc[name][task_idx])
                    task_dsc_array.append(avg_dsc)
                else:
                    # No direct DSC for this image and task
                    task_dsc_array.append(None)
                    
                if (name in self.name_to_overall_jac and 
                    task_idx in self.name_to_overall_jac[name] and 
                    self.name_to_overall_jac[name][task_idx]):
                    # Average if multiple values exist for this image and task
                    avg_jac = np.average(self.name_to_overall_jac[name][task_idx])
                    task_jac_array.append(avg_jac)
                else:
                    # No direct JAC for this image and task
                    task_jac_array.append(None)
                    
            # Add per-image arrays to excel data for this task
            excel_data[f"task_{task_idx}_direct_overall_dsc"] = task_dsc_array
            excel_data[f"task_{task_idx}_direct_overall_jac"] = task_jac_array
        
        # Create and save Excel file
        excel_df = pd.DataFrame(excel_data)
        excel_path = os.path.join(
            self.args.save_name,
            f"multitask_results_{self.args.dataset}_{self.args.fold}_{self.args.seed}.xlsx"
        )
        excel_df.to_excel(excel_path, index=False)
        print('=> Excel file saved')
        
        # Create JSON with averages (only considering non-zero values)
        json_results = {}
        
        # Calculate task and class-specific averages
        for task_idx in range(len(self.args.out_channels)):
            # For inactive tasks, set metrics to 0
            if self.args.out_channels[task_idx] == 0:
                # Add placeholder metrics with zeros
                json_results[f"task_{task_idx}_class_1_dsc"] = 0.0
                json_results[f"task_{task_idx}_class_1_jac"] = 0.0
                json_results[f"task_{task_idx}_avg_dsc"] = 0.0
                json_results[f"task_{task_idx}_avg_jac"] = 0.0
                continue
                
            task_avg_dscs = []
            task_avg_jacs = []
            
            for class_idx in range(1, self.args.out_channels[task_idx]):
                # Calculate and store class-level metrics if there are valued entries
                if self.dscs_valued[task_idx][class_idx]:
                    class_avg_dsc = np.average(self.dscs_valued[task_idx][class_idx])
                    json_results[f"task_{task_idx}_class_{class_idx}_dsc"] = class_avg_dsc
                    task_avg_dscs.append(class_avg_dsc)
                else:
                    # No valued metrics for this class, set to 0
                    json_results[f"task_{task_idx}_class_{class_idx}_dsc"] = 0.0
                    
                if self.jacs_valued[task_idx][class_idx]:
                    class_avg_jac = np.average(self.jacs_valued[task_idx][class_idx])
                    json_results[f"task_{task_idx}_class_{class_idx}_jac"] = class_avg_jac
                    task_avg_jacs.append(class_avg_jac)
                else:
                    # No valued metrics for this class, set to 0
                    json_results[f"task_{task_idx}_class_{class_idx}_jac"] = 0.0
            
            # Calculate task-level averages
            if task_avg_dscs:
                json_results[f"task_{task_idx}_avg_dsc"] = np.average(task_avg_dscs)
            else:
                json_results[f"task_{task_idx}_avg_dsc"] = 0.0
                
            if task_avg_jacs:
                json_results[f"task_{task_idx}_avg_jac"] = np.average(task_avg_jacs)
            else:
                json_results[f"task_{task_idx}_avg_jac"] = 0.0
        
        # Calculate overall metrics (only from active tasks)
        active_task_dscs = [json_results[f"task_{i}_avg_dsc"] for i in range(len(self.args.out_channels)) 
                           if self.args.out_channels[i] > 0 and f"task_{i}_avg_dsc" in json_results]
        active_task_jacs = [json_results[f"task_{i}_avg_jac"] for i in range(len(self.args.out_channels)) 
                           if self.args.out_channels[i] > 0 and f"task_{i}_avg_jac" in json_results]
        
        if active_task_dscs:
            json_results["overall_avg_dsc"] = np.average(active_task_dscs)
        else:
            json_results["overall_avg_dsc"] = 0.0
            
        if active_task_jacs:
            json_results["overall_avg_jac"] = np.average(active_task_jacs)
        else:
            json_results["overall_avg_jac"] = 0.0
        
        # Add direct overall metrics (calculated directly from masks)
        for task_idx in range(len(self.args.out_channels)):
            # Skip inactive tasks
            if self.args.out_channels[task_idx] == 0:
                continue
                
            if self.overall_dscs[task_idx]:
                json_results[f"task_{task_idx}_dsc_overall"] = np.average(self.overall_dscs[task_idx])
            else:
                json_results[f"task_{task_idx}_dsc_overall"] = 0.0
                
            if self.overall_jacs[task_idx]:
                json_results[f"task_{task_idx}_jac_overall"] = np.average(self.overall_jacs[task_idx])
            else:
                json_results[f"task_{task_idx}_jac_overall"] = 0.0
        
        # Calculate overall direct metrics (average across all active tasks)
        all_task_overall_dscs = []
        all_task_overall_jacs = []
        
        for task_idx in range(len(self.args.out_channels)):
            if self.args.out_channels[task_idx] > 0:
                if self.overall_dscs[task_idx]:
                    all_task_overall_dscs.append(np.average(self.overall_dscs[task_idx]))
                if self.overall_jacs[task_idx]:
                    all_task_overall_jacs.append(np.average(self.overall_jacs[task_idx]))
        
        if all_task_overall_dscs:
            json_results["dsc_overall"] = np.average(all_task_overall_dscs)
        else:
            json_results["dsc_overall"] = 0.0
            
        if all_task_overall_jacs:
            json_results["jac_overall"] = np.average(all_task_overall_jacs)
        else:
            json_results["jac_overall"] = 0.0
        
        # Save JSON file
        json_path = os.path.join(
            self.args.save_name,
            f"multitask_results_{self.args.dataset}_{self.args.fold}_{self.args.seed}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print('=> JSON results saved')
