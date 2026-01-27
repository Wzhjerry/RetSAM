import cv2
import numpy as np
import os
import random
import glob
import argparse


# Predefined color palette for up to 20 classes (BGR format)
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
    (0, 255, 0),      # Green
]

# Specific class IDs to visualize - only these will be shown
CLASSES_TO_VISUALIZE = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]


def visualize_mask(image_path, mask_path, output_path, alpha=0.5, contour_only=True, thickness=2, classes_to_visualize=CLASSES_TO_VISUALIZE, save_individual_classes=False):
    """
    Visualize mask by overlaying different colors for each class in the mask on the original image.
    
    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image (where different pixel values represent different classes)
        output_path (str): Path to save the visualization result
        alpha (float): Transparency of the mask overlay (0-1)
        contour_only (bool): Whether to only show contours instead of filled regions
        thickness (int): Thickness of contours when contour_only is True
        classes_to_visualize (list): List of class IDs to visualize, others will be ignored
        save_individual_classes (bool): Whether to save individual class visualizations to separate folders
    
    Returns:
        tuple: (combined_path, individual_paths) where combined_path is the main output file and individual_paths is a list of individual class files
    """
    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    if mask is None:
        raise ValueError(f"Failed to load mask from {mask_path}")
    
    # Find unique class IDs in the mask
    unique_classes = np.unique(mask)
    print("unique_classes", unique_classes)
    
    # Check if mask contains any of the classes we want to visualize
    has_target_classes = any(class_id in classes_to_visualize for class_id in unique_classes if class_id != 0)
    
    if not has_target_classes:
        print(f"Mask {mask_path} does not contain any target classes {classes_to_visualize}, skipping")
        return (None, [])
    
    # Create a copy of the original image for overlay
    result = image.copy()
    individual_outputs = []
    
    # Get base name for individual class files
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    output_dir = os.path.dirname(output_path)
    
    if contour_only:
        # Draw only contours for each class
        for class_id in unique_classes:
            if class_id == 0 or (classes_to_visualize and class_id not in classes_to_visualize):
                continue
                
            # Use green color for all classes
            color = (0, 255, 0)  # Green color in BGR format
            
            # Create a binary mask for this class
            binary_mask = np.zeros_like(mask)
            binary_mask[mask == class_id] = 255
            
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the contours on the result image
            cv2.drawContours(result, contours, -1, color, thickness)
            
            # Save individual class if requested
            if save_individual_classes:
                individual_result = image.copy()
                cv2.drawContours(individual_result, contours, -1, color, thickness)
                # Create class-specific folder
                class_output_dir = os.path.join(output_dir, f"class_{class_id}")
                os.makedirs(class_output_dir, exist_ok=True)
                individual_output_path = os.path.join(class_output_dir, f"{base_name}.png")
                cv2.imwrite(individual_output_path, individual_result)
                individual_outputs.append(individual_output_path)
    else:
        # Create a color visualization
        colored_mask = np.zeros_like(image)
        
        # Use fixed colors for each class
        for class_id in unique_classes:
            if class_id == 0 or (classes_to_visualize and class_id not in classes_to_visualize):
                continue
            
            # Use green color for all classes
            color = (0, 255, 0)  # Green color in BGR format
            colored_mask[mask == class_id] = color
            
            # Save individual class if requested
            if save_individual_classes:
                individual_colored_mask = np.zeros_like(image)
                individual_colored_mask[mask == class_id] = color
                individual_result = image.copy()
                cv2.addWeighted(individual_colored_mask, alpha, individual_result, 1 - alpha, 0, individual_result)
                # Create class-specific folder
                class_output_dir = os.path.join(output_dir, f"class_{class_id}")
                os.makedirs(class_output_dir, exist_ok=True)
                individual_output_path = os.path.join(class_output_dir, f"{base_name}.png")
                cv2.imwrite(individual_output_path, individual_result)
                individual_outputs.append(individual_output_path)
        
        # Overlay the colored mask on the original image
        cv2.addWeighted(colored_mask, alpha, result, 1 - alpha, 0, result)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined result
    cv2.imwrite(output_path, result)
    
    return (output_path, individual_outputs)


def visualize_multiple_masks(image_path, mask_paths, output_path, alpha=0.5, contour_only=True, thickness=2, classes_to_visualize=CLASSES_TO_VISUALIZE):
    """
    Visualize multiple masks on the same image with different colors.
    
    Args:
        image_path (str): Path to the original image
        mask_paths (list): List of paths to mask images
        output_path (str): Path to save the visualization result
        alpha (float): Transparency of the mask overlay (0-1)
        contour_only (bool): Whether to only show contours instead of filled regions
        thickness (int): Thickness of contours when contour_only is True
        classes_to_visualize (list): List of class IDs to visualize, others will be ignored
    
    Returns:
        str or None: Path to saved file if any mask contains classes to visualize, None otherwise
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Check if any mask contains target classes
    has_any_target_classes = False
    
    # Create a copy of the original image for overlay
    result = image.copy()
    
    # For each mask, overlay with a different color
    for i, mask_path in enumerate(mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Failed to load mask from {mask_path}, skipping")
            continue
        
        # Find unique class IDs in the mask
        unique_classes = np.unique(mask)
        
        # Check if this mask contains any target classes
        mask_has_target_classes = any(class_id in classes_to_visualize for class_id in unique_classes if class_id != 0)
        if mask_has_target_classes:
            has_any_target_classes = True
        
        if contour_only:
            # Draw only contours for each class
            for class_id in unique_classes:
                if class_id == 0 or (classes_to_visualize and class_id not in classes_to_visualize):
                    continue
                    
                # Use green color for all classes
                color = (0, 255, 0)  # Green color in BGR format
                
                # Create a binary mask for this class
                binary_mask = np.zeros_like(mask)
                binary_mask[mask == class_id] = 255
                
                # Find contours in the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the contours on the result image
                cv2.drawContours(result, contours, -1, color, thickness)
        else:
            # Create a color visualization for this mask
            colored_mask = np.zeros_like(image)
            
            # Use fixed colors for each class
            for class_id in unique_classes:
                if class_id == 0 or (classes_to_visualize and class_id not in classes_to_visualize):
                    continue
                
                # Use green color for all classes
                color = (0, 255, 0)  # Green color in BGR format
                colored_mask[mask == class_id] = color
            
            # Overlay this mask on the result
            cv2.addWeighted(colored_mask, alpha, result, 1, 0, result)
    
    # Only save if at least one mask contains target classes
    if not has_any_target_classes:
        print(f"No masks contain target classes {classes_to_visualize}, skipping")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the result
    cv2.imwrite(output_path, result)
    
    return output_path


def process_directory(images_dir, masks_dir, output_dir, contour_only=True, thickness=2, alpha=0.5, classes_to_visualize=CLASSES_TO_VISUALIZE, save_individual_classes=False):
    """
    Process all images in a directory, overlaying corresponding masks.
    
    Args:
        images_dir (str): Directory containing original images
        masks_dir (str): Directory containing mask images
        output_dir (str): Directory to save visualization results
        contour_only (bool): Whether to only show contours instead of filled regions
        thickness (int): Thickness of contours when contour_only is True
        alpha (float): Transparency of the mask overlay (0-1)
        classes_to_visualize (list): List of class IDs to visualize, others will be ignored
        save_individual_classes (bool): Whether to save individual class visualizations to separate folders
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for image_path in image_files:
        # Get the base filename without extension
        image_basename = os.path.basename(image_path)
        filename, _ = os.path.splitext(image_basename)
        
        # Look for matching mask file (with any extension)
        mask_pattern = os.path.join(masks_dir, f"{filename}.*")
        mask_files = glob.glob(mask_pattern)
        
        if not mask_files:
            print(f"No matching mask found for {image_basename}, skipping")
            skipped_count += 1
            continue
        
        # Use the first matching mask file
        mask_path = mask_files[0]
        
        # Create output path
        output_path = os.path.join(output_dir, f"{filename}.png")
        
        try:
            # Process the image-mask pair
            result = visualize_mask(
                image_path, 
                mask_path, 
                output_path, 
                alpha=alpha,
                contour_only=contour_only, 
                thickness=thickness,
                classes_to_visualize=classes_to_visualize,
                save_individual_classes=save_individual_classes
            )
            
            if result[0] is not None:  # result is now a tuple (combined_path, individual_paths)
                processed_count += 1
                combined_path, individual_paths = result
                print(f"Processed {image_basename} -> {os.path.basename(combined_path)}")
                
                # Show individual class files if any were created
                if save_individual_classes and individual_paths:
                    print(f"  📂 Saved individual class images ({len(individual_paths)} files):")
                    for path in individual_paths:
                        relative_path = os.path.relpath(path, output_dir)
                        print(f"     - {relative_path}")
            else:
                skipped_count += 1
                print(f"Skipped {image_basename} (no target classes found)")
        except Exception as e:
            print(f"Error processing {image_basename}: {str(e)}")
            skipped_count += 1
    
    print(f"Completed: {processed_count} images processed, {skipped_count} skipped")


def visualize_single_image_by_name(image_name, images_dir, masks_dir, output_dir, contour_only=True, thickness=2, alpha=0.5, show_all_classes=True, save_individual_classes=False):
    """
    Visualize a single image by name; optionally show all or selected classes.
    
    Args:
        image_name (str): image filename (with or without extension)
        images_dir (str): Directory containing original images
        masks_dir (str): Directory containing mask images
        output_dir (str): Directory to save visualization results
        contour_only (bool): Whether to only show contours instead of filled regions
        thickness (int): Thickness of contours when contour_only is True
        alpha (float): Transparency of the mask overlay (0-1)
        show_all_classes (bool): True to show all classes; False to show only CLASSES_TO_VISUALIZE
        save_individual_classes (bool): True to save per-class images
    
    Returns:
        str or None: Path to saved file if successful, None otherwise
    """
    print(f"🔍 Searching image: {image_name}")
    
    # Normalize basename (strip extension)
    base_name = os.path.splitext(image_name)[0]
    
    # Find matching image file
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_path = None
    
    for ext in image_extensions:
        # Try different extensions
        for ext_variant in [ext.lower(), ext.upper()]:
            pattern = os.path.join(images_dir, f"{base_name}.{ext_variant[2:]}")
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
                break
        if image_path:
            break
    
    if not image_path:
        print(f"❌ Image not found: {image_name} in {images_dir}")
        return None
    
    print(f"✅ Found image: {os.path.basename(image_path)}")
    
    # Find corresponding mask
    mask_pattern = os.path.join(masks_dir, f"{base_name}.*")
    mask_files = glob.glob(mask_pattern)
    
    if not mask_files:
        print(f"❌ Mask not found: {base_name}.* in {masks_dir}")
        return None
    
    mask_path = mask_files[0]
    print(f"✅ Found mask: {os.path.basename(mask_path)}")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_all_classes.png")
    
    try:
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        if mask is None:
            raise ValueError(f"Failed to load mask from {mask_path}")
        
        # Unique class ids in mask
        unique_classes = np.unique(mask)
        print("🎨 Classes in mask:", unique_classes)
        
        if show_all_classes:
            classes_to_show = [class_id for class_id in unique_classes if class_id != 0]
            print(f"📊 Showing all classes: {classes_to_show}")
        else:
            classes_to_show = [class_id for class_id in unique_classes 
                             if class_id != 0 and class_id in CLASSES_TO_VISUALIZE]
            print(f"📊 Showing selected classes: {classes_to_show}")
        
        if not classes_to_show:
            print("❌ No classes to display")
            return None
        
        # Create result image
        result = image.copy()
        individual_outputs = []
        
        if contour_only:
            # Contour mode
            for i, class_id in enumerate(classes_to_show):
                color = (0, 255, 0)  # Green color in BGR format
                
                # Binary mask
                binary_mask = np.zeros_like(mask)
                binary_mask[mask == class_id] = 255
                
                # Find and draw contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, color, thickness)
                
                print(f"  ✓ Class {class_id}: {len(contours)} contours, color {color}")
                
                # Save per-class image if requested
                if save_individual_classes:
                    individual_result = image.copy()
                    cv2.drawContours(individual_result, contours, -1, color, thickness)
                    # Per-class folder
                    class_output_dir = os.path.join(output_dir, f"class_{class_id}")
                    os.makedirs(class_output_dir, exist_ok=True)
                    individual_output_path = os.path.join(class_output_dir, f"{base_name}.png")
                    cv2.imwrite(individual_output_path, individual_result)
                    individual_outputs.append(individual_output_path)
                    print(f"    💾 Saved: class_{class_id}/{os.path.basename(individual_output_path)}")
        else:
            # Filled color mode
            colored_mask = np.zeros_like(image)
            
            for i, class_id in enumerate(classes_to_show):
                color = (0, 255, 0)  # Green color in BGR format
                colored_mask[mask == class_id] = color
                
                # Pixel count
                pixel_count = np.sum(mask == class_id)
                print(f"  ✓ Class {class_id}: {pixel_count} pixels, color {color}")
                
                # Save per-class image if requested
                if save_individual_classes:
                    individual_colored_mask = np.zeros_like(image)
                    individual_colored_mask[mask == class_id] = color
                    individual_result = image.copy()
                    cv2.addWeighted(individual_colored_mask, alpha, individual_result, 1 - alpha, 0, individual_result)
                    # Per-class folder
                    class_output_dir = os.path.join(output_dir, f"class_{class_id}")
                    os.makedirs(class_output_dir, exist_ok=True)
                    individual_output_path = os.path.join(class_output_dir, f"{base_name}.png")
                    cv2.imwrite(individual_output_path, individual_result)
                    individual_outputs.append(individual_output_path)
                    print(f"    💾 Saved: class_{class_id}/{os.path.basename(individual_output_path)}")
            
            # Overlay colored mask
            cv2.addWeighted(colored_mask, alpha, result, 1 - alpha, 0, result)
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"🎉 Visualization saved: {output_path}")
        
        if save_individual_classes and individual_outputs:
            print(f"📂 Saved individual class images ({len(individual_outputs)} files):")
            for path in individual_outputs:
                # Relative paths for display
                relative_path = os.path.relpath(path, output_dir)
                print(f"   - {relative_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return None


def batch_visualize_by_names(image_names, images_dir, masks_dir, output_dir, contour_only=True, thickness=2, alpha=0.5, show_all_classes=True, save_individual_classes=False):
    """
    Batch visualize multiple images by filename.
    
    Args:
        image_names (list): list of filenames
        Other args are the same as visualize_single_image_by_name
    
    Returns:
        dict: {image_name: output_path or None}
    """
    print(f"🚀 Start batch processing {len(image_names)} images...")
    
    results = {}
    success_count = 0
    
    for i, image_name in enumerate(image_names):
        print(f"\n📸 Processing image {i+1}/{len(image_names)}...")
        result_path = visualize_single_image_by_name(
            image_name, images_dir, masks_dir, output_dir,
            contour_only, thickness, alpha, show_all_classes, save_individual_classes
        )
        
        results[image_name] = result_path
        if result_path is not None:
            success_count += 1
    
    print(f"\n🎉 Batch done: {success_count}/{len(image_names)} images processed successfully")
    return results


def parse_arguments():
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Image mask visualization utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: batch all images (show selected classes)
  python utils/visualize.py --mode 1

  # Mode 1: batch all images and save per-class outputs
  python utils/visualize.py --mode 1 --save_individual_classes

  # Mode 2: single image (show all classes)
  python utils/visualize.py --mode 2 --image_name "your_image_name"

  # Mode 2: single image and save per-class outputs
  python utils/visualize.py --mode 2 --image_name "your_image_name" --save_individual_classes

  # Mode 3: batch specified images (show all classes)
  python utils/visualize.py --mode 3 --image_names "img1,img2,img3"

  # Custom paths
  python utils/visualize.py --mode 2 --image_name "test" --images_dir "/path/to/images" --masks_dir "/path/to/masks"
        """
    )
    
    # Basic params
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=1,
                       help='Mode: 1=batch all (selected classes), 2=single image (all classes), 3=batch specified images (all classes). Default=1')
    
    # Paths
    parser.add_argument('--images_dir', type=str, 
                       default="/workspace/julie/SAM/data/lesion_all/train/images",
                       help='Images directory')
    parser.add_argument('--masks_dir', type=str,
                       default="/workspace/julie/SAM/data/lesion_all/train/masks", 
                       help='Masks directory')
    parser.add_argument('--output_dir', type=str,
                       default="/workspace/julie/SAM/data/lesion_all/train/visualize_finding_all",
                       help='Output directory')
    
    # Mode 2
    parser.add_argument('--image_name', type=str,
                       help='Image name (without extension) - required for mode 2')
    
    # Mode 3
    parser.add_argument('--image_names', type=str,
                       help='Comma-separated image names (e.g., img1,img2,img3) - required for mode 3')
    
    # Visualization params
    parser.add_argument('--contour_only', action='store_true',
                       help='Show contours only (no fill)')
    parser.add_argument('--thickness', type=int, default=2,
                       help='Contour thickness (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay alpha (0-1) (default: 0.5)')
    parser.add_argument('--show_filtered_classes', action='store_true',
                       help='In mode 2/3, show only selected classes instead of all')
    parser.add_argument('--save_individual_classes', action='store_true',
                       help='Save per-class images into separate folders (all modes)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("🚀 Image mask visualization")
    print(f"📁 Images dir: {args.images_dir}")
    print(f"📁 Masks dir: {args.masks_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"🔧 Mode: {args.mode}")
    
    if args.mode == 1:
        print("🚀 Mode 1: batch all images")
        print("📂 Will create per-class subfolders")
        process_directory(
            args.images_dir,
            args.masks_dir,
            args.output_dir,
            contour_only=args.contour_only,
            thickness=args.thickness,
            alpha=args.alpha,
            classes_to_visualize=CLASSES_TO_VISUALIZE,
            save_individual_classes=True  # default enable per-class saving
        )
    
    elif args.mode == 2:
        if not args.image_name:
            print("❌ Error: mode 2 requires --image_name")
            print("💡 Example: python utils/visualize.py --mode 2 --image_name 'your_image_name'")
            exit(1)
            
        print(f"🚀 Mode 2: single image ({args.image_name})")
        
        result_path = visualize_single_image_by_name(
            image_name=args.image_name,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            contour_only=args.contour_only,
            thickness=args.thickness,
            alpha=args.alpha,
            show_all_classes=not args.show_filtered_classes,
            save_individual_classes=args.save_individual_classes
        )
        
        if result_path:
            print(f"✅ Single image done! Output: {result_path}")
        else:
            print(f"❌ Single image failed!")
    
    elif args.mode == 3:
        if not args.image_names:
            print("❌ Error: mode 3 requires --image_names")
            print("💡 Example: python utils/visualize.py --mode 3 --image_names 'img1,img2,img3'")
            exit(1)
            
        target_images = [name.strip() for name in args.image_names.split(',')]
        print(f"🚀 Mode 3: batch specified images ({len(target_images)})")
        print(f"📝 Targets: {target_images}")
        
        results = batch_visualize_by_names(
            image_names=target_images,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            contour_only=args.contour_only,
            thickness=args.thickness,
            alpha=args.alpha,
            show_all_classes=not args.show_filtered_classes,
            save_individual_classes=args.save_individual_classes
        )
        
        success_list = [name for name, path in results.items() if path is not None]
        failed_list = [name for name, path in results.items() if path is None]
        
        print(f"\n📊 Summary:")
        print(f"✅ Success: {len(success_list)} - {success_list}")
        print(f"❌ Failed: {len(failed_list)} - {failed_list}")
    
    print("\n🎉 Done!")
