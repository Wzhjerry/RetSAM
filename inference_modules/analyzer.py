"""
Analysis module for calculating quantitative metrics from predictions.
"""

import numpy as np
import torch
import cv2
import tempfile
from typing import Dict, Tuple, Optional
import json
import os
import sys

# Add analysis directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from .config import InferenceConfig
from .utils import Logger, MaskProcessor
from .image_processor import ImageProcessor
from datasets.utils import remove_black_edge
from .analysis_schema import (
    AnalyzerResultAssembler,
    AnalyzerSummaryBuilder,
)


class Analyzer:
    """Handles quantitative analysis of segmentation results."""
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize analyzer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.mask_processor = MaskProcessor()
        self.image_processor = ImageProcessor()
        
        # Try to import analysis modules
        self._import_analysis_modules()
        
    def _import_analysis_modules(self):
        """Import analysis modules if available."""
        self.vessel_analyzer = None
        self.disc_analyzer = None
        self.lesion_analyzer = None
        self.tessellation_analyzer = None
        
        try:
            # Prefer overall vessel analysis interface
            from utils.airdoc_av_seg import analyze_vessel_mask_overall
            self.vessel_analyzer = analyze_vessel_mask_overall
        except ImportError:
            Logger.warning("Vessel analysis module not available")
            
        try:
            from utils.airdoc_dc_seg import visualize_glaucoma
            self.disc_analyzer = visualize_glaucoma
        except ImportError:
            Logger.warning("Disc/cup analysis module not available")
            
        try:
            from utils.analyze import analyze_lesion_mask, analyze_tessellate_mask, analyze_arc_lesion
            self.lesion_analyzer = analyze_lesion_mask
            self.tessellation_analyzer = analyze_tessellate_mask
            self.arc_lesion_analyzer = analyze_arc_lesion
        except ImportError:
            Logger.warning("Lesion/tessellation analysis modules not available")
            
    def calculate_task_metrics(self, prediction: torch.Tensor, task_name: str,
                             original_shape: Tuple[int, int]) -> Dict:
        """
        Lightweight placeholder: defer detailed metrics to analyze_all_tasks using
        dedicated external analysis modules for each task.
        """
        return {
            'task': task_name,
            'original_dimensions': {
                'width': original_shape[0],
                'height': original_shape[1]
            }
        }
        
    def analyze_all_tasks(self, mask_paths: Dict[str, str], original_image_path: str,
                         original_shape: Tuple[int, int], output_dir: Optional[str] = None,
                         macula_center: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Run advanced analysis on all tasks using saved masks.
        
        Args:
            mask_paths: Dictionary of task names to mask file paths
            original_image_path: Path to original image
            original_shape: Original image dimensions
            output_dir: Optional directory to save analysis results
            macula_center: Optional macula center coordinates (y, x) for improved eye side detection
            
        Returns:
            Dictionary of analysis results for all tasks
        """
        results = {}
        
        # Calculate analysis area
        analysis_area = self.config.analysis_size * self.config.analysis_size
        
        for task_name, mask_path_or_array in mask_paths.items():
            try:
                task_key = self.config.get_display_name(task_name)
                # Determine if we need to load the mask or use the path directly
                mask = None
                mask_path = None
                
                if isinstance(mask_path_or_array, str):
                    mask_path = mask_path_or_array
                    # Only load mask for tasks that need the array
                    if task_name in ['od_oc', 'tessellation'] or self.config.is_lesion_task(task_name):
                        if os.path.isdir(mask_path):
                            # Load multiple channel masks from directory
                            mask = self._load_multichannel_mask(mask_path)
                        else:
                            # Load single mask file
                            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                            if mask is None:
                                Logger.warning(f"Failed to load mask for {task_name}: {mask_path}")
                                continue
                else:
                    # If mask is already an array, use it directly
                    mask = mask_path_or_array
                
                if task_name == 'artery_vein' and self.vessel_analyzer:
                    # Accept in-memory mask or load from path
                    if isinstance(mask, np.ndarray):
                        vessel_mask = mask.copy()
                    elif mask_path and os.path.isfile(mask_path):
                        vessel_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if vessel_mask is None:
                            results[task_key] = {'error': f'Failed to read vessel mask: {mask_path}'}
                            continue
                    else:
                        results[task_key] = {'error': 'Vessel analyzer requires mask file or array'}
                        continue

                    if len(vessel_mask.shape) != 2:
                        vessel_mask = cv2.cvtColor(vessel_mask, cv2.COLOR_BGR2GRAY)
                    ori = cv2.imread(original_image_path)
                    if ori is not None:
                        _, ymin, ymax, xmin, xmax = remove_black_edge(ori)
                        h, w = vessel_mask.shape[:2]
                        ymin = max(0, min(int(ymin), h))
                        ymax = max(0, min(int(ymax), h))
                        xmin = max(0, min(int(xmin), w))
                        xmax = max(0, min(int(xmax), w))
                        if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                            vessel_mask = vessel_mask[ymin:ymax, xmin:xmax]
                    vessel_mask = cv2.resize(
                        vessel_mask,
                        (self.config.analysis_size, self.config.analysis_size),
                        interpolation=cv2.INTER_NEAREST
                    )
                    # Build artery/vein binary masks from multi-class mask
                    artery_mask = ((vessel_mask == 1) | (vessel_mask == 3)).astype(np.uint8)
                    vein_mask = ((vessel_mask == 2) | (vessel_mask == 3)).astype(np.uint8)
                    vessel_union = ((artery_mask + vein_mask) > 0).astype(np.uint8)
                    results[task_key] = self.vessel_analyzer(vessel_union, artery_mask, vein_mask)
                        
                elif task_name == 'od_oc' and self.disc_analyzer:
                    # visualize_glaucoma expects (image, mask, ...)
                    if mask is not None:
                        original_image = cv2.imread(original_image_path)
                        if original_image is None:
                             results[task_key] = {'error': f'Failed to load image: {original_image_path}'}
                        else:
                            # Align both image and mask to black-edge crop and analysis size
                            try:
                                _, ymin, ymax, xmin, xmax = remove_black_edge(original_image)
                                h, w = mask.shape[:2]
                                ymin = max(0, min(int(ymin), h))
                                ymax = max(0, min(int(ymax), h))
                                xmin = max(0, min(int(xmin), w))
                                xmax = max(0, min(int(xmax), w))
                                if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                                    mask = mask[ymin:ymax, xmin:xmax]
                                    original_image = original_image[ymin:ymax, xmin:xmax]
                            except Exception:
                                pass
                            # Resize to analysis size
                            original_image = cv2.resize(
                                original_image,
                                (self.config.analysis_size, self.config.analysis_size),
                                interpolation=cv2.INTER_LINEAR
                            )
                            if len(mask.shape) != 2:
                                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            mask = cv2.resize(
                                mask,
                                (self.config.analysis_size, self.config.analysis_size),
                                interpolation=cv2.INTER_NEAREST
                            )
                            # Convert macula center to analysis size coordinates if available
                            macula_center_analysis = None
                            if macula_center is not None:
                                scale_y = self.config.analysis_size / original_shape[1]
                                scale_x = self.config.analysis_size / original_shape[0]
                                macula_center_analysis = (int(macula_center[1] * scale_x), int(macula_center[0] * scale_y))  # (x, y) format
                            
                            odoc_result = self.disc_analyzer(original_image, mask, macula_center=macula_center_analysis)
                            # Do not save/render analysis image for od_oc in quantitative_analysis
                            if isinstance(odoc_result, dict) and 'image' in odoc_result:
                                odoc_result.pop('image', None)
                            results[task_key] = odoc_result
                    else:
                        results[task_key] = {'error': 'Failed to load mask for od_oc'}
                        
                elif task_name in self.config.get_grouped_lesion_names():
                    try:
                        results[task_key] = self._analyze_grouped_lesion_mask(
                            mask_path_or_array,
                            task_name,
                            original_image_path,
                            macula_center,
                            results
                        )
                    except Exception as e:
                        results[task_key] = {'error': f'Grouped lesion analysis failed: {e}'}

                elif self.config.is_lesion_task(task_name):
                    # Prefer per-category analysis using per-class masks saved under masks/
                    try:
                        display_name = self.config.get_display_name(task_name)
                        if mask_path and os.path.isfile(mask_path):
                            masks_dir = os.path.dirname(mask_path)
                        else:
                            # If mask_path is a directory (rare), use it; otherwise fallback to output dir
                            masks_dir = os.path.dirname(mask_path) if isinstance(mask_path, str) else ''

                        # Prefer grouped lesion mappings when analyzing regrouped outputs
                        lesion_types_map = self.config.get_lesion_types(task_name)
                        if not lesion_types_map and task_name in self.config.get_grouped_lesion_names():
                            lesion_types_map = self.config.get_grouped_lesion_types(task_name)
                        min_comp_size = int(self.config.get_analysis_min_component_size(task_name))
                        from utils.analyze import analyze_lesions_by_categories
                        # Get eye side information from DC analysis results
                        is_left_eye = None
                        if 'od_oc' in results and 'eyeside_detected' in results['od_oc']:
                            is_left_eye = (results['od_oc']['eyeside_detected'] == 0)
                        
                        per_class_results = analyze_lesions_by_categories(
                            masks_dir=masks_dir,
                            display_name=display_name,
                            lesion_types_map=lesion_types_map,
                            analysis_size=self.config.analysis_size,
                            min_component_size=min_comp_size,
                            original_image_path=original_image_path,
                            macula_center=macula_center,
                            is_left_eye=is_left_eye
                        )
                        # Always return the unified per-category schema, even if empty
                        results[task_key] = per_class_results
                    except Exception as e:
                        results[task_key] = {'error': f'Lesion category analysis failed: {e}'}
                    
                elif task_name == 'tessellation' and self.tessellation_analyzer:
                    # Align tessellation mask with black-edge crop and resize to analysis size
                    if mask is not None:
                        fundus_area_dyn = None
                        try:
                            original_image = cv2.imread(original_image_path)
                            if original_image is not None:
                                # Compute black-edge crop on original image
                                _, ymin, ymax, xmin, xmax = remove_black_edge(original_image)
                                # Crop tessellation mask to same box
                                h, w = mask.shape[:2]
                                ymin_m = max(0, min(int(ymin), h))
                                ymax_m = max(0, min(int(ymax), h))
                                xmin_m = max(0, min(int(xmin), w))
                                xmax_m = max(0, min(int(xmax), w))
                                if (ymax_m - ymin_m) > 0 and (xmax_m - xmin_m) > 0:
                                    if len(mask.shape) != 2:
                                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                                    mask = mask[ymin_m:ymax_m, xmin_m:xmax_m]

                                # Prepare grayscale cropped original for fundus mask
                                try:
                                    orig_h, orig_w = original_image.shape[:2]
                                    ymin_o = max(0, min(int(ymin), orig_h))
                                    ymax_o = max(0, min(int(ymax), orig_h))
                                    xmin_o = max(0, min(int(xmin), orig_w))
                                    xmax_o = max(0, min(int(xmax), orig_w))
                                    if (ymax_o - ymin_o) > 0 and (xmax_o - xmin_o) > 0:
                                        original_image = original_image[ymin_o:ymax_o, xmin_o:xmax_o]
                                    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                                    # Threshold to get fundus region; then morphological cleanup
                                    bg_thr = 10  # background threshold
                                    fundus_mask = (gray > bg_thr).astype(np.uint8)
                                    kernel = np.ones((5, 5), np.uint8)
                                    fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_OPEN, kernel)
                                    fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_CLOSE, kernel)
                                    # Resize to analysis size and compute area
                                    fundus_mask = cv2.resize(
                                        fundus_mask,
                                        (self.config.analysis_size, self.config.analysis_size),
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                    fundus_area_dyn = int(np.sum(fundus_mask))
                                except Exception:
                                    fundus_area_dyn = None
                        except Exception:
                            pass

                        # Ensure single-channel and normalize resolution for consistent stats
                        if len(mask.shape) != 2:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        mask = cv2.resize(
                            mask,
                            (self.config.analysis_size, self.config.analysis_size),
                            interpolation=cv2.INTER_NEAREST
                        )
                        # Use dynamic fundus area if available; fallback to analysis_area
                        fundus_area_final = int(fundus_area_dyn) if isinstance(fundus_area_dyn, (int, float)) and fundus_area_dyn > 0 else analysis_area
                        # Pass min_component_size for noise control
                        min_comp = int(self.config.get_analysis_min_component_size('tessellation'))
                        results[task_key] = self.tessellation_analyzer(
                            mask, None, 10, fundus_area_final, min_component_size=min_comp
                        )
                    else:
                        results[task_key] = {'error': 'Failed to load mask for tessellation'}
                    
                elif task_name == 'myopia':
                    # New myopia analysis with 4-channel classes: 0=bg, 1=arc, 2=diffuse, 3=patchy
                    try:
                        if mask is None and mask_path and os.path.isfile(mask_path):
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            results[task_key] = {'error': 'Failed to load mask for myopia'}
                            continue

                        # Align mask region with black-edge removal from original image
                        ori = cv2.imread(original_image_path)
                        fundus_mask = None
                        if ori is not None:
                            _, ymin, ymax, xmin, xmax = remove_black_edge(ori)
                            h, w = mask.shape[:2]
                            ymin = max(0, min(int(ymin), h))
                            ymax = max(0, min(int(ymax), h))
                            xmin = max(0, min(int(xmin), w))
                            xmax = max(0, min(int(xmax), w))
                            if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                                mask = mask[ymin:ymax, xmin:xmax]
                            # Build fundus mask from cropped original
                            try:
                                oh, ow = ori.shape[:2]
                                ymin_o = max(0, min(int(ymin), oh))
                                ymax_o = max(0, min(int(ymax), oh))
                                xmin_o = max(0, min(int(xmin), ow))
                                xmax_o = max(0, min(int(xmax), ow))
                                if (ymax_o - ymin_o) > 0 and (xmax_o - xmin_o) > 0:
                                    ori = ori[ymin_o:ymax_o, xmin_o:xmax_o]
                                gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
                                bg_thr = 10
                                fundus_mask = (gray > bg_thr).astype(np.uint8)
                                kernel = np.ones((5, 5), np.uint8)
                                fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_OPEN, kernel)
                                fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_CLOSE, kernel)
                            except Exception:
                                fundus_mask = None

                        # Resize to analysis resolution
                        mask = cv2.resize(mask, (self.config.analysis_size, self.config.analysis_size), interpolation=cv2.INTER_NEAREST)
                        if fundus_mask is not None:
                            fundus_mask = cv2.resize(fundus_mask, (self.config.analysis_size, self.config.analysis_size), interpolation=cv2.INTER_NEAREST)
                            analysis_area = float(np.sum(fundus_mask))
                        else:
                            analysis_area = float(self.config.analysis_size * self.config.analysis_size)

                        class_map = mask.astype(np.uint8)
                        small_threshold = 50
                        medium_threshold = 200

                        # Use config-driven min component filtering for myopia
                        min_comp = int(self.config.get_analysis_min_component_size('myopia'))

                        def comp_stats(bin_mask: np.ndarray):
                            s = self._analyze_lesion_components(bin_mask, small_threshold, medium_threshold, min_component_size=min_comp)
                            area_sum = int(s['total_area'])
                            count = int(s['count'])
                            coverage = float(area_sum / analysis_area) if analysis_area > 0 else 0.0
                            return s, area_sum, count, coverage

                        overall_mask = (class_map > 0).astype(np.uint8)
                        overall_stats, overall_area, overall_count, overall_cov = comp_stats(overall_mask)

                        categories = {
                            1: {'key': 'arc_lesion', 'chinese_name': 'Arc lesion'},
                            2: {'key': 'diffuse_chorioretinal_atrophy', 'chinese_name': 'Diffuse chorioretinal atrophy'},
                            3: {'key': 'patchy_chorioretinal_atrophy', 'chinese_name': 'Patchy chorioretinal atrophy'},
                        }
                        categories_overview = {}
                        detailed_analysis = {}
                        total_area_pixels = 0
                        total_count = 0
                        total_large = 0

                        for cid, info in categories.items():
                            bin_mask = (class_map == cid).astype(np.uint8)
                            stats, area_sum, count, coverage = comp_stats(bin_mask)
                            total_area_pixels += area_sum
                            total_count += count
                            total_large += int(stats['size_distribution']['large_count'])

                            categories_overview[info['key']] = {
                                'count': count,
                                'total_area_pixels': area_sum,
                                'coverage_percentage': coverage,
                            }
                            detailed_analysis[info['key']] = {
                                'measurements': {
                                    'count': count,
                                    'total_area_pixels': area_sum,
                                    'coverage_percentage': coverage,
                                    'average_size_pixels': float(stats['total_area'] / stats['count']) if stats['count'] > 0 else 0.0,
                                },
                                'shape_characteristics': {
                                    'circularity': float(stats['morphological_metrics']['avg_circularity']) if stats['count'] > 0 else 0.0,
                                    'aspect_ratio': float(stats['morphological_metrics']['avg_aspect_ratio']) if stats['count'] > 0 else 0.0,
                                },
                            }

                        results[task_key] = {
                            'overall': {
                                'count': overall_stats['count'],
                                'total_area': overall_area,
                                'coverage_ratio': float(overall_area / analysis_area) if analysis_area > 0 else 0.0,
                            },
                            'categories': categories_overview,
                            'detailed_analysis': detailed_analysis,
                            # Backward-compatible flat keys used by classifier
                            'myopia_total_lesion_area': total_area_pixels,
                            'myopia_large_lesions': total_large,
                            
                        }
                    except Exception as e:
                        results[task_key] = {'error': f'Myopia analysis failed: {e}'}
            except Exception as e:
                Logger.warning(f"Analysis failed for {task_name}: {e}")
                results[task_name] = {'error': str(e)}
        
        assembled = AnalyzerResultAssembler(self.config).assemble(
            task_results=results,
            image_path=original_image_path,
            original_shape=original_shape,
            macula_center=macula_center,
        )

        if output_dir:
            output_path = os.path.join(output_dir, 'quantitative_analysis.json')
            try:
                from .utils import convert_numpy_types
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(convert_numpy_types(assembled), f, ensure_ascii=False, indent=2)
                Logger.info(f"Quantitative analysis saved to: {output_path}")
            except Exception as e:
                Logger.error(f"Failed to save analysis results: {e}")
        return assembled
        
    def _load_multichannel_mask(self, mask_dir: str) -> np.ndarray:
        """
        Load multi-channel masks from directory.
        
        Args:
            mask_dir: Directory containing channel_*.png files
            
        Returns:
            Multi-channel mask array
        """
        channel_files = sorted([f for f in os.listdir(mask_dir) if f.startswith('channel_')])
        if not channel_files:
            raise ValueError(f"No channel files found in {mask_dir}")
            
        masks = []
        for channel_file in channel_files:
            channel_path = os.path.join(mask_dir, channel_file)
            channel_mask = cv2.imread(channel_path, cv2.IMREAD_GRAYSCALE)
            if channel_mask is None:
                raise ValueError(f"Failed to load channel mask: {channel_path}")
            masks.append(channel_mask)
            
        # Stack masks along channel dimension
        return np.stack(masks, axis=-1) if len(masks) > 1 else masks[0]

    def _analyze_grouped_lesion_mask(self, mask_input, task_name: str,
                                     original_image_path: str,
                                     macula_center: Optional[Tuple[float, float]],
                                     results: Dict) -> Dict:
        """Analyze grouped lesion outputs from either file paths or in-memory masks."""
        display_name = self.config.get_display_name(task_name)
        grouped_lesion_types_map = self.config.get_grouped_lesion_types(task_name)
        min_comp_size = int(self.config.get_analysis_min_component_size('lesion_s1'))  # reuse lesion threshold
        from utils.analyze import analyze_lesions_by_categories
        # Get eye side information from DC analysis results
        is_left_eye = None
        if 'od_oc' in results and 'eyeside_detected' in results['od_oc']:
            is_left_eye = (results['od_oc']['eyeside_detected'] == 0)

        def run_analysis(masks_dir: str):
            return analyze_lesions_by_categories(
                masks_dir=masks_dir,
                display_name=display_name,
                lesion_types_map=grouped_lesion_types_map,
                analysis_size=self.config.analysis_size,
                min_component_size=min_comp_size,
                original_image_path=original_image_path,
                macula_center=macula_center,
                is_left_eye=is_left_eye
            )

        if isinstance(mask_input, np.ndarray):
            # Create temporary per-class masks
            with tempfile.TemporaryDirectory() as tmpdir:
                for class_id, name in grouped_lesion_types_map.items():
                    bin_mask = (mask_input == int(class_id)).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(tmpdir, f'{display_name}_{name}.png'), bin_mask)
                return run_analysis(tmpdir)
        else:
            masks_dir = ''
            if mask_input and isinstance(mask_input, str):
                if os.path.isfile(mask_input):
                    masks_dir = os.path.dirname(mask_input)
                elif os.path.isdir(mask_input):
                    masks_dir = mask_input
            return run_analysis(masks_dir)
        
    # Deprecated: per-task calculators are replaced by dedicated external analysis
    # modules invoked in analyze_all_tasks. These stubs are kept for reference.
    def _calculate_vessel_metrics(self, prediction: np.ndarray,
                                original_shape: Tuple[int, int]) -> Dict:
        return {}
        
    def _calculate_disc_cup_metrics(self, prediction: np.ndarray,
                                  original_shape: Tuple[int, int]) -> Dict:
        return {}
        
    def _calculate_lesion_metrics(self, prediction: np.ndarray, task_name: str,
                                original_shape: Tuple[int, int]) -> Dict:
        """
        Deprecated: lesion metrics are computed via dedicated analysis in
        utils.analyze (e.g., analyze_lesion_mask) inside analyze_all_tasks.
        This stub is kept for compatibility and returns an empty dict.
        """
        return {}

    def _analyze_lesions_by_categories(self, masks_dir: str, display_name: str,
                                       lesion_types_map: Dict[int, str], analysis_area: int,
                                       min_component_size: int, original_image_path: str) -> Dict:
        """Analyze lesions per category using saved per-class masks in masks_dir.

        Expects files named like: f"{display_name}_{lesion_name}.png" saved under masks_dir.
        Returns a JSON-friendly dict summarizing per-category metrics and overall summary.
        """
        import numpy as np
        import cv2

        if not masks_dir or not os.path.isdir(masks_dir) or not lesion_types_map:
            return {
                'summary': {
                    'total_lesion_categories': 0,
                    'total_area_pixels': 0,
                    'total_coverage_percentage': 0.0,
                },
                'lesion_categories_analysis': {
                    'total_categories_detected': 0,
                    'categories_overview': {},
                    'detailed_analysis': {},
                }
            }

        categories_overview: Dict[str, Dict] = {}
        detailed_analysis: Dict[str, Dict] = {}
        total_area_pixels = 0
        categories_detected = 0
        fundus_area_pixels = float(analysis_area) if analysis_area and analysis_area > 0 else float(self.config.analysis_size ** 2)

        # Helper to compute per-mask stats
        def compute_mask_stats(binary_mask: np.ndarray) -> Dict:
            import cv2
            import numpy as np
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
            areas = []
            perimeters = []
            circularities = []
            aspect_ratios = []
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                # Enforce analysis-stage min component size from config
                if area < max(1, int(min_component_size)):
                    continue
                areas.append(area)
                comp_mask = (labels == label_id).astype(np.uint8)
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = contours[0]
                    peri = float(cv2.arcLength(cnt, True))
                    perimeters.append(peri)
                    if peri > 0:
                        circ = float(min(1.0, 4 * np.pi * area / (peri * peri)))
                        circularities.append(circ)
                    rect = cv2.minAreaRect(cnt)
                    w, h = rect[1]
                    if h > 0 and w > 0:
                        aspect_ratios.append(float(max(w, h) / min(w, h)))
            # Quadrant distribution based on centroid relative to image center
            h, w = binary_mask.shape[:2]
            cy, cx = h // 2, w // 2
            q_counts = {
                'superior_temporal': 0,   # upper-right for right eye; we don't rely on eye later
                'superior_nasal': 0,      # upper-left
                'inferior_temporal': 0,   # lower-right
                'inferior_nasal': 0,      # lower-left
            }
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < max(1, int(min_component_size)):
                    continue
                x = int(stats[label_id, cv2.CC_STAT_LEFT]) + int(stats[label_id, cv2.CC_STAT_WIDTH]) // 2
                y = int(stats[label_id, cv2.CC_STAT_TOP]) + int(stats[label_id, cv2.CC_STAT_HEIGHT]) // 2
                if y < cy and x >= cx:
                    q_counts['superior_temporal'] += 1
                elif y < cy and x < cx:
                    q_counts['superior_nasal'] += 1
                elif y >= cy and x >= cx:
                    q_counts['inferior_temporal'] += 1
                else:
                    q_counts['inferior_nasal'] += 1
            quadrants_with_lesions = sum(1 for v in q_counts.values() if v > 0)
            return {
                'count': len(areas),
                'areas': areas,
                'perimeters': perimeters,
                'circularities': circularities,
                'aspect_ratios': aspect_ratios,
                'quadrant_counts': q_counts,
                'quadrants_with_lesions': quadrants_with_lesions,
            }

        # Compute black-edge crop coordinates from original image once
        crop_box = None
        try:
            ori = cv2.imread(original_image_path)
            if ori is not None:
                _, ymin, ymax, xmin, xmax = remove_black_edge(ori)
                crop_box = (int(ymin), int(ymax), int(xmin), int(xmax))
        except Exception:
            crop_box = None

        for class_id, lesion_name in lesion_types_map.items():
            file_path = os.path.join(masks_dir, f'{display_name}_{lesion_name}.png')
            if not os.path.isfile(file_path):
                continue
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Apply black-edge crop alignment if available
            if crop_box is not None:
                ymin, ymax, xmin, xmax = crop_box
                h, w = mask.shape[:2]
                ymin = max(0, min(int(ymin), h))
                ymax = max(0, min(int(ymax), h))
                xmin = max(0, min(int(xmin), w))
                xmax = max(0, min(int(xmax), w))
                if (ymax - ymin) > 0 and (xmax - xmin) > 0:
                    mask = mask[ymin:ymax, xmin:xmax]
            # Normalize to analysis resolution (e.g., 640x640) for uniform statistics
            mask_resized = cv2.resize(
                (mask > 0).astype(np.uint8),
                (self.config.analysis_size, self.config.analysis_size),
                interpolation=cv2.INTER_NEAREST
            )

            stats = compute_mask_stats(mask_resized)
            if stats['count'] == 0:
                continue

            categories_detected += 1
            area_sum = int(np.sum(stats['areas']))
            total_area_pixels += area_sum
            coverage = float(area_sum / fundus_area_pixels) if fundus_area_pixels > 0 else 0.0

            # Overview
            categories_overview[lesion_name] = {
                'count': stats['count'],
                'coverage_percentage': coverage,
                'severity': (
                    'Minimal' if coverage < 0.01 else 'Mild' if coverage < 0.03 else 'Moderate' if coverage < 0.1 else 'Severe'
                )
            }

            # Detailed
            size_small = sum(1 for a in stats['areas'] if a < 50)
            size_medium = sum(1 for a in stats['areas'] if 50 <= a < 200)
            size_large = sum(1 for a in stats['areas'] if a >= 200)

            detailed_analysis[lesion_name] = {
                'measurements': {
                    'count': stats['count'],
                    'total_area_pixels': area_sum,
                    'coverage_percentage': coverage,
                    'average_size_pixels': float(np.mean(stats['areas'])) if stats['areas'] else 0.0,
                },
                'shape_characteristics': {
                    'circularity': float(np.mean(stats['circularities'])) if stats['circularities'] else 0.0,
                    'aspect_ratio': float(np.mean(stats['aspect_ratios'])) if stats['aspect_ratios'] else 0.0,
                },
                'quadrant_distribution': {
                    'quadrants_with_lesions': int(stats.get('quadrants_with_lesions', 0)),
                    'counts': stats.get('quadrant_counts', {
                        'superior_temporal': 0,
                        'superior_nasal': 0,
                        'inferior_temporal': 0,
                        'inferior_nasal': 0,
                    })
                },
                'clinical_assessment': {
                    'lesion_to_disc_ratio': coverage,  # Legacy key now uses fundus area
                    'lesion_to_fundus_ratio': coverage,
                    'severity_level': (
                        'Minimal' if coverage < 0.01 else 'Mild' if coverage < 0.03 else 'Moderate' if coverage < 0.1 else 'Severe'
                    ),
                }
            }

        return {
            'summary': {
                'total_lesion_categories': categories_detected,
                'total_area_pixels': total_area_pixels,
                'total_coverage_percentage': float(total_area_pixels / fundus_area_pixels) if fundus_area_pixels > 0 else 0.0,
                'fundus_area_pixels': int(fundus_area_pixels),
            },
            'lesion_categories_analysis': {
                'total_categories_detected': categories_detected,
                'categories_overview': categories_overview,
                'detailed_analysis': detailed_analysis,
            }
        }
    
    def _analyze_lesion_components(self, lesion_mask: np.ndarray, small_threshold: int, medium_threshold: int, min_component_size: int = 0) -> Dict:
        """
        Analyze connected components in a lesion mask to get detailed statistics.
        
        Args:
            lesion_mask: Binary mask for a specific lesion type
            small_threshold: Maximum pixels for small lesions
            medium_threshold: Maximum pixels for medium lesions
            
        Returns:
            Dictionary with detailed lesion statistics
        """
        import cv2
        import numpy as np
        from scipy import ndimage
        
        # Find connected components with stats for area-based filtering
        num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(lesion_mask.astype(np.uint8), connectivity=8)
        
        # Initialize statistics (do not shadow comp_stats)
        stats = {
            'count': 0,
            'total_area': 0,
            'avg_area': 0.0,
            'size_distribution': {
                'small_count': 0,
                'medium_count': 0,
                'large_count': 0,
                'small_total_area': 0,
                'medium_total_area': 0,
                'large_total_area': 0,
                'small_avg_area': 0.0,
                'medium_avg_area': 0.0,
                'large_avg_area': 0.0
            },
            'morphological_metrics': {
                'avg_circularity': 0.0,
                'avg_aspect_ratio': 0.0,
                'avg_solidity': 0.0
            },
            'coverage_ratio': 0.0
        }
        
        if num_labels <= 1:  # No lesions found (background only)
            return stats
            
        areas = []
        circularities = []
        aspect_ratios = []
        solidities = []
        
        # Analyze each component (skip label 0 which is background)
        min_area = max(1, int(min_component_size))
        for label in range(1, num_labels):
            area = int(comp_stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component_mask = (labels == label)
                
            areas.append(area)
            stats['count'] += 1
            stats['total_area'] += area
            
            # Calculate morphological properties
            try:
                contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    
                    # Circularity: 4π * area / perimeter^2
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        circularities.append(min(circularity, 1.0))  # Cap at 1.0
                    
                    # Aspect ratio from bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        aspect_ratios.append(aspect_ratio)
                    
                    # Solidity: area / convex_hull_area
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        solidities.append(solidity)
                        
            except Exception:
                # If morphological analysis fails, continue with area-based statistics
                pass
            
            # Size classification
            if area <= small_threshold:
                stats['size_distribution']['small_count'] += 1
                stats['size_distribution']['small_total_area'] += area
            elif area <= medium_threshold:
                stats['size_distribution']['medium_count'] += 1
                stats['size_distribution']['medium_total_area'] += area
            else:
                stats['size_distribution']['large_count'] += 1
                stats['size_distribution']['large_total_area'] += area
        
        # Calculate averages
        if stats['count'] > 0:
            stats['avg_area'] = stats['total_area'] / stats['count']
            stats['coverage_ratio'] = stats['total_area'] / (self.config.analysis_size ** 2)
            
            # Size-specific averages
            if stats['size_distribution']['small_count'] > 0:
                stats['size_distribution']['small_avg_area'] = stats['size_distribution']['small_total_area'] / stats['size_distribution']['small_count']
            if stats['size_distribution']['medium_count'] > 0:
                stats['size_distribution']['medium_avg_area'] = stats['size_distribution']['medium_total_area'] / stats['size_distribution']['medium_count']
            if stats['size_distribution']['large_count'] > 0:
                stats['size_distribution']['large_avg_area'] = stats['size_distribution']['large_total_area'] / stats['size_distribution']['large_count']
            
            # Morphological averages
            if circularities:
                stats['morphological_metrics']['avg_circularity'] = float(np.mean(circularities))
            if aspect_ratios:
                stats['morphological_metrics']['avg_aspect_ratio'] = float(np.mean(aspect_ratios))
            if solidities:
                stats['morphological_metrics']['avg_solidity'] = float(np.mean(solidities))
                
        return stats
        
    def _calculate_tessellation_metrics(self, prediction: np.ndarray,
                                      original_shape: Tuple[int, int]) -> Dict:
        """Calculate tessellation metrics with detailed component analysis."""
        if len(prediction.shape) == 3:
            mask = np.argmax(prediction, axis=0)
        else:
            mask = prediction
            
        # Use the same detailed analysis as lesions
        small_threshold = 50    # pixels
        medium_threshold = 200  # pixels
        
        tessellation_mask = (mask > 0).astype(np.uint8)
        tessellation_stats = self._analyze_lesion_components(tessellation_mask, small_threshold, medium_threshold)
        
        # Add tessellation-specific metrics
        tessellation_stats['tessellate_count'] = tessellation_stats.get('count', 0)
        tessellation_stats['fundus_area_pixels'] = int(self.config.analysis_size ** 2)
        tessellation_stats['tessellation_coverage_ratio'] = tessellation_stats['coverage_ratio']
        tessellation_stats['tessellation_density'] = tessellation_stats['count'] / (self.config.analysis_size ** 2) if tessellation_stats['count'] > 0 else 0
        tessellation_stats['severity'] = self._assess_tessellation_severity(tessellation_stats['coverage_ratio'])
        
        # Add pattern characteristics
        tessellation_stats['pattern_characteristics'] = {
            'regularity_score': min(tessellation_stats['morphological_metrics']['avg_circularity'] * 1.2, 1.0),
            'spacing_uniformity': max(1.0 - (tessellation_stats['morphological_metrics']['avg_aspect_ratio'] - 1.0) * 0.3, 0.0)
        }
        
        return tessellation_stats
        
    def _calculate_myopia_metrics(self, prediction: np.ndarray,
                                original_shape: Tuple[int, int]) -> Dict:
        """Calculate myopia-related metrics with detailed component analysis."""
        if len(prediction.shape) == 3:
            class_map = np.argmax(prediction, axis=0).astype(np.uint8)
        else:
            class_map = prediction.astype(np.uint8)
            
        # Use the same detailed analysis as lesions
        small_threshold = 50    # pixels
        medium_threshold = 200  # pixels
        
        analysis_area = self.config.analysis_size * self.config.analysis_size

        # Overall (mask>0)
        overall_mask = (class_map > 0).astype(np.uint8)
        overall_stats = self._analyze_lesion_components(overall_mask, small_threshold, medium_threshold)
        overall_stats['atrophy_coverage_ratio'] = overall_stats['coverage_ratio']
        overall_stats['severity'] = self._assess_myopia_severity(overall_stats['coverage_ratio'])

        # Per-category analysis for three classes
        categories = {
            1: {
                'key': 'arc_lesion',
                'chinese_name': 'Arc lesion'
            },
            2: {
                'key': 'diffuse_chorioretinal_atrophy',
                'chinese_name': 'Diffuse chorioretinal atrophy'
            },
            3: {
                'key': 'patchy_chorioretinal_atrophy',
                'chinese_name': 'Patchy chorioretinal atrophy'
            }
        }

        categories_overview: Dict[str, Dict] = {}
        detailed_analysis: Dict[str, Dict] = {}
        total_area_pixels = 0
        total_count = 0

        for class_id, info in categories.items():
            cat_mask = (class_map == class_id).astype(np.uint8)
            stats = self._analyze_lesion_components(cat_mask, small_threshold, medium_threshold)
            area_sum = int(stats['total_area'])
            count = int(stats['count'])
            coverage = float(area_sum / analysis_area) if analysis_area > 0 else 0.0

            total_area_pixels += area_sum
            total_count += count

            # Overview
            categories_overview[info['key']] = {
                'count': count,
                'coverage_percentage': coverage
            }

            # Detailed
            detailed_analysis[info['key']] = {
                'lesion_info': {
                    'name': info['key'],
                    'chinese_name': info['chinese_name'],
                    'class_index': class_id
                },
                'measurements': {
                    'count': count,
                    'total_area_pixels': area_sum,
                    'coverage_percentage': coverage,
                    'average_size_pixels': float(stats['total_area'] / stats['count']) if stats['count'] > 0 else 0.0
                },
                'shape_characteristics': {
                    'circularity': float(stats['morphological_metrics']['avg_circularity']) if stats['count'] > 0 else 0.0,
                    'aspect_ratio': float(stats['morphological_metrics']['avg_aspect_ratio']) if stats['count'] > 0 else 0.0
                }
            }

        # Compose result
        results = {
            'myopia_overall': overall_stats,
            'myopia_categories_analysis': {
                'total_categories_detected': sum(1 for v in categories_overview.values() if v['count'] > 0),
                'categories_overview': categories_overview,
                'detailed_analysis': detailed_analysis
            },
            # For backward compatibility with grading expectations
            'myopia_total_lesion_area': total_area_pixels,
        }

        return results
        
    def _assess_tessellation_severity(self, coverage_ratio: float) -> str:
        """Assess tessellation severity based on coverage."""
        coverage_percentage = coverage_ratio * 100
        
        if coverage_percentage < 5:
            return "Minimal"
        elif coverage_percentage < 15:
            return "Mild"
        elif coverage_percentage < 30:
            return "Moderate"
        else:
            return "Severe"

            
    def _assess_myopia_severity(self, coverage_ratio: float) -> str:
        """Assess myopia severity based on feature coverage."""
        coverage_percentage = coverage_ratio * 100
        
        if coverage_percentage < 2:
            return "Minimal"
        elif coverage_percentage < 5:
            return "Mild"
        elif coverage_percentage < 10:
            return "Moderate"
        else:
            return "Severe"
