"""
Simple programmatic entry point for RETSam.

Usage:
    from main import RetSAM
    retsam = RetSAM(
        model_path="/path/to/ckpt.ckpt",
        device="cuda",
    )
    result = retsam.inference(
        img_path="/path/to/image.png",
        save_path="/tmp/outs",
        visualization=True,
        analysis=True,
    )
    # result contains masks/visualization paths and analysis results (if enabled)
"""

import os
from typing import Optional, Tuple, Dict, Any

from inference import FundusInference


class RetSAM:
    """Lightweight wrapper to expose RETSam as a simple Python API."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        multitask: bool = True,
        output_channels: Optional[Tuple[int, ...]] = None,
        has_coordinate_head: Optional[bool] = None,
        num_coordinates: Optional[int] = None,
        enable_noise_filter: bool = True,
        quiet: bool = True,
    ):
        """
        Initialize RETSam and load the model once.
        """
        # Default output_dir; can be overridden per call in inference()
        default_output_dir = os.path.join(os.getcwd(), "retsam_outputs")

        self.runner = FundusInference(
            model_path=model_path,
            output_dir=default_output_dir,
            device=device,
            multitask=multitask,
            output_channels=output_channels,
            has_coordinate_head=has_coordinate_head,
            num_coordinates=num_coordinates,
            enable_disease_classification=False,
            enable_noise_filter=enable_noise_filter,
            input_root=None,
            binary_masks_only=False,
            analysis_only=False,
            save_visualizations=True,
            quiet=quiet,
        )

    def inference(
        self,
        img_path: str,
        save_path: Optional[str] = None,
        visualization: bool = True,
        analysis: bool = True,
        binary_masks_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            img_path: Path to the input image.
            save_path: Directory to save outputs; if None, uses the default.
            visualization: Save visual overlays (combined and per-task).
            analysis: Run quantitative analysis; if True, sets run_analysis.
            binary_masks_only: Only save 0/255 masks, skip visualization/analysis outputs.
        Returns:
            Dictionary with masks/visualization paths and (optional) analysis.
        """
        # Update output directory if provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.runner.output_dir = save_path

        # Configure per-call flags
        self.runner.save_visualizations = visualization and not binary_masks_only

        if not visualization and not analysis:
            # No visuals, no analysis: do not save masks/visuals
            self.runner.binary_masks_only = False
            self.runner.analysis_only = True   # blocks saving masks/visuals in _save_results
            self.runner.run_analysis = False
        else:
            # Respect caller settings
            self.runner.binary_masks_only = binary_masks_only
            # analysis_only means no visual saving; masks kept in-memory for analysis
            self.runner.analysis_only = analysis and not visualization
            # Only run analysis when requested and masks are available
            self.runner.run_analysis = analysis and not binary_masks_only

        return self.runner.process_single_image(img_path)


__all__ = ["RetSAM"]
