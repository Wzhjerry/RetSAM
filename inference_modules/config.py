"""
Configuration module for inference pipeline.
"""

from typing import Dict, List, Tuple, Optional


class InferenceConfig:
    """Central configuration for inference pipeline."""
    
    def __init__(self):
        # Task configurations
        self.task_names = [
            'artery_vein', 'od_oc', 'tessellation', 'myopia', 'lesion_s1'
        ]
        
        self.lesion_task_names = ['lesion_s1']
        # Grouped lesion result names (for saving/visualization)
        self.grouped_lesion_names = ['lesion_dr', 'lesion_amd']
        
        # No display mapping needed: internal names are final names
        
        # Lesion type mappings
        self.lesion_type_mappings = {
            'lesion_s1': {
                1: 'hemorrhage', 
                2: 'exudate', 
                3: 'cotton_wool_spot', 
                4: 'drusen',
                5: 'laser_spot'  # new class for task2 (lesion) 6-channel output
            },
        }

        # Additional task class mappings (non-lesion tasks)
        self.task_class_mappings = {
            'artery_vein': {
                1: 'artery',
                2: 'vein',
            },
            'od_oc': {
                1: 'optic_disc',
                2: 'optic_cup',
            },
            'tessellation': {
                1: 'tessellation_region',
            },
            'myopia': {
                1: 'arc_lesion',
                2: 'diffuse_chorioretinal_atrophy',
                3: 'patchy_chorioretinal_atrophy',
            },
        }

        # Grouped lesion type mappings for final saved outputs
        # Each group maps to 4-class outputs (0 background, 1..4 meaningful)
        self.grouped_lesion_type_mappings = {
            'lesion_dr': {
                1: 'hemorrhage',
                2: 'exudate',
                3: 'cotton_wool_spot',
                4: 'laser_spot',
            },
            'lesion_amd': {
                1: 'drusen',
                2: 'patch_hemorrhage',
            },
        }
        
        # Task output types
        self.task_output_types = {
            # AV head is now 2-channel multilabel (artery/vein), decoded with per-channel thresholds
            'artery_vein': 'multilabel',
            'od_oc': 'multiclass', 
            'lesion_s1': 'multiclass',
            'tessellation': 'multiclass',
            'myopia': 'multiclass',
            # Grouped lesion outputs
            'lesion_dr': 'multiclass',
            'lesion_amd': 'multiclass',
        }
        
        # Image processing settings
        self.image_size = 640
        self.analysis_size = 640
        self.normalization_mean = [0.42575, 0.29737, 0.21294]
        self.normalization_std = [0.2767, 0.2024, 0.1686]
        
        # Model settings
        self.default_output_channels = (2, 3, 2, 4, 6)
        self.model_params = {
            'in_channels': 3,
            'feature_size': 128,
            'depths': (2, 2, 18, 2),
            'num_heads': (4, 8, 16, 32),
            'window_size': 10,
            'patch_size': 4,
            'norm_name': 'instance',
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'dropout_path_rate': 0.0,
            'normalize': True,
            'use_checkpoint': False,
            'img_size': 640,  # Used for decoder kernel calculation
            'spatial_dims': 2,
            'downsample': 'merging',
            'use_v2': False
        }
        
        # Visualization settings
        self.visualization_colors = {
            'artery': [255, 0, 0],      # Red
            'vein': [0, 0, 255],        # Blue
            'disc': [0, 255, 0],        # Green
            'cup': [255, 255, 0],       # Yellow
            'lesion': [255, 0, 255],    # Magenta
            'tessellation': [0, 255, 255],  # Cyan
            'myopia': [255, 128, 0]     # Orange
        }

        # Overlay transparency (alpha) per task for visualizations
        # Higher value = more opaque overlay
        self.overlay_alpha_default: float = 0.6
        self.overlay_alpha_per_task = {
            'artery_vein': 0.6,
            'od_oc': 0.55,
            'lesion_s1': 0.6,
            'tessellation': 0.6,
            'myopia': 0.55,
            # Grouped lesion overlays
            'lesion_dr': 0.6,
            'lesion_amd': 0.6,
        }

        # Probability thresholding before clamping (to gate low-confidence predictions)
        # Applied as: create a binary mask from probabilities > threshold, and multiply after clamping
        self.prob_threshold_default: float = 0.7
        self.prob_threshold_per_task = {
            # Add per-task overrides here if needed, otherwise default is used
            'lesion_s1': 0.7,
            'artery_vein': 0.3,
            # Avoid gating out cup/disc; keep raw logits for accurate ISNT/CD
            'od_oc': 0.0,
            'tessellation': 0.7,
            'myopia': 0.5,
            # Grouped lesion outputs
            'lesion_dr': 0.7,
            'lesion_amd': 0.7,
        }

        # Minimum connected component size used during analysis counting
        # This should be aligned with post-processing in noise_filter to keep counts consistent with saved masks
        self.analysis_min_component_size_default: int = 5
        self.analysis_min_component_size_per_task = {
            'lesion_s1': 15,
            'artery_vein': 10,
            'od_oc': 50,
            'tessellation': 20,
            'myopia': 15,
            # Grouped lesion outputs
            'lesion_dr': 15,
            'lesion_amd': 15,
        }
        
        # Disease classification settings
        self.supported_diseases = ['dr', 'glaucoma', 'amd', 'myopia', 'cataract']
        self.disease_names = {
            'dr': 'Diabetic Retinopathy',
            'glaucoma': 'Glaucoma',
            'amd': 'Age-related Macular Degeneration',
            'myopia': 'Pathological Myopia',
            'cataract': 'Cataract'
        }
        
        # Disease grade descriptions
        self.disease_grade_descriptions = {
            'dr': {
                0: 'No retinopathy',
                1: 'Mild non-proliferative diabetic retinopathy',
                2: 'Moderate non-proliferative diabetic retinopathy',
                3: 'Severe non-proliferative diabetic retinopathy',
                4: 'Proliferative diabetic retinopathy'
            },
            'glaucoma': {
                0: 'Normal',
                1: 'Glaucoma suspect',
                2: 'Mild glaucoma',
                3: 'Moderate glaucoma',
                4: 'Severe glaucoma'
            },
            'amd': {
                0: 'Normal',
                1: 'Early AMD',
                2: 'Intermediate AMD',
                3: 'Late AMD'
            },
            'myopia': {
                0: 'Normal',
                1: 'Mild pathological myopia',
                2: 'Moderate pathological myopia (diffuse atrophy)',
                3: 'Severe pathological myopia (patchy atrophy)',
                4: 'Extreme pathological myopia (macular atrophy)'
            },
            'cataract': {
                0: 'No cataract',
                1: 'Suspected cataract'
            }
        }

        # Disease classification thresholds (tunable)
        self.classification_thresholds = {
            'dr': {
                'pdr_area_threshold': 0.03,
                'pdr_tri_count_threshold': 50,
                'per_quadrant_heavy_threshold': 20,
                'small_hemorrhage_area_px': 50,
            },
            'glaucoma': {
                'cdr_normal_upper': 0.5,
                'cdr_suspect_upper': 0.6,
                'cdr_mild_upper': 0.7,
                'cdr_moderate_upper': 0.8,
                'enforce_isnt_rule': True,
            },
            'amd': {
                'grade1_drusen_count': 50,
                'grade1_area_ratio': 0.02,
                'grade2_drusen_count': 100,
                'grade2_med_large_count': 20,
                'grade2_area_ratio': 0.05,
            },
            'myopia': {
                'tessellation_coverage_grade1': 0.30,
                'diffuse_area_threshold_px': 1000,
                'diffuse_ratio_threshold': 0.02,
                'patchy_area_threshold_px': 5000,
                'diffuse_min_count': 3,
            },
            'cataract': {
                'vessel_signal_min': 0.001,
                'min_failures_for_cataract': 3,
            },
        }
        
    def get_task_name(self, task_idx: int) -> str:
        """Get task name by index."""
        if task_idx < len(self.task_names):
            return self.task_names[task_idx]
        return f'task_{task_idx}'
        
    def get_display_name(self, task_name: str) -> str:
        """Return task_name directly (no mapping)."""
        return task_name
    
    def to_internal_task_name(self, name: str) -> str:
        """Return name directly (no reverse mapping)."""
        return name
        
    def get_lesion_types(self, task_name: str) -> Dict[int, str]:
        """Get lesion type mapping for a task."""
        return self.lesion_type_mappings.get(task_name, {})

    def get_grouped_lesion_types(self, group_name: str) -> Dict[int, str]:
        """Get lesion type mapping for a grouped lesion output."""
        return self.grouped_lesion_type_mappings.get(group_name, {})

    def get_task_class_mapping(self, task_name: str) -> Dict[int, str]:
        """
        Get class name mapping for a task (including non-lesion tasks).
        """
        if task_name in self.lesion_type_mappings:
            return self.lesion_type_mappings[task_name]
        if task_name in self.grouped_lesion_type_mappings:
            return self.grouped_lesion_type_mappings[task_name]
        return self.task_class_mappings.get(task_name, {})

    def get_grouped_lesion_names(self) -> List[str]:
        """Get list of final grouped lesion outputs to be saved."""
        return self.grouped_lesion_names.copy()
        
    def is_lesion_task(self, task_name: str) -> bool:
        """Check if task is a lesion-related task."""
        return task_name in self.lesion_task_names or task_name in self.grouped_lesion_names
        
    def get_output_type(self, task_name: str) -> str:
        """Get output type for a task."""
        return self.task_output_types.get(task_name, 'multiclass')

    def get_overlay_alpha(self, task_name: str) -> float:
        """Get overlay alpha for a given internal task name."""
        return self.overlay_alpha_per_task.get(task_name, self.overlay_alpha_default)

    def get_prob_threshold(self, task_name: str) -> float:
        """Get probability gating threshold for a given task."""
        return float(self.prob_threshold_per_task.get(task_name, self.prob_threshold_default))

    def get_analysis_min_component_size(self, task_name: str) -> int:
        """Get minimum connected component size for analysis stage counting for a given task."""
        return int(self.analysis_min_component_size_per_task.get(task_name, self.analysis_min_component_size_default))
        
    def get_supported_diseases(self) -> List[str]:
        """Get list of supported diseases for classification."""
        return self.supported_diseases.copy()
        
    def get_disease_name(self, disease_code: str) -> str:
        """Get Chinese name for disease."""
        return self.disease_names.get(disease_code, disease_code)
        
    def get_disease_grade_description(self, disease_code: str, grade: int) -> str:
        """Get description for disease grade."""
        if disease_code in self.disease_grade_descriptions:
            return self.disease_grade_descriptions[disease_code].get(grade, f'Unknown grade {grade}')
        return f'Unknown disease {disease_code} grade {grade}'

    def get_disease_thresholds(self, disease_code: str) -> dict:
        """Return adjustable classification thresholds for a disease."""
        return self.classification_thresholds.get(disease_code, {})
