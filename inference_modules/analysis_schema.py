"""Helpers for assembling quantitative analysis JSON schema."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple


@dataclass
class AnalyzerMeta:
    image_path: str
    original_shape: Tuple[int, int]
    analysis_size: int
    module_versions: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict:
        height, width = self.original_shape
        meta = {
            'image_id': os.path.basename(self.image_path) if self.image_path else None,
            'image_path': self.image_path,
            'width': int(width),
            'height': int(height),
            'analysis_resolution': {
                'width': self.analysis_size,
                'height': self.analysis_size,
                'unit': 'pixels'
            },
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'schema_version': '2.1.0',
        }
        if self.module_versions:
            meta['module_versions'] = self.module_versions
        return meta


class AnalyzerSummaryBuilder:
    """Build global summary fields."""

    @staticmethod
    def from_task_results(task_results: Dict[str, Dict]) -> Dict:
        total_area = 0.0
        total_coverage = 0.0

        for task_result in task_results.values():
            if not isinstance(task_result, dict):
                continue
            summary = task_result.get('summary')
            if isinstance(summary, dict):
                total_area += float(summary.get('total_area_pixels', 0) or 0)
                total_coverage += float(summary.get('total_coverage_percentage', 0) or 0)

        return {
            'total_pathological_area_px': int(total_area),
            'total_coverage_percentage': float(total_coverage),
            'tasks_analyzed': sorted(list(task_results.keys())),
        }


class AnalyzerResultAssembler:
    """Orchestrates creation of standardized quantitative JSON structure."""

    def __init__(self, config):
        self.config = config

    def assemble(self, task_results: Dict[str, Dict], image_path: str,
                 original_shape: Tuple[int, int], module_versions: Optional[Dict[str, str]] = None,
                 macula_center: Optional[Tuple[float, float]] = None) -> Dict:
        from .analysis_schema_builders import (
            VesselMeasurementBuilder,
            DiscMeasurementBuilder,
            LesionMeasurementBuilder,
            TessellationMeasurementBuilder,
            MyopiaMeasurementBuilder,
            MaculaMeasurementBuilder,
        )

        meta = AnalyzerMeta(
            image_path=image_path,
            original_shape=original_shape,
            analysis_size=self.config.analysis_size,
            module_versions=module_versions,
        ).to_dict()

        measurements = {}

        if 'artery_vein' in task_results:
            measurements['vessels'] = VesselMeasurementBuilder(self.config).build(task_results['artery_vein'])

        if 'od_oc' in task_results:
            measurements['optic_disc_cup'] = DiscMeasurementBuilder(self.config).build(
                task_results['od_oc'],
                macula_center,
                disc_reference=task_results.get('artery_vein'),
            )

        if macula_center is not None:
            macula_block = MaculaMeasurementBuilder(self.config).build(
                macula_center,
                task_results.get('od_oc'),
            )
            if macula_block:
                measurements['macula'] = macula_block

        grouped_lesions = {
            name: data
            for name, data in task_results.items()
            if name in self.config.get_grouped_lesion_names()
        }
        if grouped_lesions:
            measurements['lesions'] = LesionMeasurementBuilder(self.config).build(grouped_lesions)

        if 'tessellation' in task_results:
            measurements['tessellation'] = TessellationMeasurementBuilder(self.config).build(
                task_results['tessellation']
            )

        if 'myopia' in task_results:
            measurements['myopia'] = MyopiaMeasurementBuilder(self.config).build(task_results['myopia'])

        return {
            'meta': meta,
            'measurements': measurements,
        }
