"""Measurement builders for quantitative analysis schema."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


class VesselMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, vessel_analysis: Dict) -> Dict:
        if not isinstance(vessel_analysis, dict):
            return {}

        return {
            'av_ratio': vessel_analysis.get('av_ratio'),
            'CRAE': vessel_analysis.get('CRAE'),
            'CRVE': vessel_analysis.get('CRVE'),
            'fractal_dimension': {
                'artery': vessel_analysis.get('FDa'),
                'vein': vessel_analysis.get('FDv'),
            },
            'tortuosity': {
                'artery': vessel_analysis.get('artery_tortuosity'),
                'vein': vessel_analysis.get('vein_tortuosity'),
            },
            'pixel_length': vessel_analysis.get('pixel_len'),
            'qc_flag': vessel_analysis.get('qc_flag'),
            'fg_rate': vessel_analysis.get('fg_rate'),
        }

    @staticmethod
    def _scalar(value, unit: Optional[str] = None) -> Optional[Dict]:
        if isinstance(value, (int, float)):
            block = {'value': float(value)}
            if unit:
                block['unit'] = unit
            return block
        return None

    @staticmethod
    def _extract_av_ratio(vessel_analysis: Dict):
        value = vessel_analysis.get('av_ratio')
        if value is None:
            value = vessel_analysis.get('av_ratio_upper')
        if value is None:
            value = vessel_analysis.get('av_ratio_lower')
        if isinstance(value, (list, tuple)) and value:
            # take average if two halves provided
            numeric = [float(v) for v in value if isinstance(v, (int, float))]
            if numeric:
                return sum(numeric) / len(numeric)
        return value

    @staticmethod
    def _extract_sequence(vessel_analysis: Dict, primary_key: str, fallback_key: Optional[str] = None):
        value = vessel_analysis.get(primary_key)
        if value is None and fallback_key:
            value = vessel_analysis.get(fallback_key)
        return value

    def _sequence(self, value) -> Optional[Dict]:
        if isinstance(value, (list, tuple)) and value:
            sanitized = [float(v) for v in value if isinstance(v, (int, float))]
            if not sanitized:
                return None
            return {'value': sanitized, 'unit': 'pixels'}
        return None

    def _fractal(self, vessel_analysis: Dict) -> Optional[Dict]:
        artery = vessel_analysis.get('FDa')
        vein = vessel_analysis.get('FDv')
        out = {}
        if isinstance(artery, (int, float)):
            out['artery'] = float(artery)
        if isinstance(vein, (int, float)):
            out['vein'] = float(vein)
        return out or None

    def _tortuosity(self, vessel_analysis: Dict) -> Optional[Dict]:
        artery = vessel_analysis.get('artery_tortuosity')
        vein = vessel_analysis.get('vein_tortuosity')
        out = {}
        if isinstance(artery, (int, float)):
            out['artery'] = float(artery)
        if isinstance(vein, (int, float)):
            out['vein'] = float(vein)
        return out or None

    def _disc_reference(self, vessel_analysis: Dict) -> Optional[Dict]:
        center = vessel_analysis.get('disc_center')
        radius = vessel_analysis.get('disc_radius')
        disc = {}
        if isinstance(center, (list, tuple)) and len(center) == 2:
            disc['center'] = {
                'x': float(center[0]),
                'y': float(center[1]),
                'unit': 'pixels'
            }
        if isinstance(radius, (int, float)):
            disc['radius'] = {'value': float(radius), 'unit': 'pixels'}
        return disc or None

    @staticmethod
    def _qc(vessel_analysis: Dict) -> Dict:
        return {
            'pass': bool(vessel_analysis.get('qc_flag', True)),
            'foreground_rate': float(vessel_analysis.get('fg_rate', 0.0)),
            'confidence': float(vessel_analysis.get('qc_confidence')) if 'qc_confidence' in vessel_analysis else None
        }


class DiscMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, disc_analysis: Dict, macula_center: Optional[Tuple[float, float]],
              disc_reference: Optional[Dict] = None) -> Dict:
        if not isinstance(disc_analysis, dict):
            return {}

        isnt = disc_analysis.get('isnt_list') or disc_analysis.get('isnt_values')
        isnt_dict = None
        if isinstance(isnt, (list, tuple)) and len(isnt) == 4:
            # Elements may already be scalars or nested lists; extract first scalar
            def _as_scalar(val):
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                try:
                    return float(val)
                except Exception:
                    return 0.0

            isnt_dict = {
                'inferior': _as_scalar(isnt[0]),
                'superior': _as_scalar(isnt[1]),
                'nasal': _as_scalar(isnt[2]),
                'temporal': _as_scalar(isnt[3])
            }
        elif isinstance(isnt, dict):
            # Accept already keyed dicts
            isnt_dict = {
                'inferior': float(isnt.get('inferior', 0) or 0),
                'superior': float(isnt.get('superior', 0) or 0),
                'nasal': float(isnt.get('nasal', 0) or 0),
                'temporal': float(isnt.get('temporal', 0) or 0),
            }

        eye_side_val = disc_analysis.get('eyeside_detected')
        eye_side = None
        if isinstance(eye_side_val, int):
            eye_side = 'left' if eye_side_val == 0 else 'right'

        # Cup/disc diameters from cd_list: [hori_cd, vert_cd, hori_dd, vert_dd]
        cd_list = disc_analysis.get('cd_list')
        cd_list_out = None
        if isinstance(cd_list, (list, tuple)) and len(cd_list) >= 4:
            try:
                cd_list_out = [
                    float(cd_list[0]),
                    float(cd_list[1]),
                    float(cd_list[2]),
                    float(cd_list[3]),
                ]
            except Exception:
                cd_list_out = None

        data = {
            'cup_disc_ratio': self._scalar(disc_analysis.get('cd_ratio')),
            'vertical_cd_ratio': self._scalar(disc_analysis.get('vh_ratio')),
            'axis_angle': self._scalar(self._parse_angle(disc_analysis.get('angle_str')), unit='deg'),
            'isnt': isnt_dict,
            'eye_side': eye_side,
            'qc': {'pass': True},
        }

        if cd_list_out:
            data['cd_list'] = cd_list_out

        # areas in pixels if provided
        if isinstance(disc_analysis.get('disc_area_px'), (int, float)):
            data['disc_area_px'] = int(disc_analysis['disc_area_px'])
        if isinstance(disc_analysis.get('cup_area_px'), (int, float)):
            data['cup_area_px'] = int(disc_analysis['cup_area_px'])

        disc_center = disc_analysis.get('disc_center') if isinstance(disc_analysis, dict) else None
        disc_radius = disc_analysis.get('disc_radius') if isinstance(disc_analysis, dict) else None
        if disc_reference and isinstance(disc_reference, dict):
            disc_center = disc_center or disc_reference.get('disc_center')
            disc_radius = disc_radius if disc_radius is not None else disc_reference.get('disc_radius')

        disc_block = {}
        if isinstance(disc_center, (list, tuple)) and len(disc_center) == 2:
            disc_block['center'] = [float(disc_center[0]), float(disc_center[1])]
        if isinstance(disc_radius, (int, float)):
            disc_block['radius'] = float(disc_radius)
        if disc_block:
            data['disc'] = disc_block

        return data

    @staticmethod
    def _scalar(value, unit: Optional[str] = None) -> Optional[Dict]:
        if isinstance(value, (int, float)):
            val = float(value)
        elif isinstance(value, str):
            try:
                val = float(value)
            except ValueError:
                return None
        else:
            return None
        block = {'value': val}
        if unit:
            block['unit'] = unit
        return block

    @staticmethod
    def _parse_angle(angle_str) -> Optional[float]:
        if isinstance(angle_str, (int, float)):
            return float(angle_str)
        if isinstance(angle_str, str):
            try:
                return float(angle_str)
            except ValueError:
                return None
        return None


class LesionMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, lesion_results: Dict[str, Dict]) -> Dict:
        grouped = {}
        for group_name, data in lesion_results.items():
            grouped[group_name] = self._group_block(group_name, data)
        return grouped

    def _group_block(self, group_name: str, data: Dict) -> Dict:
        summary = data.get('summary', {}) if isinstance(data, dict) else {}
        lesion_analysis = data.get('lesion_categories_analysis', {}) if isinstance(data, dict) else {}
        categories_overview = lesion_analysis.get('categories_overview', {})
        detailed_analysis = lesion_analysis.get('detailed_analysis', {})
        # Ensure every expected lesion category is present even if analysis found none
        expected_categories_map = self.config.get_grouped_lesion_types(group_name)
        category_names = []
        if isinstance(expected_categories_map, dict) and expected_categories_map:
            category_names.extend(expected_categories_map.values())
        # Append any additional categories produced at runtime
        for lesion_name in categories_overview.keys():
            if lesion_name not in category_names:
                category_names.append(lesion_name)
        # If still empty, nothing to report
        if not category_names:
            category_names = list(categories_overview.keys())
        
        categories = {}
        total_small = total_medium = total_large = 0
        for lesion_name in category_names:
            overview = categories_overview.get(lesion_name, {}) if isinstance(categories_overview, dict) else {}
            details = detailed_analysis.get(lesion_name, {}) if isinstance(detailed_analysis, dict) else {}
            measurements = details.get('measurements', {}) if isinstance(details, dict) else {}
            shape = details.get('shape_characteristics', {}) if isinstance(details, dict) else {}
            quadrant = details.get('quadrant_distribution', {}) if isinstance(details, dict) else {}
            counts = quadrant.get('counts', {}) if isinstance(quadrant, dict) else {}
            size_dist = details.get('size_distribution', {}) if isinstance(details, dict) else {}
            small_c = int(size_dist.get('small_lesions', 0))
            medium_c = int(size_dist.get('medium_lesions', 0))
            large_c = int(size_dist.get('large_lesions', 0))
            total_small += small_c
            total_medium += medium_c
            total_large += large_c
            categories[lesion_name] = {
                'count': int(measurements.get('count', 0)),
                'area_px': int(measurements.get('total_area_pixels', 0)),
                'coverage_ratio': float(measurements.get('coverage_percentage', 0.0)),
                'avg_size_px': float(measurements.get('average_size_pixels', 0.0)) if measurements.get('count') else None,
                'size_distribution': {
                    'small': small_c,
                    'medium': medium_c,
                    'large': large_c,
                },
                'shape': {
                    'circularity': float(shape.get('circularity', 0.0)) if shape else None,
                    'aspect_ratio': float(shape.get('aspect_ratio', 0.0)) if shape else None,
                },
                'quadrant_distribution': {
                    'definition': 'macula-centered',
                    'superior_temporal': int(counts.get('superior_temporal', 0)),
                    'superior_nasal': int(counts.get('superior_nasal', 0)),
                    'inferior_temporal': int(counts.get('inferior_temporal', 0)),
                    'inferior_nasal': int(counts.get('inferior_nasal', 0)),
                },
                'severity': overview.get('severity') if isinstance(overview, dict) else None,
            }

        coverage_ratio = float(summary.get('total_coverage_percentage', 0.0))
        return {
            'summary': {
                'category_count': int(summary.get('total_lesion_categories', len(categories))),
                'total_area_px': int(summary.get('total_area_pixels', 0)),
                'coverage_ratio': coverage_ratio,
                'size_buckets': {
                    'small': int(total_small),
                    'medium': int(total_medium),
                    'large': int(total_large),
                },
            },
            'categories': categories,
        }


class TessellationMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, tessellation_result: Dict) -> Dict:
        if not isinstance(tessellation_result, dict):
            return {}

        coverage_ratio = tessellation_result.get('tessellate_ratio')
        summary = tessellation_result.get('summary')
        if isinstance(summary, dict):
            coverage_ratio = summary.get('tessellation_coverage_ratio', coverage_ratio)

        return {
            'overall': {
                'total_area_px': int(tessellation_result.get('total_tessellate_area', 0)),
                'coverage_ratio': float(coverage_ratio or 0.0),
            },
            'shape_stats': {
                'mean_circularity': float(tessellation_result.get('mean_circularity', 0.0)),
                'mean_aspect_ratio': float(tessellation_result.get('mean_aspect_ratio', 0.0)),
            },
            'centroid_dispersion': self._scalar(tessellation_result.get('centroid_dispersion'), 'pixels'),
        }

    @staticmethod
    def _scalar(value, unit: Optional[str] = None) -> Optional[Dict]:
        if isinstance(value, (int, float)):
            block = {'value': float(value)}
            if unit:
                block['unit'] = unit
            return block
        return None


class MyopiaMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, myopia_result: Dict) -> Dict:
        if not isinstance(myopia_result, dict):
            return {}

        summary = myopia_result.get('overall') if isinstance(myopia_result.get('overall'), dict) else {}
        categories = myopia_result.get('categories') if isinstance(myopia_result.get('categories'), dict) else {}

        return {
            'overall': {
                'count': int(summary.get('count', 0)),
                'total_area_px': int(summary.get('total_area', summary.get('total_area_px', 0))),
                'coverage_ratio': float(summary.get('coverage_ratio', 0.0)),
            },
            'categories': {
                name: {
                    'count': int(info.get('count', 0)),
                    'area_px': int(info.get('area_px', info.get('total_area_pixels', 0))),
                    'coverage_ratio': float(info.get('coverage_ratio', info.get('coverage_percentage', 0.0))),
                }
                for name, info in categories.items()
            },
        }


class MaculaMeasurementBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, macula_center, od_oc_result: Optional[Dict]) -> Dict:
        if macula_center is None:
            return {}

        macula_block = {
            'center': {
                'x': float(macula_center[1]),
                'y': float(macula_center[0]),
                'unit': 'pixels'
            }
        }

        if isinstance(od_oc_result, dict):
            macula_block['context'] = {
                'eye_side': 'left' if od_oc_result.get('eyeside_detected') == 0 else 'right' if od_oc_result.get('eyeside_detected') == 1 else None,
            }

        return macula_block
