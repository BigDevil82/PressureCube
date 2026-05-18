import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.structure.enums import BeamType
from src.structure.layout.base import Zone
from src.structure.layout.params import EngineeringLayoutParameters


@dataclass
class Opening:
    face: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    sealed: bool = True

    def contains(self, x: float, y: float, z: float, tol: float = 1e-4) -> bool:
        if self.face in {"front", "back"}:
            face_hit = (
                math.isclose(x, self.x_range[0], abs_tol=tol) or math.isclose(x, self.x_range[1], abs_tol=tol)
            )
            return (
                face_hit
                and self.y_range[0] + tol < y < self.y_range[1] - tol
                and self.z_range[0] + tol < z < self.z_range[1] - tol
            )

        if self.face in {"left", "right"}:
            face_hit = (
                math.isclose(y, self.y_range[0], abs_tol=tol) or math.isclose(y, self.y_range[1], abs_tol=tol)
            )
            return (
                face_hit
                and self.x_range[0] + tol < x < self.x_range[1] - tol
                and self.z_range[0] + tol < z < self.z_range[1] - tol
            )

        return False


class EngineeringLayoutEngine:
    """工程版布局引擎。"""

    def __init__(self, params: EngineeringLayoutParameters):
        self.p = params
        self.zones: list[Zone] = []
        self.main_zone: Optional[Zone] = None

        self.grids_x = set()
        self.grids_y = set()
        self.grids_z = set()

        self.secondary_x_locs = set()
        self.secondary_z_locs = set()
        self.brace_x_locs = set()
        self.brace_z_locs = set()

        self.front_back_window_spans: set[tuple[float, float]] = set()
        self.right_window_spans: set[tuple[float, float]] = set()
        self.front_back_window_side_y: set[float] = set()
        self.right_window_side_x: set[float] = set()
        self.ext_left_door_side_x: set[float] = set()
        self.window_anchor_zs: set[float] = set()
        self.window_expanded_brace_zs: set[float] = set()
        self.window_vertical_ranges: list[tuple[float, float]] = []
        self.openings: list[Opening] = []

        self._init_geometry()

    def _init_geometry(self):
        L, W, H = self.p.length, self.p.width, self.p.height
        ext = self.p.extension

        self.main_zone = Zone("main", (0, W), (0, L), (0, H))
        self.zones.append(self.main_zone)

        if ext.active:
            ext_x_start = ext.offset_x
            ext_x_end = ext.offset_x + ext.width
            self.zones.append(Zone("ext", (ext_x_start, ext_x_end), (-ext.length, 0), (0, ext.height)))
            self.zones.append(
                Zone("equip", (0, ext_x_start), (-ext.length, 0), (0, ext.height), is_hollow=True)
            )

        for zone in self.zones:
            self.grids_x.update(zone.x_range)
            self.grids_y.update(zone.y_range)
            self.grids_z.update(zone.z_range)

        self._fill_main_beam_y_grids()
        self._fill_secondary_x_grids_centered()
        self._fill_secondary_z_grids_with_windows()
        self._build_openings()
        self._fill_front_back_window_support_y_grids()

        self.sorted_x = sorted(self.grids_x)
        self.sorted_y = sorted(self.grids_y)
        self.sorted_z = sorted(self.grids_z)

    def _fill_main_beam_y_grids(self):
        curr = 0.0
        while curr <= self.p.length + 1e-5:
            self.grids_y.add(curr)
            curr += self.p.main_beam_spacing
        self.grids_y.add(self.p.length)

        if self.p.extension.active:
            self.grids_y.add(-self.p.extension.length)

    def _center_out_positions(self, total: float, spacing: float, min_edge_remainder: float) -> list[float]:
        if spacing <= 0 or total <= 0:
            return []

        center = total / 2
        positions = [center]
        step_index = 1

        while True:
            left = center - step_index * spacing
            right = center + step_index * spacing
            added = False

            if left > 0:
                positions.append(left)
                added = True
            if right < total:
                positions.append(right)
                added = True

            if not added:
                break
            step_index += 1

        positions = sorted(set(positions))

        while positions:
            left_gap = positions[0]
            right_gap = total - positions[-1]
            if left_gap >= min_edge_remainder and right_gap >= min_edge_remainder:
                break

            if len(positions) == 1:
                positions = []
                break

            positions = positions[1:-1]

        return positions

    def _support_points_in_span(self, start: float, end: float, max_spacing: float) -> list[float]:
        if max_spacing <= 0:
            return []

        span = end - start
        if span <= max_spacing:
            return []

        division_count = int(math.ceil(span / max_spacing - 1e-9))
        step = span / division_count
        return [start + i * step for i in range(1, division_count)]

    def _insert_support_points(self, bounds: list[float], max_spacing: float, target_set: set[float]):
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            target_set.update(self._support_points_in_span(start, end, max_spacing))

    def _main_face_spans(self) -> list[tuple[float, float]]:
        ys = sorted(y for y in self.grids_y if y >= 0)
        return [(ys[i], ys[i + 1]) for i in range(len(ys) - 1)]

    def _safe_span(self, spans: list[tuple[float, float]], idx: int) -> Optional[tuple[float, float]]:
        if idx < 0 or idx >= len(spans):
            return None
        return spans[idx]

    def _register_front_back_window_spans(self):
        window = self.p.window
        if not window.active:
            return

        spans = self._main_face_spans()
        for idx in [*window.front_spans, *window.back_spans]:
            span = self._safe_span(spans, idx)
            if span is not None:
                self.front_back_window_spans.add(span)

    def _fill_secondary_x_grids_centered(self):
        cfg = self.p
        secondary_x = self._center_out_positions(self.p.width, self.p.main_beam_spacing, cfg.min_edge_remainder)

        for x in secondary_x:
            self.grids_x.add(x)
            self.secondary_x_locs.add(x)

        if self.p.add_support_beam:
            bounds = [0.0] + sorted(self.secondary_x_locs) + [self.p.width]
            self._insert_support_points(bounds, cfg.support_beam_max_spacing, self.brace_x_locs)
            self.grids_x.update(self.brace_x_locs)

    def _collect_vertical_anchor_levels(self) -> list[float]:
        anchors: set[float] = set()
        window = self.p.window

        self._register_front_back_window_spans()

        if window.active and (self.front_back_window_spans or window.right_spans):
            anchors.add(window.sill_height)
            anchors.add(window.sill_height + window.height)

        return sorted(z for z in anchors if 0 < z < self.p.height)

    def _fill_secondary_z_grids_with_windows(self):
        anchors = self._collect_vertical_anchor_levels()
        if anchors:
            z_vals = set(anchors)
        else:
            z_vals = set(
                self._center_out_positions(self.p.height, self.p.main_beam_spacing, self.p.min_edge_remainder)
            )

        for z in z_vals:
            self.grids_z.add(z)
            self.secondary_z_locs.add(z)

        if self.p.add_support_beam:
            bounds = [0.0] + sorted(self.secondary_z_locs) + [self.p.height]
            self._insert_support_points(bounds, self.p.support_beam_max_spacing, self.brace_z_locs)
            self.grids_z.update(self.brace_z_locs)

    def _spans_for_indices(self, indices: list[int]) -> list[tuple[float, float]]:
        spans = self._main_face_spans()
        return [span for idx in indices if (span := self._safe_span(spans, idx)) is not None]

    def _fill_front_back_window_support_y_grids(self):
        if not self.p.add_support_beam or not self.p.window.active:
            return

        for indices in (self.p.window.front_spans, self.p.window.back_spans):
            window_spans = sorted(self._spans_for_indices(indices))
            for left, right in zip(window_spans, window_spans[1:]):
                bounds = sorted(y for y in self.grids_y if left[1] <= y <= right[0])
                support_y: set[float] = set()
                self._insert_support_points(bounds, self.p.support_beam_max_spacing, support_y)
                self.grids_y.update(support_y)

    def _resolve_right_window_spans(self) -> list[tuple[float, float]]:
        spans = []
        x_lines = sorted(self.secondary_x_locs)
        available_spans = [(0.0, x_lines[0])] if x_lines else []
        available_spans.extend((x_lines[i], x_lines[i + 1]) for i in range(len(x_lines) - 1))
        if x_lines:
            available_spans.append((x_lines[-1], self.p.width))
        else:
            available_spans.append((0.0, self.p.width))

        for idx in self.p.window.right_spans:
            span = self._safe_span(available_spans, idx)
            if span is not None:
                spans.append(span)
                self.right_window_spans.add(span)
                self.grids_x.update(span)
        return spans

    def _build_openings(self):
        self._build_ext_overlap_opening()
        self._build_window_openings()
        self._build_door_opening()

    def _build_ext_overlap_opening(self):
        ext = self.p.extension
        if not ext.active:
            return

        self.openings.append(
            Opening("right", (ext.offset_x, self.p.width), (0.0, 0.0), (0.0, self.p.height), sealed=False)
        )

    def _build_window_openings(self):
        window = self.p.window
        if not window.active:
            return

        z0 = window.sill_height
        z1 = window.sill_height + window.height
        if z0 >= z1:
            return
        self.window_anchor_zs.update({z0, z1})
        self.window_vertical_ranges.append((z0, z1))

        spans = self._main_face_spans()

        for idx in window.front_spans:
            span = self._safe_span(spans, idx)
            if span is not None:
                self.front_back_window_side_y.add((self.p.width, span[0]))
                self.front_back_window_side_y.add((self.p.width, span[1]))
                self.openings.append(Opening("front", (self.p.width, self.p.width), span, (z0, z1)))

        for idx in window.back_spans:
            span = self._safe_span(spans, idx)
            if span is not None:
                self.front_back_window_side_y.add((0.0, span[0]))
                self.front_back_window_side_y.add((0.0, span[1]))
                self.openings.append(Opening("back", (0.0, 0.0), span, (z0, z1)))

        for span in self._resolve_right_window_spans():
            self.right_window_side_x.add((self.p.length, span[0]))
            self.right_window_side_x.add((self.p.length, span[1]))
            self.openings.append(Opening("right", span, (self.p.length, self.p.length), (z0, z1)))

    def _build_door_opening(self):
        door = self.p.door
        if not door.active:
            return

        ext = self.p.extension
        if not ext.active or door.face != "left":
            return

        x1 = self.p.width
        x0 = max(ext.offset_x, x1 - door.width)
        z0 = 0.0
        z1 = min(self.p.height, self.p.window.sill_height + self.p.window.height)

        if x1 <= x0 or z1 <= z0:
            return

        self.grids_x.update({x0, x1})
        self.grids_z.update({z0, z1})
        self.ext_left_door_side_x.update({x0, x1})
        self.secondary_z_locs.update(z for z in (z0, z1) if 0 < z < self.p.height)
        self.openings.append(Opening("left", (x0, x1), (-ext.length, -ext.length), (z0, z1)))

    def _is_secondary_x(self, x: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(x, value, abs_tol=tol) for value in self.secondary_x_locs)

    def _is_secondary_z(self, z: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(z, value, abs_tol=tol) for value in self.secondary_z_locs)

    def _is_brace_x(self, x: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(x, value, abs_tol=tol) for value in self.brace_x_locs)

    def _is_brace_z(self, z: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(z, value, abs_tol=tol) for value in self.brace_z_locs)

    def _is_window_side_y(self, x: float, y: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(x, px, abs_tol=tol) and math.isclose(y, py, abs_tol=tol) for px, py in self.front_back_window_side_y)

    def _is_window_side_x(self, x: float, y: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(y, py, abs_tol=tol) and math.isclose(x, px, abs_tol=tol) for py, px in self.right_window_side_x)

    def _is_window_mid_height(self, z: float, tol: float = 1e-4) -> bool:
        return any(z0 + tol < z < z1 - tol for z0, z1 in self.window_vertical_ranges)

    def _is_window_expanded_brace_z(self, z: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(z, value, abs_tol=tol) for value in self.window_expanded_brace_zs)

    def _is_ext_left_face(self, y: float, tol: float = 1e-4) -> bool:
        return self.p.extension.active and math.isclose(y, -self.p.extension.length, abs_tol=tol)

    def _is_ext_door_side_x(self, x: float, tol: float = 1e-4) -> bool:
        return any(math.isclose(x, value, abs_tol=tol) for value in self.ext_left_door_side_x)

    def _is_main_left_overlap_face(self, x: float, y: float, tol: float = 1e-4) -> bool:
        return self.p.extension.active and math.isclose(y, 0.0, abs_tol=tol) and x < self.p.extension.offset_x - tol

    def is_in_opening(self, x, y, z, tol=1e-4) -> bool:
        return any(opening.contains(x, y, z, tol) for opening in self.openings)

    def is_in_sealed_opening(self, x, y, z, tol=1e-4) -> bool:
        return any(opening.sealed and opening.contains(x, y, z, tol) for opening in self.openings)

    def determine_beam(
        self, p_start: np.ndarray, p_end: np.ndarray, axis: str
    ) -> tuple[Optional[BeamType], str, bool]:
        mid = (p_start + p_end) / 2
        x, y, z = mid

        active_zones = [zone for zone in self.zones if zone.contains(x, y, z)]
        if not active_zones:
            return None, "", False

        if self.is_in_opening(x, y, z):
            return None, "", False

        is_on_any_surface = any(zone.on_surface(x, y, z) for zone in active_zones)
        if not is_on_any_surface:
            return None, "", False

        face_name = active_zones[0].get_face(x, y, z)
        is_frame_edge = any(zone.on_edge(x, y, z) for zone in active_zones)

        in_hollow = all(zone.is_hollow for zone in active_zones)
        if in_hollow:
            return (BeamType.FRAME, face_name, False) if is_frame_edge else (None, "", False)

        if is_frame_edge:
            return BeamType.FRAME, face_name, False

        proposed_type = None
        is_virtual = False

        if axis == "x":
            if math.isclose(z, 0, abs_tol=1e-4) or math.isclose(z, self.p.height, abs_tol=1e-4):
                proposed_type = BeamType.MAIN
            else:
                if self._is_window_mid_height(z) or self._is_window_expanded_brace_z(z):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.BRACE if self._is_brace_z(z) else BeamType.SECONDARY

        elif axis == "y":
            if math.isclose(z, 0, abs_tol=1e-4) or math.isclose(z, self.p.height, abs_tol=1e-4):
                if self._is_brace_x(x):
                    proposed_type = BeamType.BRACE
                elif self._is_secondary_x(x):
                    proposed_type = BeamType.SECONDARY
            else:
                if self._is_window_mid_height(z) or self._is_window_expanded_brace_z(z):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.BRACE if self._is_brace_z(z) else BeamType.SECONDARY

        elif axis == "z":
            if math.isclose(x, self.p.width, abs_tol=1e-4) or math.isclose(x, 0.0, abs_tol=1e-4):
                if self._is_window_side_y(x, y):
                    proposed_type = BeamType.FRAME
                elif self._is_secondary_z(z):
                    proposed_type = BeamType.SECONDARY
                elif self._is_brace_z(z):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.MAIN
            elif self._is_ext_left_face(y):
                if self._is_ext_door_side_x(x):
                    proposed_type = BeamType.FRAME
                else:
                    return None, "", False
            elif self._is_main_left_overlap_face(x, y):
                if self._is_brace_x(x):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.MAIN
            elif math.isclose(y, self.p.length, abs_tol=1e-4) or math.isclose(y, 0.0, abs_tol=1e-4):
                if self._is_window_side_x(x, y):
                    proposed_type = BeamType.FRAME
                elif self._is_brace_x(x):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.MAIN
                    is_virtual = True
            else:
                proposed_type = BeamType.MAIN

        if proposed_type is not None:
            return proposed_type, face_name, is_virtual
        return None, "", False
