import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.structure.enums import BeamType
from src.structure.layout.base import Zone
from structure.layout.params import StructuralParameters


@dataclass
class Opening:
    face: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]

    def contains(self, x: float, y: float, z: float, tol: float = 1e-4) -> bool:
        return (
            self.x_range[0] - tol <= x <= self.x_range[1] + tol
            and self.y_range[0] - tol <= y <= self.y_range[1] + tol
            and self.z_range[0] - tol <= z <= self.z_range[1] + tol
        )


class EngineeringLayoutEngine:
    """Engineering-oriented layout engine (initial implementation)."""

    def __init__(self, params: StructuralParameters):
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

        for z in self.zones:
            self.grids_x.update(z.x_range)
            self.grids_y.update(z.y_range)
            self.grids_z.update(z.z_range)

        self._fill_main_beam_y_grids()
        self._fill_secondary_x_grids_centered()
        self._fill_secondary_z_grids_with_windows()
        self._build_window_openings()

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

        c = total / 2
        pos = [c]
        k = 1

        while True:
            left = c - k * spacing
            right = c + k * spacing
            added = False

            if left > 0:
                pos.append(left)
                added = True
            if right < total:
                pos.append(right)
                added = True

            if not added:
                break
            k += 1

        pos = sorted(set(pos))

        while len(pos) > 1:
            rem_left = pos[0]
            rem_right = total - pos[-1]
            if rem_left >= min_edge_remainder and rem_right >= min_edge_remainder:
                break
            pos = pos[1:-1]

        return pos

    def _insert_support_points(self, bounds: list[float], max_spacing: float, target_set: set):
        if max_spacing <= 0:
            return

        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            seg = b - a
            if seg <= max_spacing:
                continue
            n = int(math.ceil(seg / max_spacing))
            step = seg / n
            for j in range(1, n):
                target_set.add(a + j * step)

    def _fill_secondary_x_grids_centered(self):
        cfg = self.p.engineering
        sec = self._center_out_positions(self.p.width, self.p.main_beam_spacing, cfg.min_edge_remainder)

        for x in sec:
            self.grids_x.add(x)
            self.secondary_x_locs.add(x)

        if self.p.add_support_beam:
            bounds = [0.0] + sorted(self.secondary_x_locs) + [self.p.width]
            self._insert_support_points(bounds, cfg.support_beam_max_spacing, self.brace_x_locs)
            self.grids_x.update(self.brace_x_locs)

    def _fill_secondary_z_grids_with_windows(self):
        cfg = self.p.engineering
        w = cfg.window

        has_window = w.active and (w.front_spans or w.back_spans or w.right_spans)
        if has_window:
            anchors = [w.sill_height, w.sill_height + w.height]
            anchors = [z for z in anchors if 0 < z < self.p.height]
        else:
            anchors = self._center_out_positions(
                self.p.height, self.p.main_beam_spacing, cfg.min_edge_remainder
            )

        z_vals = set(anchors)
        spacing = self.p.main_beam_spacing

        for z0 in anchors:
            k = 1
            while z0 - k * spacing > 0:
                z_vals.add(z0 - k * spacing)
                k += 1
            if z0 - (k - 1) * spacing < cfg.min_edge_remainder and k > 1:
                z_vals.discard(z0 - (k - 1) * spacing)

            k = 1
            while z0 + k * spacing < self.p.height:
                z_vals.add(z0 + k * spacing)
                k += 1
            last = z0 + (k - 1) * spacing
            if self.p.height - last < cfg.min_edge_remainder and k > 1:
                z_vals.discard(last)

        for z in z_vals:
            self.grids_z.add(z)
            self.secondary_z_locs.add(z)

        if self.p.add_support_beam:
            bounds = [0.0] + sorted(self.secondary_z_locs) + [self.p.height]
            self._insert_support_points(bounds, cfg.support_beam_max_spacing, self.brace_z_locs)
            self.grids_z.update(self.brace_z_locs)

    def _build_window_openings(self):
        cfg = self.p.engineering
        w = cfg.window
        if not w.active:
            return

        z0 = w.sill_height
        z1 = w.sill_height + w.height
        if z0 >= z1:
            return

        y_spans = self._main_face_spans()

        for idx in w.front_spans:
            span = self._safe_span(y_spans, idx)
            if span is None:
                continue
            self.openings.append(Opening("front", (self.p.width, self.p.width), span, (z0, z1)))

        for idx in w.back_spans:
            span = self._safe_span(y_spans, idx)
            if span is None:
                continue
            self.openings.append(Opening("back", (0.0, 0.0), span, (z0, z1)))

    def _main_face_spans(self) -> list[tuple[float, float]]:
        ys = sorted(y for y in self.grids_y if y >= 0)
        spans = []
        for i in range(len(ys) - 1):
            spans.append((ys[i], ys[i + 1]))
        return spans

    def _safe_span(self, spans: list[tuple[float, float]], idx: int) -> Optional[tuple[float, float]]:
        if idx < 0 or idx >= len(spans):
            return None
        return spans[idx]

    def is_in_opening(self, x, y, z, tol=1e-4) -> bool:
        ext = self.p.extension
        if ext.active and math.isclose(y, 0, abs_tol=tol):
            in_x = (x > ext.offset_x + tol) and (x < self.p.width - tol)
            in_z = (z > 0 + tol) and (z < ext.height - tol)
            if in_x and in_z:
                return True

        if not self.openings:
            return False

        for op in self.openings:
            if op.contains(x, y, z, tol):
                return True
        return False

    def determine_beam(
        self, p_start: np.ndarray, p_end: np.ndarray, axis: str
    ) -> tuple[Optional[BeamType], str, bool]:
        mid = (p_start + p_end) / 2
        x, y, z = mid

        active_zones = [zn for zn in self.zones if zn.contains(x, y, z)]
        if not active_zones:
            return None, "", False

        if self.is_in_opening(x, y, z):
            return None, "", False

        is_on_any_surface = any(zn.on_surface(x, y, z) for zn in active_zones)
        if not is_on_any_surface:
            return None, "", False

        face_name = active_zones[0].get_face(x, y, z)
        is_frame_edge = any(zn.on_edge(x, y, z) for zn in active_zones)

        in_hollow = all(zn.is_hollow for zn in active_zones)
        if in_hollow:
            return (BeamType.FRAME, face_name, False) if is_frame_edge else (None, "", False)

        if is_frame_edge:
            return BeamType.FRAME, face_name, False

        proposed_type = None
        is_virtual = False

        if axis == "x":
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                proposed_type = BeamType.MAIN
            else:
                proposed_type = (
                    BeamType.BRACE
                    if any(math.isclose(z, bz) for bz in self.brace_z_locs)
                    else BeamType.SECONDARY
                )

        elif axis == "y":
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                if any(math.isclose(x, bx) for bx in self.brace_x_locs):
                    proposed_type = BeamType.BRACE
                elif any(math.isclose(x, sx) for sx in self.secondary_x_locs):
                    proposed_type = BeamType.SECONDARY
            else:
                proposed_type = (
                    BeamType.BRACE
                    if any(math.isclose(z, bz) for bz in self.brace_z_locs)
                    else BeamType.SECONDARY
                )

        elif axis == "z":
            if math.isclose(y, self.p.length):
                if any(math.isclose(x, sx) for sx in self.secondary_x_locs):
                    proposed_type = BeamType.FRAME
                elif any(math.isclose(x, bx) for bx in self.brace_x_locs):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.MAIN
                    is_virtual = True
            else:
                proposed_type = BeamType.MAIN

        if proposed_type is not None:
            return proposed_type, face_name, is_virtual
        return None, "", False
