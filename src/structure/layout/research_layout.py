import math
from typing import Optional

import numpy as np

from src.structure.enums import BeamType
from src.structure.layout.base import Zone
from src.structure.layout.params import StructuralParameters


class ResearchLayoutEngine:
    """Current research-oriented layout engine (legacy behavior)."""

    def __init__(self, params: StructuralParameters):
        self.p = params
        self.zones: list[Zone] = []
        self.main_zone = None

        self.grids_x = set()
        self.grids_y = set()
        self.grids_z = set()

        self.secondary_x_locs = set()
        self.brace_x_locs = set()
        self.brace_z_locs = set()

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
        self._fill_secondary_x_grids()
        self._fill_secondary_z_grids()

        self.sorted_x = sorted(list(self.grids_x))
        self.sorted_y = sorted(list(self.grids_y))
        self.sorted_z = sorted(list(self.grids_z))

    def _fill_main_beam_y_grids(self):
        curr = 0.0
        while curr <= self.p.length + 1e-5:
            self.grids_y.add(curr)
            curr += self.p.main_beam_spacing
        self.grids_y.add(self.p.length)

        if self.p.extension.active:
            self.grids_y.add(-self.p.extension.length)

    def _fill_secondary_x_grids(self):
        n_segs = self.p.num_secondary_width
        step = self.p.width / n_segs

        for i in range(1, n_segs):
            pos = i * step
            self.grids_x.add(pos)
            self.secondary_x_locs.add(pos)
            if self.p.add_support_beam:
                brace_pos = pos - (step / 2)
                self.grids_x.add(brace_pos)
                self.brace_x_locs.add(brace_pos)

        if self.p.add_support_beam:
            self.grids_x.add(self.p.width - (step / 2))
            self.brace_x_locs.add(self.p.width - (step / 2))

    def _fill_secondary_z_grids(self):
        n_segs = self.p.num_secondary_height
        step = self.p.height / n_segs

        for i in range(1, n_segs):
            pos = i * step
            self.grids_z.add(pos)
            if self.p.add_support_beam:
                brace_pos = pos - (step / 2)
                self.grids_z.add(brace_pos)
                self.brace_z_locs.add(brace_pos)

        if self.p.add_support_beam:
            self.grids_z.add(self.p.height - (step / 2))
            self.brace_z_locs.add(self.p.height - (step / 2))

    def is_in_opening(self, x, y, z, tol=1e-4) -> bool:
        ext = self.p.extension
        if not ext.active:
            return False

        if not math.isclose(y, 0, abs_tol=tol):
            return False

        in_x = (x > ext.offset_x + tol) and (x < self.p.width - tol)
        in_z = (z > 0 + tol) and (z < ext.height - tol)
        return in_x and in_z

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

        is_split_line = False
        if self.p.extension.active:
            is_split_line = math.isclose(x, self.p.extension.offset_x, abs_tol=1e-4)

        check_virtual_necessity = is_split_line and (y > 1e-4)

        proposed_type = None
        is_virtual = False

        if axis == "x":
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                proposed_type = BeamType.MAIN
            else:
                is_brace = any(math.isclose(z, bz) for bz in self.brace_z_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.SECONDARY

        elif axis == "y":
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                is_brace = any(math.isclose(x, bx) for bx in self.brace_x_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                elif any(math.isclose(x, sx) for sx in self.secondary_x_locs):
                    proposed_type = BeamType.SECONDARY
            else:
                is_brace = any(math.isclose(z, bz) for bz in self.brace_z_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.SECONDARY

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
        if check_virtual_necessity:
            return BeamType.MAIN, face_name, True
        return None, "", False
