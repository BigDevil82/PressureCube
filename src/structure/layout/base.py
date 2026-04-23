import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from src.structure.enums import BeamType


@dataclass
class Zone:
    name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    is_hollow: bool = False

    def contains(self, x, y, z, tol=1e-4) -> bool:
        return (
            self.x_range[0] - tol <= x <= self.x_range[1] + tol
            and self.y_range[0] - tol <= y <= self.y_range[1] + tol
            and self.z_range[0] - tol <= z <= self.z_range[1] + tol
        )

    def on_surface(self, x, y, z, tol=1e-4) -> bool:
        if not self.contains(x, y, z, tol):
            return False
        on_x = math.isclose(x, self.x_range[0], abs_tol=tol) or math.isclose(x, self.x_range[1], abs_tol=tol)
        on_y = math.isclose(y, self.y_range[0], abs_tol=tol) or math.isclose(y, self.y_range[1], abs_tol=tol)
        on_z = math.isclose(z, self.z_range[0], abs_tol=tol) or math.isclose(z, self.z_range[1], abs_tol=tol)
        return on_x or on_y or on_z

    def on_edge(self, x, y, z, tol=1e-4) -> bool:
        if not self.contains(x, y, z, tol):
            return False
        on_x = math.isclose(x, self.x_range[0], abs_tol=tol) or math.isclose(x, self.x_range[1], abs_tol=tol)
        on_y = math.isclose(y, self.y_range[0], abs_tol=tol) or math.isclose(y, self.y_range[1], abs_tol=tol)
        on_z = math.isclose(z, self.z_range[0], abs_tol=tol) or math.isclose(z, self.z_range[1], abs_tol=tol)
        return (on_x and on_y) or (on_x and on_z) or (on_y and on_z)

    def get_face(self, x, y, z, tol=1e-4) -> Optional[str]:
        if math.isclose(z, self.z_range[1], abs_tol=tol):
            return "top"
        if math.isclose(z, self.z_range[0], abs_tol=tol):
            return "bottom"
        if math.isclose(x, self.x_range[0], abs_tol=tol):
            return "back"
        if math.isclose(x, self.x_range[1], abs_tol=tol):
            return "front"
        if math.isclose(y, self.y_range[0], abs_tol=tol):
            return "left"
        if math.isclose(y, self.y_range[1], abs_tol=tol):
            return "right"
        return None


class LayoutEngineProtocol(Protocol):
    zones: list[Zone]
    sorted_x: list[float]
    sorted_y: list[float]
    sorted_z: list[float]

    def determine_beam(
        self, p_start: np.ndarray, p_end: np.ndarray, axis: str
    ) -> Tuple[Optional[BeamType], str, bool]: ...

    def is_in_opening(self, x, y, z, tol=1e-4) -> bool: ...

    def is_in_sealed_opening(self, x, y, z, tol=1e-4) -> bool: ...
