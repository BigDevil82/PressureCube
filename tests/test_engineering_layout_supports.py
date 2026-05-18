import math

from src.structure.layout.engineering_layout import EngineeringLayoutEngine
from src.structure.layout.params import (
    DoorParameters,
    EngineeringLayoutParameters,
    ExtensionParameters,
    WindowParameters,
)


def _has(values, target):
    return any(math.isclose(value, target, abs_tol=1e-4) for value in values)


def test_front_window_gap_supports_are_evenly_spaced():
    params = EngineeringLayoutParameters(
        length=4.0,
        width=3.0,
        height=3.2,
        main_beam_spacing=1.0,
        support_beam_max_spacing=0.4,
        extension=ExtensionParameters(active=False),
        door=DoorParameters(active=False),
        window=WindowParameters(front_spans=[0, 2], back_spans=[], right_spans=[]),
    )

    engine = EngineeringLayoutEngine(params)

    assert _has(engine.sorted_y, 4 / 3)
    assert _has(engine.sorted_y, 5 / 3)


def test_window_vertical_supports_are_evenly_spaced_to_boundaries():
    params = EngineeringLayoutParameters(
        length=4.0,
        width=3.0,
        height=3.2,
        main_beam_spacing=1.0,
        support_beam_max_spacing=0.4,
        extension=ExtensionParameters(active=False),
        door=DoorParameters(active=False),
        window=WindowParameters(front_spans=[0], back_spans=[], right_spans=[]),
    )

    engine = EngineeringLayoutEngine(params)

    assert _has(engine.sorted_z, 0.65)
    assert _has(engine.sorted_z, 2.65)
