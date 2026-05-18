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

    for z in (0.3, 0.6, 1.275, 1.65, 2.025, 2.8):
        assert _has(engine.sorted_z, z)


def test_default_window_height_is_split_by_support_spacing():
    params = EngineeringLayoutParameters(
        length=8.9,
        width=3.7,
        height=3.3,
        main_beam_spacing=0.95,
        support_beam_max_spacing=0.5,
        window=WindowParameters(front_spans=[], back_spans=[], right_spans=[1, 2]),
    )

    engine = EngineeringLayoutEngine(params)
    inside = [z for z in engine.sorted_z if 0.9 <= z <= 2.4]

    assert all(_has(inside, z) for z in (0.9, 1.4, 1.9, 2.4))
    assert len(inside) == 4
