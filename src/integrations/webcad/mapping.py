from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite

from src.structure.enums import BeamType
from src.structure.section_library import BoxSection, ChannelSection, HShapeSection
from src.structure.structural_model import StructuralModel


FAMILIES = {
    "box": ("squareTubeSteel", "0001"),
    "channel": ("channelSteel", "0002"),
    "gong": ("gongShapedSteel", "0003"),
}
SECTION_FIELDS = {
    "box": {"sideLengthA", "sideLengthB", "wallThicknessT", "outerArcRadiusR"},
    "gong": {"height", "legWidthB", "thicknessD", "thicknessT", "radiusR", "legArcRadiusR1"},
    "channel": {"height", "legWidthB", "thicknessD", "thicknessT", "radiusR", "legArcRadiusR1"},
}


@dataclass
class MappingConfig:
    # Section name -> {dimensions_confirmed: bool, sectionParams: {...}}.
    section_rules: dict[str, dict] = field(default_factory=dict)
    # Keys are "BEAM_TYPE:face:axis:sign", e.g. MAIN:bottom:x:+.
    rotations: dict[str, float] = field(default_factory=dict)
    horizontal_frames_confirmed: bool = False
    # Confirm after inspecting trial members in the target editor/workplane.
    world_placement_confirmed: bool = False


def _section(section, config: MappingConfig) -> tuple[str, dict, list[str]]:
    pending = []
    rule = config.section_rules.get(section.name, {})
    if isinstance(section, BoxSection):
        family = "box"
        values = {"sideLengthA": section.B * 1000, "sideLengthB": section.B * 1000, "wallThicknessT": section.t * 1000}
    elif isinstance(section, (HShapeSection, ChannelSection)):
        family = "channel" if isinstance(section, ChannelSection) else "gong"
        values = {"height": section.H * 1000, "legWidthB": section.B * 1000,
                  "thicknessD": section.tw * 1000, "thicknessT": section.tf * 1000}
        if not rule.get("dimensions_confirmed"):
            pending.append(f"{section.name}: confirm local thickness definitions and platform {family} section equivalence")
    else:
        raise ValueError(f"Unsupported section class: {type(section).__name__}")
    overrides = rule.get("sectionParams", {})
    if set(overrides) - SECTION_FIELDS[family]:
        raise ValueError(f"Unknown section fields for {section.name}")
    values.update(overrides)
    missing = SECTION_FIELDS[family] - values.keys()
    if missing:
        pending.append(f"{section.name}: missing confirmed dimensions {', '.join(sorted(missing))}")
    if any(not isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError(f"Non-positive or non-finite section dimension: {section.name}")
    if family == "box" and 2 * values["wallThicknessT"] >= min(values["sideLengthA"], values["sideLengthB"]):
        raise ValueError(f"Box section has no hollow interior: {section.name}")
    if family != "box" and (2 * values["thicknessT"] >= values["height"] or values["thicknessD"] >= values["legWidthB"]):
        raise ValueError(f"Invalid open section dimensions: {section.name}")
    return family, values, pending


def build_export_plan(model: StructuralModel, config: MappingConfig | None = None) -> dict:
    """Read actual final beam sections, retaining unresolved engineering decisions."""
    config = config or MappingConfig()
    members = []
    for beam_id, beam in sorted(model.beams.items()):
        if beam.is_virtual:
            continue
        if beam.parent_id is not None:
            raise ValueError("Export the original final model, not its FEA mesh")
        a = model.points[beam.start_node].coordinates().tolist()
        b = model.points[beam.end_node].coordinates().tolist()
        delta = [b[i] - a[i] for i in range(3)]
        axes = [i for i, v in enumerate(delta) if not isclose(v, 0, abs_tol=1e-8)]
        if len(axes) != 1:
            raise ValueError(f"Beam {beam_id}: only nonzero axis-aligned members are confirmed for this integration")
        axis = axes[0]
        pillar = beam.beam_type == BeamType.FRAME and axis == 2
        if pillar and a[2] > b[2]:
            a, b = b, a
            delta[2] *= -1
        family, section, pending = _section(beam.section, config)
        if beam.beam_type == BeamType.FRAME and not pillar and not config.horizontal_frames_confirmed:
            pending.append("Confirm horizontal FRAME members should be exported as steel girders")
        if not config.world_placement_confirmed:
            pending.append("Confirm world-coordinate placement and active workplane behavior using trial members")
        key = f"{beam.beam_type.name}:{beam.on_face}:{'xyz'[axis]}:{'+' if delta[axis] > 0 else '-'}"
        angle = config.rotations.get(key)
        if angle is None:
            pending.append(f"Confirm WebCAD cross-section rotation for {key}")
        elif not isfinite(angle):
            raise ValueError(f"Non-finite rotation for {key}")
        prefix, suffix = FAMILIES[family]
        type_code = "02-003" if pillar else "15-001"
        params = {
            "startPoint": dict(zip("xyz", [v * 1000 for v in a])),
            "endPoint": dict(zip("xyz", [v * 1000 for v in b])),
            "sectionParams": section, "baseType": 0,
            "typeCode": type_code, "componentCode": f"{type_code}-{suffix}",
        }
        if angle is not None:
            params["crosssectionRotationAngle"] = angle
        members.append({
            "source_id": beam_id,
            "group_id": model.section_assignment.beam_groups[beam_id],
            "beam_type": beam.beam_type.name, "face": beam.on_face,
            "rotation_key": key, "section_name": beam.section.name,
            "type": f"structure.{prefix}{'Pillar' if pillar else 'Girder'}",
            "params": params, "pending": pending,
        })
    return {"units": "mm", "members": members,
            "pending": sorted({p for member in members for p in member["pending"]})}
