"""Deterministic outer-envelope placement and square end cuts for orthogonal steel."""

from dataclasses import asdict, dataclass
import json

import numpy as np

from src.integrations.webcad.mapping import MappingConfig, _section
from src.integrations.webcad.profiles import prism_params, section_profile
from src.structure.layout import create_layout_engine


TOL = 1e-6
FACE_NORMALS = {
    "top": [0, 0, 1],
    "bottom": [0, 0, -1],
    "front": [1, 0, 0],
    "back": [-1, 0, 0],
    "left": [0, -1, 0],
    "right": [0, 1, 0],
}
RANK = {"FRAME": 0, "MAIN": 1, "SECONDARY": 2, "BRACE": 3}


@dataclass
class ProductionConfig:
    gap_mm: float = 0.0
    use_box_reference_radius: bool = True


def _stock_members(model, config):
    mapping = MappingConfig(use_box_reference_radius=config.use_box_reference_radius)
    groups = {}
    for beam_id, beam in sorted(model.beams.items()):
        if beam.is_virtual:
            continue
        if beam.parent_id is not None:
            raise ValueError("Use the original model, not an FEA mesh")
        a = np.array(model.points[beam.start_node].coordinates()) * 1000
        b = np.array(model.points[beam.end_node].coordinates()) * 1000
        axes = np.flatnonzero(np.abs(b - a) > TOL)
        if len(axes) != 1:
            raise ValueError(
                "Production conversion currently supports orthogonal straight members only"
            )
        axis = int(axes[0])
        if a[axis] > b[axis]:
            a, b = b, a
        family, params, pending, geometry = _section(beam.section, mapping)
        if pending:
            raise ValueError("; ".join(pending))
        group = model.section_assignment.beam_groups[beam_id]
        key = (
            axis,
            tuple(np.round(np.delete(a, axis), 6)),
            family,
            json.dumps(params, sort_keys=True),
            beam.beam_type.name,
            group,
            beam.on_face,
        )
        groups.setdefault(key, []).append(
            {
                "source_ids": [int(beam_id)],
                "start": a,
                "end": b,
                "axis": axis,
                "family": family,
                "section_params": params,
                "section_name": beam.section.name,
                "section_geometry": geometry,
                "beam_type": beam.beam_type.name,
                "group_id": group,
                "face": beam.on_face,
            }
        )
    stocks = []
    for segments in groups.values():
        segments.sort(key=lambda m: m["start"][m["axis"]])
        current = segments[0]
        for segment in segments[1:]:
            axis = current["axis"]
            if abs(segment["start"][axis] - current["end"][axis]) < TOL:
                current["end"] = segment["end"]
                current["source_ids"] += segment["source_ids"]
            else:
                stocks.append(current)
                current = segment
        stocks.append(current)
    return sorted(
        stocks,
        key=lambda m: (
            RANK[m["beam_type"]],
            -m["axis"],
            tuple(m["start"]),
            m["section_name"],
        ),
    )


def _place(stock, zones):
    axis = stock["axis"]
    direction = np.eye(3)[axis]
    height = np.array(FACE_NORMALS[stock["face"]], dtype=float)
    if abs(height[axis]) > TOL:
        raise ValueError("A member axis cannot be normal to its assigned face")
    width = np.cross(height, direction)
    profile = section_profile(stock["family"], stock["section_params"])
    size = np.abs(width) * profile["width_mm"] + np.abs(height) * profile["height_mm"]
    center = (stock["start"] + stock["end"]) / 2
    original = center.copy()
    # Main compartment is first: it owns a shared main/extension boundary.
    for zone in zones:
        low = np.array([zone.x_range[0], zone.y_range[0], zone.z_range[0]]) * 1000
        high = np.array([zone.x_range[1], zone.y_range[1], zone.z_range[1]]) * 1000
        if np.all(original >= low - TOL) and np.all(original <= high + TOL):
            for j in range(3):
                if j != axis:
                    center[j] = np.clip(
                        center[j], low[j] + size[j] / 2, high[j] - size[j] / 2
                    )
            break
    # Open channel flanges towards the centre of the panel rather than outward.
    if stock["family"] == "channel":
        zone = zones[0]
        panel_center = (
            np.array(
                [
                    (zone.x_range[0] + zone.x_range[1]) / 2,
                    (zone.y_range[0] + zone.y_range[1]) / 2,
                    (zone.z_range[0] + zone.z_range[1]) / 2,
                ]
            )
            * 1000
        )
        if np.dot(width, panel_center - center) < 0:
            width, height = -width, -height
    stock.update(
        center=center,
        size=size,
        width_axis=width,
        height_axis=height,
        axis_vector=direction,
        profile=profile,
        offset=center - original,
    )
    return stock


def _subtract(start, end, intervals):
    cursor = start
    remaining = []
    for a, b in sorted(intervals):
        a, b = max(a, start), min(b, end)
        if b <= cursor + TOL:
            continue
        if a > cursor + TOL:
            remaining.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end - TOL:
        remaining.append((cursor, end))
    return remaining


def _reconcile_grids(stocks, zones, gap):
    """Keep an end bay from collapsing into an inward-offset perimeter frame."""
    main = zones[0]
    ranges = [main.x_range, main.y_range, main.z_range]
    changes = {}
    for axis, (lo, hi) in enumerate(ranges):
        lo, hi = lo * 1000, hi * 1000
        levels = {}
        for stock in stocks:
            if stock["axis"] != axis:
                value = float(stock["start"][axis])
                levels[value] = max(levels.get(value, 0), stock["size"][axis])
        lower_band, upper_band = levels.get(lo, 0), levels.get(hi, 0)
        for value, size in levels.items():
            if lo + TOL < value < hi - TOL:
                placed = float(
                    np.clip(
                        value,
                        lo + lower_band + size / 2 + gap,
                        hi - upper_band - size / 2 - gap,
                    )
                )
                if abs(placed - value) > TOL:
                    changes[(axis, value)] = placed - value
    for stock in stocks:
        axis = stock["axis"]
        for j in range(3):
            if j != axis:
                stock["center"][j] += changes.get((j, float(stock["start"][j])), 0)
        stock["offset"] = stock["center"] - (stock["start"] + stock["end"]) / 2
        stock["interval"] = [
            float(stock["start"][axis])
            + changes.get((axis, float(stock["start"][axis])), 0),
            float(stock["end"][axis])
            + changes.get((axis, float(stock["end"][axis])), 0),
        ]
    return [
        {"axis": "xyz"[axis], "nominal_mm": value, "placed_mm": value + delta}
        for (axis, value), delta in sorted(changes.items())
    ]


def overlap_pairs(parts, tolerance=TOL):
    if not parts:
        return []
    low = np.array([p["bounds_mm"][:3] for p in parts])
    high = np.array([p["bounds_mm"][3:] for p in parts])
    overlaps = []
    for i in range(len(parts)):
        depth = np.minimum(high[i], high[i + 1 :]) - np.maximum(low[i], low[i + 1 :])
        for relative in np.flatnonzero(np.all(depth > tolerance, axis=1)):
            overlaps.append((parts[i]["id"], parts[i + 1 + relative]["id"]))
    return overlaps


def _stock_cuts(stock, parts, gap):
    axis = stock["axis"]
    transverse = [j for j in range(3) if j != axis]
    start, end = stock["interval"]
    low = stock["center"] - stock["size"] / 2
    high = stock["center"] + stock["size"] / 2
    intervals, hosts = [], []
    for host in parts:
        bounds = host["bounds_mm"]
        if all(
            min(high[j], bounds[j + 3]) - max(low[j], bounds[j]) > TOL
            for j in transverse
        ):
            a, b = bounds[axis] - gap, bounds[axis + 3] + gap
            if min(end, b) - max(start, a) > TOL:
                intervals.append((a, b))
                hosts.append(
                    {
                        "host_id": host["id"],
                        "removed_interval_mm": [
                            float(max(start, a)),
                            float(min(end, b)),
                        ],
                    }
                )
    return _subtract(start, end, intervals), hosts


def _make_part(stock, stock_id, part_id, a, b):
    axis = stock["axis"]
    p1, p2 = stock["center"].copy(), stock["center"].copy()
    p1[axis], p2[axis] = a, b
    low = stock["center"] - stock["size"] / 2
    high = stock["center"] + stock["size"] / 2
    low[axis], high[axis] = a, b
    metadata = {
        key: stock[key]
        for key in [
            "group_id",
            "beam_type",
            "face",
            "family",
            "section_name",
            "section_params",
            "section_geometry",
            "axis",
            "profile",
        ]
    }
    return {
        **metadata,
        "id": f"production-{part_id:04d}",
        "stock_id": stock_id,
        "source_ids": list(stock["source_ids"]),
        "start_mm": p1.tolist(),
        "end_mm": p2.tolist(),
        "length_mm": float(b - a),
        "width_axis": stock["width_axis"].tolist(),
        "height_axis": stock["height_axis"].tolist(),
        "axis_vector": stock["axis_vector"].tolist(),
        "bounds_mm": [*low.tolist(), *high.tolist()],
        "offset_mm": stock["offset"].tolist(),
        "volume_mm3": stock["profile"]["area_mm2"] * (b - a),
    }


def _cut_stocks(stocks, gap):
    parts, stock_records = [], []
    for index, stock in enumerate(stocks):
        spans, hosts = _stock_cuts(stock, parts, gap)
        record = {
            "stock_id": index + 1,
            "source_ids": stock["source_ids"],
            "group_id": stock["group_id"],
            "original_start_mm": stock["start"].tolist(),
            "original_end_mm": stock["end"].tolist(),
            "offset_mm": stock["offset"].tolist(),
            "joints": hosts,
            "part_ids": [],
        }
        for a, b in spans:
            part = _make_part(stock, index + 1, len(parts) + 1, a, b)
            parts.append(part)
            record["part_ids"].append(part["id"])
        stock_records.append(record)
    return parts, stock_records


def _verify_layout(model, stocks, stock_records, parts):
    if overlap_pairs(parts):
        raise ValueError("Production envelope overlap remains")
    if any(not stock["part_ids"] for stock in stock_records):
        raise ValueError(
            "A complete stock would be removed: revise the physical grid or section sizes"
        )
    stock_lookup = {i + 1: stock for i, stock in enumerate(stocks)}
    for part in parts:
        stock = stock_lookup[part["stock_id"]]
        normal = np.array(FACE_NORMALS[part["face"]])
        j = int(np.flatnonzero(normal)[0])
        boundary = part["bounds_mm"][j + 3 if normal[j] > 0 else j]
        if abs(boundary - stock["start"][j]) > TOL:
            raise ValueError(
                f"Part {part['id']} is not flush with its assigned outer face"
            )
    low = np.min([p["bounds_mm"][:3] for p in parts], axis=0)
    high = np.max([p["bounds_mm"][3:] for p in parts], axis=0)
    expected_low = [
        0,
        -model.params.extension.length * 1000 if model.params.extension.active else 0,
        0,
    ]
    expected_high = [
        model.params.width * 1000,
        model.params.length * 1000,
        model.params.height * 1000,
    ]
    if not np.allclose(
        [*low, *high], [*expected_low, *expected_high], atol=TOL, rtol=0
    ):
        raise ValueError("Outer steel envelope differs from the requested dimensions")
    originals = {i for i, b in model.beams.items() if not b.is_virtual}
    accounted = {i for stock in stock_records for i in stock["source_ids"]}
    if accounted != originals:
        raise ValueError("Source-member coverage mismatch")
    return {
        "source_segments": len(originals),
        "merged_stocks": len(stocks),
        "parts": len(parts),
        "fully_consumed_stocks": 0,
        "bounds_mm": [*low.tolist(), *high.tolist()],
        "overlap_pairs": 0,
        "steel_mass_kg": sum(p["volume_mm3"] for p in parts) * 7850e-9,
    }


def build_production_plan(model, config: ProductionConfig | None = None):
    config = config or ProductionConfig()
    if config.gap_mm < 0:
        raise ValueError("gap_mm must be nonnegative")
    zones = create_layout_engine(model.params).zones
    stocks = [_place(stock, zones) for stock in _stock_members(model, config)]
    grid_changes = _reconcile_grids(stocks, zones, config.gap_mm)
    parts, stock_records = _cut_stocks(stocks, config.gap_mm)
    summary = _verify_layout(model, stocks, stock_records, parts)
    return {
        "units": "mm",
        "reference": "steel_outer_faces_excluding_cladding",
        "config": asdict(config),
        "model_parameters": {
            k: getattr(model.params, k)
            for k in [
                "length",
                "width",
                "height",
                "main_beam_spacing",
                "support_beam_max_spacing",
            ]
        },
        "parts": parts,
        "stocks": stock_records,
        "grid_adjustments": grid_changes,
        "summary": summary,
        "limitations": [
            "Geometry only: welds, bolts, plates and connection strength are not detailed.",
            "Opening inputs remain nominal grid dimensions, not guaranteed clear opening sizes.",
            "Source FEA model is unchanged; offsets and joints need engineering review.",
            "Generic prism nodes retain section and source mapping in the external BOM.",
        ],
    }


def production_request(plan, batch_size=100):
    nodes = []
    for part in plan["parts"]:
        params = prism_params(part)
        params["id"] = part["id"]
        nodes.append({"type": "engine.prism", "params": params})
    return {
        "nodeList": nodes,
        "components": [],
        "batchSize": batch_size,
        "skipCutRelations": True,
    }
