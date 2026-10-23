"""Analytic rolled-steel outlines in width/height coordinates, in millimetres."""

from math import atan2, pi, tan

import numpy as np


def rounded_polygon(vertices, radii):
    vertices = np.asarray(vertices, dtype=float)
    corners = []
    for i, (p, radius) in enumerate(zip(vertices, radii)):
        incoming = p - vertices[i - 1]
        outgoing = vertices[(i + 1) % len(vertices)] - p
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        turn = atan2(
            np.linalg.det(np.array([incoming, outgoing])), np.dot(incoming, outgoing)
        )
        distance = radius * tan(abs(turn) / 2)
        before, after = p - incoming * distance, p + outgoing * distance
        center = before + np.array([-incoming[1], incoming[0]]) * radius * np.sign(turn)
        corners.append((before, after, center, turn, radius))
    segments = []
    for i, (before, after, center, turn, radius) in enumerate(corners):
        start = corners[i - 1][1]
        if np.linalg.norm(before - start) > 1e-8:
            segments.append(
                {"type": "line", "start": start.tolist(), "end": before.tolist()}
            )
        if radius:
            segments.append(
                {
                    "type": "arc",
                    "start": before.tolist(),
                    "end": after.tolist(),
                    "center": center.tolist(),
                    "angle": turn * 180 / pi,
                    "radius": radius,
                }
            )
    return segments


def section_profile(family: str, p: dict) -> dict:
    if family == "box":
        b, h, t, r = (
            p["sideLengthA"],
            p["sideLengthB"],
            p["wallThicknessT"],
            p["outerArcRadiusR"],
        )
        outer = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
        inner = [
            (-b / 2 + t, -h / 2 + t),
            (-b / 2 + t, h / 2 - t),
            (b / 2 - t, h / 2 - t),
            (b / 2 - t, -h / 2 + t),
        ]
        loops = [
            rounded_polygon(outer, [r] * 4),
            rounded_polygon(inner, [max(r - t, 0)] * 4),
        ]
    elif family == "h":
        b, h, tw, tf, r = (
            p[k] for k in ["widthB", "height", "thicknessT1", "thicknessT2", "radius"]
        )
        vertices = [
            (-b / 2, -h / 2),
            (b / 2, -h / 2),
            (b / 2, -h / 2 + tf),
            (tw / 2, -h / 2 + tf),
            (tw / 2, h / 2 - tf),
            (b / 2, h / 2 - tf),
            (b / 2, h / 2),
            (-b / 2, h / 2),
            (-b / 2, h / 2 - tf),
            (-tw / 2, h / 2 - tf),
            (-tw / 2, -h / 2 + tf),
            (-b / 2, -h / 2 + tf),
        ]
        loops = [
            rounded_polygon(
                vertices, [r if i in {3, 4, 9, 10} else 0 for i in range(12)]
            )
        ]
    elif family == "channel":
        b, h, d, t, r, r1 = (
            p[k]
            for k in [
                "legWidthB",
                "height",
                "thicknessD",
                "thicknessT",
                "radiusR",
                "legArcRadiusR1",
            ]
        )

        # 1:10 flange taper: checked against the platform's actual 20b BREP.
        def inside(u):
            return h / 2 - t + 0.1 * (u - (b + d) / 2)

        vertices = [
            (0, -h / 2),
            (b, -h / 2),
            (b, -inside(b)),
            (d, -inside(d)),
            (d, inside(d)),
            (b, inside(b)),
            (b, h / 2),
            (0, h / 2),
        ]
        vertices = [(u - b / 2, v) for u, v in vertices]
        loops = [rounded_polygon(vertices, [0, 0, r1, r, r, r1, 0, 0])]
    else:
        raise ValueError(f"Unsupported production section: {family}")
    return {
        "loops": loops,
        "width_mm": b,
        "height_mm": h,
        "area_mm2": profile_area(loops),
    }


def profile_area(loops):
    area = 0.0
    for loop in loops:
        for segment in loop:
            a, b = np.array(segment["start"]), np.array(segment["end"])
            if segment["type"] == "line":
                area += (a[0] * b[1] - a[1] * b[0]) / 2
            else:
                c = np.array(segment["center"])
                delta = b - a
                area += (
                    c[0] * delta[1]
                    - c[1] * delta[0]
                    + segment["radius"] ** 2 * segment["angle"] * pi / 180
                ) / 2
    return area


def prism_params(part: dict) -> dict:
    origin = np.array(part["start_mm"])
    width = np.array(part["width_axis"])
    height = np.array(part["height_axis"])
    axis = np.array(part["axis_vector"])

    def point(uv):
        return dict(zip("xyz", (origin + width * uv[0] + height * uv[1]).tolist()))

    workplane = {
        "origin": dict(zip("xyz", origin.tolist())),
        "normal": dict(zip("xyz", axis.tolist())),
        "xvec": dict(zip("xyz", width.tolist())),
    }
    sketch = []
    for loop in part["profile"]["loops"]:
        for segment in loop:
            item = {
                "type": segment["type"],
                "start": point(segment["start"]),
                "workplane": workplane,
            }
            if segment["type"] == "line":
                item["end"] = point(segment["end"])
            else:
                item.update(
                    center=point(segment["center"]),
                    angle=segment["angle"],
                    radius=segment["radius"],
                    normal=workplane["normal"],
                )
            sketch.append(item)
    return {"sketch": sketch, "lengthStart": 0, "lengthEnd": part["length_mm"]}
