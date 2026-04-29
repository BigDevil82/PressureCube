from __future__ import annotations

import streamlit as st

from src.structure.layout.engineering_layout import EngineeringLayoutEngine
from src.structure.enums import BeamType
from src.structure.layout.params import (
    EngineeringLayoutParameters,
    ResearchLayoutParameters,
    WindowParameters,
)


def default_ui_state() -> dict:
    return {
        "layout_mode": "research",
        "L": 8.9,
        "W": 3.7,
        "H": 3.3,
        "beam_spacing": 0.95,
        "num_sec_w": 3,
        "num_sec_h": 3,
        "add_brace": True,
        "frame_idx": 2,
        "main_idx": 8,
        "sec_idx": 7,
        "brace_idx": 14,
        "support_beam_max_spacing": 0.75,
        "window_active": True,
        "window_sill_height": 0.9,
        "window_height": 1.5,
        "front_spans": [],
        "back_spans": [],
        "right_spans": [1, 2],
    }


def hydrate_defaults() -> dict:
    defaults = default_ui_state()
    saved_params = st.session_state.get("current_params")
    if saved_params is None:
        return defaults

    defaults["layout_mode"] = getattr(saved_params, "layout_strategy", "research")
    defaults["L"] = saved_params.length
    defaults["W"] = saved_params.width
    defaults["H"] = saved_params.height
    defaults["beam_spacing"] = saved_params.main_beam_spacing
    defaults["add_brace"] = saved_params.add_support_beam
    saved_model = st.session_state.get("current_model")
    if saved_model is not None:
        section_indices = saved_model.section_assignment.section_indices
        defaults["frame_idx"] = section_indices.get(f"{BeamType.FRAME.name}-SG1", defaults["frame_idx"])
        defaults["main_idx"] = section_indices.get(f"{BeamType.MAIN.name}-SG1", defaults["main_idx"])
        defaults["sec_idx"] = section_indices.get(f"{BeamType.SECONDARY.name}-SG1", defaults["sec_idx"])
        defaults["brace_idx"] = section_indices.get(f"{BeamType.BRACE.name}-SG1", defaults["brace_idx"])

    if isinstance(saved_params, ResearchLayoutParameters):
        defaults["num_sec_w"] = saved_params.num_secondary_width
        defaults["num_sec_h"] = saved_params.num_secondary_height

    if isinstance(saved_params, EngineeringLayoutParameters):
        defaults["support_beam_max_spacing"] = saved_params.support_beam_max_spacing
        defaults["window_active"] = saved_params.window.active
        defaults["window_sill_height"] = saved_params.window.sill_height
        defaults["window_height"] = saved_params.window.height
        defaults["front_spans"] = saved_params.window.front_spans
        defaults["back_spans"] = saved_params.window.back_spans
        defaults["right_spans"] = saved_params.window.right_spans

    return defaults


def build_engineering_preview(
    length: float, width: float, height: float, beam_spacing: float, add_brace: bool, support_spacing: float
) -> EngineeringLayoutEngine:
    preview_params = EngineeringLayoutParameters(
        length=length,
        width=width,
        height=height,
        main_beam_spacing=beam_spacing,
        add_support_beam=add_brace,
        support_beam_max_spacing=support_spacing,
        window=WindowParameters(active=False),
    )
    return EngineeringLayoutEngine(preview_params)
