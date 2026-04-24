from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from src.analysis.post_processor import DesignCheckResult
from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.structure.layout.params import BaseLayoutParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel


@dataclass
class AnalysisState:
    analysis_done: bool = False
    analysis_type: str = "静力分析"
    mesh_size: float = 0.5
    pressure_pa: int = 30000
    load_case: str = "内压 (PL)"
    num_modes: int = 6
    meshed_model: StructuralModel | None = None
    analysis_results: AnalysisResults | None = None
    analysis_stats: DesignCheckResult | None = None
    modal_results: ModalAnalysisResults | None = None
    case_name: str = ""


class AppSession:
    _MODEL_BUNDLE_KEY = "app_model_bundle"
    _ANALYSIS_STATE_KEY = "app_analysis_state"

    def __init__(self):
        self._state = st.session_state

    def ensure_defaults(self):
        self._state.setdefault(self._MODEL_BUNDLE_KEY, {})
        self._state.setdefault(self._ANALYSIS_STATE_KEY, AnalysisState())

    @property
    def model_bundle(self) -> dict[str, Any]:
        self.ensure_defaults()
        return self._state[self._MODEL_BUNDLE_KEY]

    @property
    def analysis(self) -> AnalysisState:
        self.ensure_defaults()
        return self._state[self._ANALYSIS_STATE_KEY]

    def has_model(self) -> bool:
        bundle = self.model_bundle
        return bundle.get("model") is not None and bundle.get("params") is not None

    def set_model_bundle(
        self, params: BaseLayoutParameters, model: StructuralModel, section_library: SectionLibrary
    ):
        signature = repr(params)
        previous_signature = self.model_bundle.get("signature")
        self._state[self._MODEL_BUNDLE_KEY] = {
            "params": params,
            "model": model,
            "section_library": section_library,
            "signature": signature,
        }
        self._state["current_params"] = params
        self._state["current_model"] = model
        self._state["section_library"] = section_library
        if previous_signature != signature:
            self.reset_analysis()

    def reset_analysis(self):
        self._state[self._ANALYSIS_STATE_KEY] = AnalysisState()

    @property
    def params(self) -> BaseLayoutParameters | None:
        return self.model_bundle.get("params")

    @property
    def model(self) -> StructuralModel | None:
        return self.model_bundle.get("model")

    @property
    def section_library(self) -> SectionLibrary | None:
        return self.model_bundle.get("section_library")
