from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from src.structure.layout.params import (
    BaseLayoutParameters,
    EngineeringLayoutParameters,
    ResearchLayoutParameters,
)
from src.structure.section_library import SectionLibrary, SectionType


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: Literal["categorical", "int", "float", "bool"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: Sequence[Any] | None = None
    default: Any = None


@dataclass
class DesignConfig:
    L: float
    W: float
    H: float
    pressure: float
    main_beam_spacing: float
    add_support_beam: bool
    frame_section_index: int
    main_beam_section_index: int
    secondary_beam_section_index: int
    brace_section_index: int
    layout_strategy: str = field(init=False, default="base")

    def section_indices(self) -> list[int]:
        return [
            self.frame_section_index,
            self.main_beam_section_index,
            self.secondary_beam_section_index,
            self.brace_section_index,
        ]

    def topology_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_feature_vector(self, use_physics: bool = True) -> np.ndarray:
        if not use_physics:
            raise NotImplementedError("Index-based features are deprecated. Use physics-based features instead.")

        lib = get_section_lib(self.layout_strategy)
        adapter = get_layout_adapter(self.layout_strategy)
        sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        sections = [lib.get_section(sec_type, idx) for sec_type, idx in zip(sec_types, self.section_indices())]

        volume = self.L * self.W * self.H
        floor_area = self.L * self.W
        aspect_ratio = self.L / self.W if self.W > 0 else 1.0
        pressure_kpa = self.pressure / 1000.0
        total_load = self.pressure * floor_area
        n_main_beams = int(self.L / self.main_beam_spacing) + 1

        areas = [section.area * 1e4 for section in sections]
        ixs = [section.Ix * 1e8 for section in sections]
        iys = [section.Iy * 1e8 for section in sections]

        total_area = sum(areas)
        avg_ix = float(np.mean(ixs))
        min_ix = min(ixs)
        max_ix = max(ixs)
        ix_ratio = min_ix / (max_ix + 1e-10)
        stiffness_load_ratio = avg_ix / (total_load / 1e6 + 1e-10)
        strength_load_ratio = total_area / (total_load / 1e6 + 1e-10)

        features = [
            self.L,
            self.W,
            self.H,
            volume,
            floor_area,
            aspect_ratio,
            pressure_kpa,
            total_load / 1e6,
            self.main_beam_spacing,
            *adapter.topology_feature_values(self),
            float(n_main_beams),
            *areas,
            *ixs,
            *iys,
            total_area,
            avg_ix,
            min_ix,
            ix_ratio,
            stiffness_load_ratio,
            strength_load_ratio,
        ]
        return np.array(features, dtype=np.float32)

    @classmethod
    def feature_names(cls, layout_strategy: str, use_physics: bool = True) -> list[str]:
        if not use_physics:
            raise NotImplementedError("Index-based features are deprecated. Use physics-based features instead.")

        adapter = get_layout_adapter(layout_strategy)
        return [
            "L",
            "W",
            "H",
            "volume",
            "floor_area",
            "aspect_ratio",
            "pressure_kPa",
            "total_load_MN",
            "main_beam_spacing",
            *adapter.topology_feature_names(),
            "n_main_beams",
            "frame_area",
            "main_area",
            "sec_area",
            "brace_area",
            "frame_Ix",
            "main_Ix",
            "sec_Ix",
            "brace_Ix",
            "frame_Iy",
            "main_Iy",
            "sec_Iy",
            "brace_Iy",
            "total_area",
            "avg_Ix",
            "min_Ix",
            "Ix_ratio",
            "stiffness_load_ratio",
            "strength_load_ratio",
        ]


@dataclass
class ResearchDesignConfig(DesignConfig):
    num_secondary_width: int = 3
    num_secondary_height: int = 3
    layout_strategy: Literal["research"] = field(init=False, default="research")

    def topology_dict(self) -> dict[str, Any]:
        return {
            "main_beam_spacing": self.main_beam_spacing,
            "num_secondary_width": self.num_secondary_width,
            "num_secondary_height": self.num_secondary_height,
            "add_support_beam": self.add_support_beam,
        }


@dataclass
class EngineeringDesignConfig(DesignConfig):
    support_beam_max_spacing: float = 0.75
    layout_strategy: Literal["engineering"] = field(init=False, default="engineering")

    def topology_dict(self) -> dict[str, Any]:
        return {
            "main_beam_spacing": self.main_beam_spacing,
            "support_beam_max_spacing": self.support_beam_max_spacing,
            "add_support_beam": self.add_support_beam,
        }


class LayoutDesignAdapter:
    layout_strategy: str
    config_cls: type[DesignConfig]
    params_cls: type[BaseLayoutParameters]
    default_section_lib_path: str

    def topology_specs(self) -> list[VariableSpec]:
        raise NotImplementedError

    def topology_feature_names(self) -> list[str]:
        raise NotImplementedError

    def topology_feature_values(self, config: DesignConfig) -> list[float]:
        raise NotImplementedError

    def default_topology(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.topology_specs()}

    def build_config(self, *, geometry: dict[str, float], pressure: float, topology: dict[str, Any], sections: list[int]) -> DesignConfig:
        return self.config_cls(
            L=geometry["L"],
            W=geometry["W"],
            H=geometry["H"],
            pressure=pressure,
            frame_section_index=sections[0],
            main_beam_section_index=sections[1],
            secondary_beam_section_index=sections[2],
            brace_section_index=sections[3],
            **topology,
        )

    def build_params(self, config: DesignConfig) -> BaseLayoutParameters:
        return self.params_cls(
            length=config.L,
            width=config.W,
            height=config.H,
            main_beam_spacing=config.main_beam_spacing,
            add_support_beam=config.add_support_beam,
            frame_section_index=config.frame_section_index,
            main_beam_section_index=config.main_beam_section_index,
            secondary_beam_section_index=config.secondary_beam_section_index,
            brace_section_index=config.brace_section_index,
            **self._extra_param_kwargs(config),
        )

    def _extra_param_kwargs(self, config: DesignConfig) -> dict[str, Any]:
        return {}


class ResearchLayoutAdapter(LayoutDesignAdapter):
    layout_strategy = "research"
    config_cls = ResearchDesignConfig
    params_cls = ResearchLayoutParameters
    default_section_lib_path = "src/data/steel_sections_large.json"

    def topology_specs(self) -> list[VariableSpec]:
        return [
            VariableSpec("main_beam_spacing", "categorical", choices=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2], default=0.7),
            VariableSpec("num_secondary_width", "int", low=2, high=6, default=4),
            VariableSpec("num_secondary_height", "int", low=2, high=6, default=4),
            VariableSpec("add_support_beam", "bool", choices=[True, False], default=True),
        ]

    def topology_feature_names(self) -> list[str]:
        return ["num_secondary_width", "num_secondary_height", "add_support_beam"]

    def topology_feature_values(self, config: DesignConfig) -> list[float]:
        assert isinstance(config, ResearchDesignConfig)
        return [float(config.num_secondary_width), float(config.num_secondary_height), float(config.add_support_beam)]

    def _extra_param_kwargs(self, config: DesignConfig) -> dict[str, Any]:
        assert isinstance(config, ResearchDesignConfig)
        return {
            "num_secondary_width": config.num_secondary_width,
            "num_secondary_height": config.num_secondary_height,
        }


class EngineeringLayoutAdapter(LayoutDesignAdapter):
    layout_strategy = "engineering"
    config_cls = EngineeringDesignConfig
    params_cls = EngineeringLayoutParameters
    default_section_lib_path = "src/data/steel_sections.json"

    def topology_specs(self) -> list[VariableSpec]:
        return [
            VariableSpec("main_beam_spacing", "categorical", choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], default=0.95),
            VariableSpec("support_beam_max_spacing", "categorical", choices=[0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05], default=0.75),
            VariableSpec("add_support_beam", "bool", choices=[True, False], default=True),
        ]

    def topology_feature_names(self) -> list[str]:
        return ["support_beam_max_spacing", "add_support_beam"]

    def topology_feature_values(self, config: DesignConfig) -> list[float]:
        assert isinstance(config, EngineeringDesignConfig)
        return [float(config.support_beam_max_spacing), float(config.add_support_beam)]

    def _extra_param_kwargs(self, config: DesignConfig) -> dict[str, Any]:
        assert isinstance(config, EngineeringDesignConfig)
        return {"support_beam_max_spacing": config.support_beam_max_spacing}


_ADAPTERS: dict[str, LayoutDesignAdapter] = {
    "research": ResearchLayoutAdapter(),
    "engineering": EngineeringLayoutAdapter(),
}

_SECTION_LIB_CACHE: dict[str, SectionLibrary] = {}


def get_default_section_lib_path(layout_strategy: str) -> str:
    return get_layout_adapter(layout_strategy).default_section_lib_path


def get_section_lib(layout_strategy: str = "research", file_path: str | None = None) -> SectionLibrary:
    resolved_path = file_path or get_default_section_lib_path(layout_strategy)
    if resolved_path not in _SECTION_LIB_CACHE:
        _SECTION_LIB_CACHE[resolved_path] = SectionLibrary(resolved_path)
    return _SECTION_LIB_CACHE[resolved_path]


def get_layout_adapter(layout_strategy: str) -> LayoutDesignAdapter:
    if layout_strategy not in _ADAPTERS:
        raise ValueError(f"Unsupported layout strategy: {layout_strategy}")
    return _ADAPTERS[layout_strategy]


def build_params_from_config(config: DesignConfig) -> BaseLayoutParameters:
    return get_layout_adapter(config.layout_strategy).build_params(config)


def compute_mass_fast(config: DesignConfig, section_lib: SectionLibrary) -> float:
    from src.structure.structural_model import StructuralModel

    params = build_params_from_config(config)
    model = StructuralModel(params, section_lib)
    model.generate()
    return model.calculate_mass_kg()


def design_config_from_params(params: BaseLayoutParameters, pressure: float) -> DesignConfig:
    common = dict(
        L=params.length,
        W=params.width,
        H=params.height,
        pressure=pressure,
        main_beam_spacing=params.main_beam_spacing,
        add_support_beam=params.add_support_beam,
        frame_section_index=params.frame_section_index,
        main_beam_section_index=params.main_beam_section_index,
        secondary_beam_section_index=params.secondary_beam_section_index,
        brace_section_index=params.brace_section_index,
    )
    if isinstance(params, ResearchLayoutParameters):
        return ResearchDesignConfig(
            **common,
            num_secondary_width=params.num_secondary_width,
            num_secondary_height=params.num_secondary_height,
        )
    if isinstance(params, EngineeringLayoutParameters):
        return EngineeringDesignConfig(
            **common,
            support_beam_max_spacing=params.support_beam_max_spacing,
        )
    raise TypeError(f"Unsupported params type: {type(params)}")


def sample_spec_from_unit(spec: VariableSpec, value: float) -> Any:
    value = min(max(value, 0.0), 0.999999)
    if spec.kind in {"categorical", "bool"}:
        choices = list(spec.choices or [])
        idx = min(int(value * len(choices)), len(choices) - 1)
        return choices[idx]
    if spec.kind == "int":
        assert spec.low is not None and spec.high is not None
        span = int(spec.high) - int(spec.low) + 1
        return min(int(spec.low) + int(value * span), int(spec.high))
    if spec.kind == "float":
        assert spec.low is not None and spec.high is not None
        raw = float(spec.low) + value * (float(spec.high) - float(spec.low))
        if spec.step:
            steps = round((raw - float(spec.low)) / float(spec.step))
            raw = float(spec.low) + steps * float(spec.step)
        return raw
    raise ValueError(f"Unsupported variable spec kind: {spec.kind}")


def suggest_spec_with_optuna(trial: Any, spec: VariableSpec) -> Any:
    if spec.kind in {"categorical", "bool"}:
        return trial.suggest_categorical(spec.name, list(spec.choices or []))
    if spec.kind == "int":
        return trial.suggest_int(spec.name, int(spec.low), int(spec.high))
    if spec.kind == "float":
        kwargs = {}
        if spec.step is not None:
            kwargs["step"] = spec.step
        return trial.suggest_float(spec.name, float(spec.low), float(spec.high), **kwargs)
    raise ValueError(f"Unsupported variable spec kind: {spec.kind}")
