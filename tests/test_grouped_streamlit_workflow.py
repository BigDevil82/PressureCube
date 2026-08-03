import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.optimization.section_constraints import (
    apply_group_section_constraints,
    apply_ungrouped_section_constraints,
)
from src.reinforcement.sb3.optimize import select_best_state
from src.structure.enums import BeamType
from src.structure.layout.params import EngineeringLayoutParameters, WindowParameters
from src.structure.section_grouping.discovery import sample_representative_designs
from src.structure.section_grouping.models import SectionGroupingRuleset


class SurrogateFeatureSchemaTest(unittest.TestCase):
    def test_selects_group_feature_schema_from_model_dimension(self):
        model = AdvancedSurrogateModel()
        model.trainer = SimpleNamespace(X_mean=np.zeros(2))

        with (
            patch("src.analysis.surrogate_model.design_to_feature_vector", return_value=np.zeros(1)),
            patch("src.analysis.surrogate_model.design_to_group_feature_vector", return_value=np.ones(2)),
            patch("src.analysis.surrogate_model.design_to_group_feature_vector_v2", return_value=np.ones(3)),
        ):
            values = model._feature_vector(SimpleNamespace())

        np.testing.assert_array_equal(values, np.ones(2))


class RepresentativeDesignTest(unittest.TestCase):
    def test_current_engineering_parameters_are_kept_in_default_sample(self):
        params = EngineeringLayoutParameters(
            length=8.9,
            width=3.6,
            height=3.3,
            main_beam_spacing=1.05,
            support_beam_max_spacing=0.65,
            add_support_beam=True,
            window=WindowParameters(right_spans=[1, 2]),
        )

        designs = sample_representative_designs(
            geometry={"L": params.length, "W": params.width, "H": params.height},
            pressure=30000.0,
            sections=[2, 8, 7, 14],
            layout_strategy="engineering",
            n_samples=1,
            base_params=params,
        )

        sampled = designs[0].params
        self.assertEqual(sampled.main_beam_spacing, 1.05)
        self.assertEqual(sampled.support_beam_max_spacing, 0.65)
        self.assertEqual(sampled.window.right_spans, [1, 2])


class FinalSectionConstraintTest(unittest.TestCase):
    def test_frame_groups_use_at_least_box_150x6(self):
        rules = SectionGroupingRuleset(
            layout_strategy="engineering",
            n_groups_by_type={BeamType.FRAME: 2, BeamType.MAIN: 1},
            prototypes=[],
            feature_schema=[],
            created_from_samples=0,
        )
        sections = {
            "FRAME-SG1": 0,
            "FRAME-SG2": 3,
            "MAIN-SG1": 1,
        }

        constrained = apply_group_section_constraints(sections, rules.group_beam_types())

        self.assertEqual(constrained["FRAME-SG1"], 2)
        self.assertEqual(constrained["FRAME-SG2"], 3)
        self.assertEqual(constrained["MAIN-SG1"], 1)

    def test_ungrouped_frame_uses_at_least_box_150x6(self):
        self.assertEqual(apply_ungrouped_section_constraints([0, 8, 7, 14]), [2, 8, 7, 14])
        self.assertEqual(apply_ungrouped_section_constraints([3, 8, 7, 14]), [3, 8, 7, 14])


class RLResultSelectionTest(unittest.TestCase):
    def test_selects_lightest_feasible_state(self):
        trajectory = [
            {"step": 0, "mass": 120.0, "u_sigma": 0.8, "u_delta": 0.9, "feasible": True},
            {"step": 1, "mass": 140.0, "u_sigma": 0.7, "u_delta": 0.8, "feasible": True},
            {"step": 2, "mass": 100.0, "u_sigma": 1.2, "u_delta": 0.8, "feasible": False},
        ]

        self.assertEqual(select_best_state(trajectory)["step"], 0)

    def test_selects_least_violating_state_when_none_feasible(self):
        trajectory = [
            {"step": 0, "mass": 100.0, "u_sigma": 1.5, "u_delta": 1.0, "feasible": False},
            {"step": 1, "mass": 130.0, "u_sigma": 1.1, "u_delta": 1.1, "feasible": False},
            {"step": 2, "mass": 90.0, "u_sigma": 2.0, "u_delta": 1.0, "feasible": False},
        ]

        self.assertEqual(select_best_state(trajectory)["step"], 1)


if __name__ == "__main__":
    unittest.main()
