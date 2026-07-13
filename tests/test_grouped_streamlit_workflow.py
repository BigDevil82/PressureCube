import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.structure.layout.params import EngineeringLayoutParameters, WindowParameters
from src.structure.section_grouping.discovery import sample_representative_designs


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


if __name__ == "__main__":
    unittest.main()
