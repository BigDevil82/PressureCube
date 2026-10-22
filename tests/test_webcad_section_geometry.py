import copy
import math
import unittest
from pathlib import Path

from src.integrations.webcad.mapping import MappingConfig, _section
from src.integrations.webcad.section_geometry import find_section_geometry, geometry_catalog
from src.structure.section_library import HShapeSection, SectionLibrary, SectionType


ROOT = Path(__file__).resolve().parents[1]


class SectionGeometryTests(unittest.TestCase):
    def test_all_used_open_sections_match_standard_tables(self):
        for name in ["steel_sections.json", "steel_sections_large.json"]:
            library = SectionLibrary(str(ROOT / "src/data" / name))
            for section_type in [SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]:
                for index in range(library.section_counts[section_type]):
                    section = library.get_section(section_type, index)
                    with self.subTest(library=name, section=section.name):
                        self.assertIsNotNone(find_section_geometry(section))
                        record = find_section_geometry(section)
                        for field, value in [("area_cm2", section.area * 10000), ("Ix_cm4", section.Ix * 1e8), ("Iy_cm4", section.Iy * 1e8)]:
                            self.assertTrue(math.isclose(record[field], value, rel_tol=1e-8, abs_tol=1e-8))

    def test_catalog_has_source_pages_and_expected_counts(self):
        rows = geometry_catalog()["sections"]
        self.assertEqual(sum(row["family"] == "h" for row in rows), 21)
        self.assertEqual(sum(row["family"] == "channel" for row in rows), 20)
        self.assertEqual(sum(row["family"] == "box" for row in rows), 23)
        for row in rows:
            self.assertIn(row["source"]["standard"], geometry_catalog()["standards"])
            self.assertGreater(row["source"]["pdf_page"], 0)
            self.assertIn(row["source"]["table"], ["A.1", "A.2", "A.1 footnote"])

    def test_h150_geometry_is_h_not_tapered_i(self):
        library = SectionLibrary(str(ROOT / "src/data/steel_sections.json"))
        section = library.get_section(SectionType.H_MAIN, 8)
        family, params, pending, provenance = _section(section, MappingConfig())
        self.assertEqual(family, "h")
        self.assertEqual(params, {"height": 150, "widthB": 100, "thicknessT1": 6, "thicknessT2": 10, "radius": 8})
        self.assertFalse(pending)
        self.assertEqual(provenance["source"]["pdf_page"], 17)

    def test_channel20b_uses_standard_bim_flange_without_changing_analysis(self):
        library = SectionLibrary(str(ROOT / "src/data/steel_sections.json"))
        section = library.get_section(SectionType.CHANNEL, 14)
        before = copy.deepcopy(vars(section))
        family, params, pending, provenance = _section(section, MappingConfig())
        self.assertEqual(family, "channel")
        self.assertFalse(pending)
        self.assertEqual(params, {"height": 200, "legWidthB": 75, "thicknessD": 9, "thicknessT": 11, "radiusR": 11, "legArcRadiusR1": 5.5})
        self.assertEqual(vars(section), before)
        self.assertAlmostEqual(section.tf * 1000, 9)
        difference = provenance["analysis_differences"][0]
        self.assertEqual(difference["field"], "thicknessT")
        self.assertAlmostEqual(difference["analysis_mm"], 9)
        self.assertEqual(difference["bim_mm"], 11)

    def test_unknown_numeric_size_is_not_matched_by_name(self):
        section = HShapeSection(0, "H-150x100.0", .003, 1e-5, 1e-5, 1e-6, H=.151, B=.1, tw=.006, tf=.01)
        self.assertIsNone(find_section_geometry(section))

    def test_square_tube_reference_radius_requires_explicit_policy(self):
        library = SectionLibrary(str(ROOT / "src/data/steel_sections.json"))
        for index in range(library.section_counts[SectionType.BOX]):
            section = library.get_section(SectionType.BOX, index)
            self.assertIsNotNone(find_section_geometry(section))
            _, params, pending, _ = _section(section, MappingConfig())
            self.assertNotIn("outerArcRadiusR", params)
            self.assertTrue(any("outerArcRadiusR" in item for item in pending))

    def test_square_tube_reference_policy_preserves_actual_and_reference_distinction(self):
        library = SectionLibrary(str(ROOT / "src/data/steel_sections.json"))
        section = library.get_section(SectionType.BOX, 2)
        before = copy.deepcopy(vars(section))
        _, params, pending, provenance = _section(section, MappingConfig(use_box_reference_radius=True))
        self.assertEqual(params["outerArcRadiusR"], 9)
        self.assertFalse(pending)
        self.assertEqual(provenance["radius_upper_bound_mm"], 12)
        self.assertFalse(provenance["listed_dimensions"])
        self.assertIn("not measured", provenance["radius_basis"])
        self.assertAlmostEqual(provenance["reference_area_cm2"], 33.941952)
        self.assertEqual(vars(section), before)

    def test_square_tube_reference_is_within_standard_limit(self):
        for row in geometry_catalog()["sections"]:
            if row["family"] == "box":
                self.assertLessEqual(row["reference_radius_mm"], row["radius_upper_bound_mm"])

    def test_explicit_overrides_are_distinguished_from_standard_values(self):
        library = SectionLibrary(str(ROOT / "src/data/steel_sections.json"))
        section = library.get_section(SectionType.H_MAIN, 8)
        cfg = MappingConfig(section_rules={section.name: {"sectionParams": {"radius": 9}}})
        _, params, _, provenance = _section(section, cfg)
        self.assertEqual(params["radius"], 9)
        self.assertEqual(provenance["explicit_overrides"], {"radius": 9})
        self.assertEqual(find_section_geometry(section)["sectionParams"]["radius"], 8)


if __name__ == "__main__":
    unittest.main()
