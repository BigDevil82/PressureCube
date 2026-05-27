from src.structure.section_grouping.application import assign_beam_groups, section_counts_for_groups
from src.structure.section_grouping.geometric import assign_geometric_beam_groups
from src.structure.section_grouping.models import BeamGroupPrototype, SectionGroupingRuleset

__all__ = [
    "BeamGroupPrototype",
    "SectionGroupingDiscoveryConfig",
    "SectionGroupingRuleset",
    "analyze_representative_designs",
    "assign_beam_groups",
    "assign_geometric_beam_groups",
    "build_rules_from_designs",
    "build_rules_from_random_topology_samples",
    "sample_representative_designs",
    "section_counts_for_groups",
]


def __getattr__(name):
    discovery_names = {
        "SectionGroupingDiscoveryConfig",
        "analyze_representative_designs",
        "build_rules_from_designs",
        "build_rules_from_random_topology_samples",
        "sample_representative_designs",
    }
    if name in discovery_names:
        from src.structure.section_grouping import discovery

        return getattr(discovery, name)
    raise AttributeError(name)
