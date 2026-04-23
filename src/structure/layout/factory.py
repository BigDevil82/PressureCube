from src.structure.layout.engineering_layout import EngineeringLayoutEngine
from src.structure.layout.research_layout import ResearchLayoutEngine


def create_layout_engine(params):
    strategy = getattr(params, "layout_strategy", "research")

    if strategy == "research":
        return ResearchLayoutEngine(params)

    if strategy == "engineering":
        return EngineeringLayoutEngine(params)

    raise ValueError(f"Unknown layout strategy: {strategy}")
