from src.structure.layout.engineering_layout import EngineeringLayoutEngine
from src.structure.layout.params import EngineeringLayoutParameters, ResearchLayoutParameters
from src.structure.layout.research_layout import ResearchLayoutEngine


def create_layout_engine(params):
    if isinstance(params, ResearchLayoutParameters):
        return ResearchLayoutEngine(params)

    if isinstance(params, EngineeringLayoutParameters):
        return EngineeringLayoutEngine(params)

    strategy = getattr(params, "layout_strategy", None)
    if strategy == "research":
        return ResearchLayoutEngine(params)
    if strategy == "engineering":
        return EngineeringLayoutEngine(params)

    raise ValueError(f"Unknown layout strategy: {strategy}")
