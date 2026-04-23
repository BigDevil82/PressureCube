from src.structure.layout.research_layout import ResearchLayoutEngine


def create_layout_engine(params):
    strategy = getattr(params, "layout_strategy", "research")

    if strategy == "research":
        return ResearchLayoutEngine(params)

    if strategy == "engineering":
        raise NotImplementedError("Engineering layout engine is not implemented yet.")

    raise ValueError(f"Unknown layout strategy: {strategy}")
