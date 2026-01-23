# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intelligent design system for pressurized modular building structures (增压建筑智能设计系统). The system combines parametric modeling, finite element analysis (FEA) using OpenSees, and reinforcement learning-based structural optimization.

**Key Technologies:**
- **Structural Analysis**: OpenSeesPy for FEA
- **Optimization**: DQN (Deep Q-Network) reinforcement learning for steel section optimization
- **UI**: Streamlit-based web interface
- **Visualization**: PyVista, Plotly for 3D structural visualization

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run DQN training
python -m src.reinforcement.dqn.train

### Common Development Tasks
```bash
# Quick sanity check for DQN environment
python src/reinforcement/dqn/sanity_check.py

```

## Code Architecture

### Module Structure

```
src/
├── structure/          # Core structural modeling
│   ├── structural_model.py    # Main model: StructuralModel, Beam, Node classes
│   ├── section_library.py     # Steel section properties database
│   ├── mesh_generator.py      # Mesh generation for plates
│   └── enums.py               # BeamType, SectionType enums
│
├── analysis/           # FEA engine
│   ├── opensees_analysis.py   # OpenSees wrapper, multi-load combination analysis
│   ├── post_processor.py      # Results extraction and processing
│   └── unit.py                # Unit conversions (m→mm, kPa→MPa, etc.)
│
├── reinforcement/      # RL optimization
│   └── dqn/
│       ├── env.py             # Gym environment for steel section optimization
│       ├── agent.py           # DQN agent implementation
│       ├── train.py           # Training loop with epsilon decay
│       └── metric.py          # EpisodeStats, MetricsLogger
│
└── utils/
    ├── logger.py              # Logging utilities
    ├── visualize.py           # 3D visualization helpers
    └── constants.py           # Physical constants
```

### Key Design Patterns

**1. Structural Model Flow**
```
StructuralParameters → StructuralModel.build() → Beam/Node graph → OpenSeesAnalyzer.build_model()
```
- `StructuralModel` is the central data structure holding all beams, nodes, plates
- `StructuralParameters` (dataclass) defines geometry: length, width, height, beam spacing, section indices
- Beam types: `FRAME`, `MAIN_BEAM`, `SECONDARY_BEAM`, `BRACE` (see `BeamType` enum)

**2. OpenSees Analysis Pipeline**
```
OpenSeesAnalyzer.build_model() → analyze() → get_analysis_results() → CombinedResults
```
- **Unit conversions critical**: OpenSees uses mm, N, MPa internally (see `unit.py`)
- Load combinations: 1.0×Dead + 1.0×Internal_Pressure, 1.2×Dead + 1.4×Live + Internal_Pressure
- Results stored per load case, then combined using envelope method
- `CombinedResults.evaluate()` returns normalized metrics: mass_norm, u_sigma, u_delta (displacement ratio violations)

**3. RL Environment Design**
- **Observation**: Normalized structural metrics (mass, max stress, max displacement ratio)
- **Action**: Discrete selection of beam section indices (frame, main_beam, secondary_beam, brace)
- **Reward**: Based on mass reduction and constraint satisfaction (stress < 355 MPa, displacement ratio limits)
- **Episode termination**: Agent chooses "done" action or hits max steps
- **Action masking**: Prevents selecting invalid/unchanged sections

### Important Implementation Details

**Section Library (`section_library.py`)**
- Steel sections loaded from `src/data/steel_sections_large.json`
- Organized by `SectionType`: H_BEAM, RECT_TUBE, ROUND_TUBE
- Properties: area, Iy, Iz, J (torsion), weight_per_meter
- Indexed access: `SectionLibrary().get_section(SectionType.H_BEAM, index=5)`

**Displacement Ratio Constraint**
- Critical for structural safety
- Calculated as: `max_displacement / span_length`
- Different limits for different beam types (typically L/250 for beams)
- Span length = beam element length (not total structure width)

**DQN Training Specifics**
- Epsilon-greedy exploration: starts at 1.0, decays to 0.05 over 80% of training
- Replay buffer: capacity 50,000
- Training starts after 10+ samples in buffer
- Checkpoints saved every N episodes to `results/dqn/{timestamp}/`
- Metrics logged to CSV: return, mass_norm, feasibility, steps_to_feasible

## Notes

- do not read image/json files to avoid excessive token usage
- focus mainly on Python code files under `src/`