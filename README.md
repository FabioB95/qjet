# QJET — Constraint-Based Hamiltonian Framework for Black Hole Jet Core Geometry

QJET is a stationary, constraint-based Hamiltonian formulation for modeling the smooth, magnetically dominated jet core on a Kerr background.  
Instead of forward-evolving GRMHD, QJET defines an energy functional whose low-energy states represent magnetically stable jet-core configurations under physically motivated ingredients and constraints.

**Target regime:** the slowly varying, quasi-parabolic backbone of the inner jet (not radiatively bright knots, shocks, or time-dependent variability).

---

## Main idea (one paragraph)

Magnetic field configurations are represented by unit-direction degrees of freedom on a Kerr-embedded lattice.  
We construct a Hamiltonian that encodes neighbor alignment, frame-dragging (axial bias), twist/current suppression, a Blandford–Znajek-inspired power target, and strict solenoidality control.  
The ground state is obtained via variational optimization on a constrained manifold, with augmented-Lagrangian enforcement of divergence and horizon-flux consistency.

---

## Repository layout

- `src/qjet/` core library (grid, mapping, Hamiltonian terms, constraints, optimizer, metrics)
- `src/fits_to_profile.py` extract transverse profiles from FITS using a DS9 axis region
- `src/external_benchmark.py` fit/calibrate parameters (e.g., L0), export `metrics.csv` and `overlay.png`
- `configs/` per-source configuration files (M87, 3C 273, …)
- `data/` raw and processed observational inputs
- `outputs/` figures and metrics produced by runs
- `paper/` LaTeX manuscript and figures

---

## Installation

### Option A: pip (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
