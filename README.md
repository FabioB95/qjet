# QJET — Constraint-Based Hamiltonian Modeling of Black Hole Jet Cores

QJET is a research-oriented Python framework for modeling the large-scale magnetic structure of relativistic black hole jet cores using a **constraint-based Hamiltonian formulation** on a Kerr background.

Rather than forward-evolving the GRMHD equations, QJET defines an energy functional whose low-energy states correspond to **stationary, magnetically dominated jet configurations** satisfying physical and geometrical constraints.  
The framework is designed to explore *optimal jet-core equilibria* and to provide a complementary tool to time-dependent GRMHD simulations.

---

## Scientific Motivation

Relativistic jets from spinning black holes exhibit remarkably stable, quasi-parabolic core geometries over large spatial scales.  
Standard GRMHD simulations describe their dynamical evolution, but they are not optimized to answer a different question:

> *Which magnetic configurations are intrinsically stable and energetically optimal for jet launching?*

QJET addresses this question by reframing jet modeling as a **variational ground-state problem**, rather than a time-integration problem.

---

## Core Idea

The jet core is represented as a lattice of unit magnetic direction vectors embedded in a Kerr spacetime.  
A Hamiltonian functional is constructed to encode the competition between multiple physical effects:

- local magnetic alignment (smoothness),
- frame-dragging–induced axial bias,
- suppression of excessive twist and current-driven instabilities,
- divergence control (solenoidality),
- phenomenological jet-power scaling inspired by Blandford–Znajek theory.

Stable jet configurations are obtained by minimizing this Hamiltonian under explicit constraints.

---

## Features

- Hamiltonian formulation of jet-core magnetic structure  
- Kerr-background–aware lattice geometry  
- Constraint enforcement via augmented Lagrangian methods  
- Designed for stationary, magnetically dominated jet backbones  
- Modular structure for testing alternative physical terms  

---

src/
│
├── tools/ # Optimization utilities, calibration and helpers
├── rt/ # Core routines for lattice construction and Hamiltonian terms
├── *.py # Main modules and entry points
└── README.md



Only source code is included in this repository.  
Large datasets, simulation outputs, and observational files are intentionally excluded.

---

## Installation

Create a clean Python environment (recommended):

```bash
conda create -n qjet python=3.10
conda activate qjet


pip install numpy scipy matplotlib


## Repository Structure

