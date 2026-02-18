# Continuous Piecewise Linear (CPWL) Fitting

![Python](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

This project provides a robust Python environment for fitting optimal Continuous Piecewise Linear (CPWL) functions to datasets in general dimensions ($n$D).

The core script formulates the fitting problem as a Mixed-Integer Linear Programming (MILP) model, allowing for precise optimization using Google's **OR-Tools**. It supports a variety of industry-standard solvers, including **GUROBI**, **HiGHS**, and **SCIP**.

Additionally, the environment includes built-in tools to visualize the resulting CPWL functions for 2D and 3D datasets.

## Prerequisites

* [Python](https://www.python.org/downloads/) >=3.10
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [Git](https://git-scm.com/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/quentinplsrd/cpwl-nd-optimization.git
   cd cpwl-nd-optimization
   ```
2. **Create a virtual environment:**

   * **Windows:**
     ```bash
     uv venv .venv
     source .venv\Scripts\activate
     uv sync
     ```
   * **Mac/Linux:**
     ```bash
     uv venv .venv
     source .venv/bin/activate
     uv sync
     ```

## Usage

To run the main analysis script:

```bash
cd scripts
uv run run_case_studies.py
```

## Citation:

If you have used this code for research purposes, you can cite our publication by:

[Quentin Ploussard, Xiang Li, Matija Pavičević (2026) Tightening the Difference-of-Convex Formulation for the Piecewise Linear Approximation in General Dimensions. INFORMS Journal on Optimization 0(0). https://doi.org/10.1287/ijoo.2025.0074](https://pubsonline.informs.org/doi/full/10.1287/ijoo.2025.0074)

BibTex:

```
@article{ploussardDoC2025,
	title = {Tightening the Difference}-of-{Convex} {Formulation} for the {Piecewise} {Linear} {Approximation} in {General} {Dimensions}},
	issn = {2575-1484, 2575-1492},
	doi = {10.1287/ijoo.2025.0074},
	journal = {INFORMS Journal on Optimization},
	author = {Ploussard, Quentin and Li, Xiang and Pavičević, Matija},
	month = dec,
	year = {2025},
	pages = {ijoo.2025.0074},
}

```
