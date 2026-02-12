# Continuous Piecewise Linear (CPWL) Fitting

This project provides a robust Python environment for fitting optimal Continuous Piecewise Linear (CPWL) functions to datasets in general dimensions ($n$D).

The core script formulates the fitting problem as a Mixed-Integer Linear Programming (MILP) model, allowing for precise optimization using Google's **OR-Tools**. It supports a variety of industry-standard solvers, including **GUROBI**, **HiGHS**, and **SCIP**.

Additionally, the environment includes built-in tools to visualize the resulting CPWL functions for 2D and 3D datasets.

## Prerequisites

* Python 3.10 or higher
* Git

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/quentinplsrd/cpwl-nd-optimization.git](https://github.com/quentinplsrd/cpwl-nd-optimization.git)
    cd cpwl-nd-optimization
    ```

2.  **Create a virtual environment:**
    * **Windows:**
        ```bash
        python -m venv .venv
        ```
    * **Mac/Linux:**
        ```bash
        python3 -m venv .venv
        ```

3.  **Activate the environment:**
    * **Windows:** `.venv\Scripts\activate`
    * **Mac/Linux:** `source .venv/bin/activate`

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the main analysis script:

```bash
python src/main_script.py
