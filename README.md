\# Continuous Piecewise Linear (CPWL) Fitting



This project provides a robust Python environment for fitting optimal Continuous Piecewise Linear (CPWL) functions to datasets in general dimensions ($n$D).



The core script formulates the fitting problem as a Mixed-Integer Linear Programming (MILP) model, allowing for precise optimization using Google's \*\*OR-Tools\*\*. It supports a variety of industry-standard solvers, including \*\*GUROBI\*\*, \*\*HiGHS\*\*, and \*\*SCIP\*\*.



Additionally, the environment includes built-in tools to visualize the resulting CPWL functions for 2D and 3D datasets.



\## Prerequisites



\* Python 3.10 or higher

\* Git



\## Installation



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/your-username/cpwl-nd-optimization.git](https://github.com/your-username/cpwl-nd-optimization.git)

&nbsp;   cd cpwl-nd-optimization

&nbsp;   ```



2\.  \*\*Create a virtual environment:\*\*

&nbsp;   \* \*\*Windows:\*\*

&nbsp;       ```bash

&nbsp;       python -m venv .venv

&nbsp;       ```

&nbsp;   \* \*\*Mac/Linux:\*\*

&nbsp;       ```bash

&nbsp;       python3 -m venv .venv

&nbsp;       ```



3\.  \*\*Activate the environment:\*\*

&nbsp;   \* \*\*Windows:\*\* `.venv\\Scripts\\activate`

&nbsp;   \* \*\*Mac/Linux:\*\* `source .venv/bin/activate`



4\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\## Usage



To run the main analysis script:



```bash

python src/main\_script.py

