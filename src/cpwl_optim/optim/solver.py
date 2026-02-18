# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""


from typing import Optional, Literal
import numpy as np
from datetime import timedelta
from ortools.math_opt.python import mathopt, model_parameters


# Functions


def solve_CPWL_model(
    data: np.ndarray,
    max_error: float,
    N_plus: int,
    N_minus: int,
    objective: Literal[
        "max error", "average error", "pieces of f", "pieces of f+", "pieces of f-"
    ] = "max error",
    fix_first_affine_piece: bool = False,
    sort_affine_pieces: bool = False,
    impose_d_plus_1_points_per_piece: Literal["f+ and f-", "f", "no"] = "no",
    big_M_constraint: Literal["indicator", "default", "tight"] = "default",
    bounded_variables: bool = False,
    default_big_M: Optional[float] = 1e5,
    tight_parameters: Optional[dict] = None,
    solver: Literal["GUROBI", "HIGHS", "SCIP"] = "HIGHS",
    enable_output: bool = True,
    relative_gap_tolerance: float = 1e-4,
    integer_feasibility_tolerance: float = 1e-9,
    time_limit_seconds: int = 120,
    solution_hint: Optional[dict] = None,
    solution_pool_size: Optional[int] = None,
):
    """Solve the continuous piecewise linear (CPWL) approximation problem of a data set in general dimension.

    Args:
      data: The data set to approximate.

      max_error: The maximum approximation error.

      N_plus: The number of pieces of the convex CPWL component f^+.

      N_minus: The number of pieces of the concave CPWL component f^-.

      objective: The objective function to minimize.

      fix_first_affine_piece: Indicates whether the first affine piece of f^- should be fixed to 0.

      sort_affine_pieces: Indicates whether the affine pieces of f^+ and f^- should be sorted. This serves as a symmetry-breaking constraint.

      impose_d_plus_1_points_per_piece: Indicates whether d+1 points should be imposed for each affine piece of f^+ and f^-, for each affine piece of f, or for no affine piece.

      big_M_constraint: Indicates whether the big-M constraint should use the default big-M parameter, a tight big-M parameter, or be modeled by an indicator constraint.

      bounded_variables: Indicates whether the variables should be bounded.

      default_big_M: Value of the default big-M parameter.

      tight_parameters: Dictionary of parameter values to tighten the problem.

      solver: Name of the MILP solver to use.

      enable_output: If the solver should print out its log messages.

      relative_gap_tolerance: The relative optimality tolerance for the MILP.

      integer_feasibility_tolerance: The integer feasibility tolerance.
      It is recommended to have a low integer feasibility tolerance (1e-9) due to the relatively large big-M parameter.

      time_limit_seconds: Time limit to solve the MILP problem, in seconds.

      solution_hint: Possible MILP solution to improve solution time.

      solution_pool_size: Number of feasible MILP solutions to return.

    Returns:
      The model, a dictionary of the variables, and the solver results.
    """

    solver_map = {
        "SCIP": mathopt.SolverType.GSCIP,
        "HIGHS": mathopt.SolverType.HIGHS,
        "GUROBI": mathopt.SolverType.GUROBI,
    }
    objective_set = {
        "max error",
        "average error",
        "pieces of f",
        "pieces of f+",
        "pieces of f-",
    }
    impose_d_plus_1_points_per_piece_set = {"no", "f+ and f-", "f"}
    big_M_constraint_set = {"indicator", "default", "tight"}

    # --- Type checks ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {data.shape}.")
    if not isinstance(max_error, (float, np.floating)):
        raise TypeError(f"`max_error` must be a float, got {type(max_error).__name__}.")
    if not isinstance(N_plus, int) or not isinstance(N_minus, int):
        raise TypeError("`N_plus` and `N_minus` must be integers.")
    if not isinstance(objective, str):
        raise TypeError("`objective` must be a string.")
    if not isinstance(fix_first_affine_piece, bool):
        raise TypeError("`fix_first_affine_piece` must be a boolean.")
    if not isinstance(sort_affine_pieces, bool):
        raise TypeError("`sort_affine_pieces` must be a boolean.")
    if impose_d_plus_1_points_per_piece is not None:
        if not isinstance(impose_d_plus_1_points_per_piece, str):
            raise TypeError("`impose_d_plus_1_points_per_piece` must be a string.")
    if not isinstance(big_M_constraint, str):
        raise TypeError("`big_M_constraint` must be a string.")
    if not isinstance(bounded_variables, bool):
        raise TypeError("`bounded_variables` must be a boolean.")
    if not isinstance(default_big_M, (float, np.floating)):
        raise TypeError(
            f"`default_big_M` must be a float, got {type(default_big_M).__name__}."
        )
    if tight_parameters is not None:
        if not isinstance(tight_parameters, dict):
            raise TypeError(
                f"`tight_parameters` must be a dictionary, got {type(tight_parameters).__name__}."
            )
    if not isinstance(solver, str):
        raise TypeError("`solver` must be a string.")
    if not isinstance(enable_output, bool):
        raise TypeError("`enable_output` must be a boolean.")
    if not isinstance(relative_gap_tolerance, (float, np.floating)):
        raise TypeError(
            f"`relative_gap_tolerance` must be a float, got {type(relative_gap_tolerance).__name__}."
        )
    if not isinstance(time_limit_seconds, int):
        raise TypeError(
            f"`time_limit_seconds` must be a float, got {type(time_limit_seconds).__name__}."
        )
    if solution_hint is not None:
        if not isinstance(solution_hint, dict):
            raise TypeError(
                f"`solution_hint` must be a dictionary, got {type(solution_hint).__name__}."
            )
    if solution_pool_size is not None:
        if not isinstance(solution_pool_size, int):
            raise TypeError("`solution_pool_size` must be an integer.")

    # --- Value checks ---
    N_points, d_plus_1 = data.shape
    d = d_plus_1 - 1

    if d < 1:
        raise ValueError(
            f"Invalid dimension {d+1}: "
            f"the data set should have at least 1 input and 1 output dimension."
        )
    if N_points < d + 1:
        raise ValueError(
            f"Invalid data shape {data.shape}: "
            f"number of points ({N_points}) must be >= dimension ({d+1})."
        )
    if max_error < 0:
        raise ValueError("`max_error` must be nonnegative.")
    if N_plus < 1 or N_minus < 1:
        raise ValueError("`N_plus` and `N_minus` must be >= 1.")
    if objective not in objective_set:
        supported = ", ".join(objective_set)
        raise ValueError(
            "The objective {objective} is not supported. "
            f"Please use one of the following: {supported}."
        )
    if impose_d_plus_1_points_per_piece not in impose_d_plus_1_points_per_piece_set:
        supported = ", ".join(impose_d_plus_1_points_per_piece_set)
        raise ValueError(
            "The constraint {impose_d_plus_1_points_per_piece} is not supported. "
            f"Please use one of the following: {supported}."
        )
    if big_M_constraint not in big_M_constraint_set:
        supported = ", ".join(big_M_constraint_set)
        raise ValueError(
            "The big-M constraint {big_M_constraint} is not supported. "
            f"Please use one of the following: {supported}."
        )
    if default_big_M is not None:
        if default_big_M <= 0:
            raise ValueError("`default_big_M` must be positive.")
    else:
        if big_M_constraint == "default":
            raise ValueError("You must provide a value for `default_big_M`.")
    if solver not in solver_map:
        supported = ", ".join(solver_map.keys())
        raise ValueError(
            f"The solver '{solver}' is not a supported MathOpt MILP solver. "
            f"Please use one of the following: {supported}."
        )
    if (big_M_constraint == "indicator") and (solver != "GUROBI"):
        raise ValueError("The indicator constraint is only implemented in Gurobi. ")
    if relative_gap_tolerance < 0:
        raise ValueError("`relative_gap_tolerance` must be nonnegative.")
    if time_limit_seconds <= 0:
        raise ValueError("`time_limit_seconds` must be positive.")
    if solution_pool_size is not None:
        if solver not in {"SCIP", "GUROBI"}:
            raise ValueError(
                "To return multiple feasible solutions, you must use the SCIP or Gurobi solver. "
            )
        if solution_pool_size < 1:
            raise ValueError("`solution_pool_size` must be >= 1.")
    if tight_parameters is None:
        if (big_M_constraint == "tight") or bounded_variables:
            raise ValueError("You must provide a valid `tight_parameters`. ")
    if solution_hint is not None:
        if solver not in {"GUROBI", "SCIP"}:
            print(
                f"Warning: The solver {solver} does not support warm-start, the solution hint was discarded."
            )

    # --- Build model ---
    x = data[:, :-1]
    z = data[:, -1]
    piecePlus = range(N_plus)
    pieceMinus = range(N_minus)
    points = range(N_points)
    dims = range(d)

    model = mathopt.Model(name="CPWL")

    # --- Create variables ---
    zPWL = [model.add_variable(name=f"zPWL({i})") for i in points]
    zPlus = [model.add_variable(name=f"zPlus({i})") for i in points]
    zMinus = [model.add_variable(name=f"zMinus({i})") for i in points]
    aPlus = [
        [model.add_variable(name=f"aPlus({j},{r})") for r in dims] for j in piecePlus
    ]
    bPlus = [model.add_variable(name=f"bPlus({j})") for j in piecePlus]
    aMinus = [
        [model.add_variable(name=f"aMinus({j},{r})") for r in dims] for j in pieceMinus
    ]
    bMinus = [model.add_variable(name=f"bMinus({j})") for j in pieceMinus]
    decPlus = [
        [model.add_binary_variable(name=f"decPlus({i},{j})") for j in piecePlus]
        for i in points
    ]
    decMinus = [
        [model.add_binary_variable(name=f"decMinus({i},{j})") for j in pieceMinus]
        for i in points
    ]
    error = [model.add_variable(lb=0.0, name=f"error({i})") for i in points]
    dec = None
    gamma = None
    alphaPlus = None
    alphaMinus = None
    if impose_d_plus_1_points_per_piece == "f":
        dec = [
            [
                [
                    model.add_variable(lb=0.0, ub=1.0, name=f"dec({i},{j},{k})")
                    for k in pieceMinus
                ]
                for j in piecePlus
            ]
            for i in points
        ]
        gamma = [
            [
                model.add_variable(lb=0.0, ub=1.0, name=f"gamma({j},{k})")
                for k in pieceMinus
            ]
            for j in piecePlus
        ]
    if objective == "pieces of f+":
        alphaPlus = [
            model.add_variable(lb=0.0, ub=1.0, name=f"alphaPlus({j})")
            for j in piecePlus
        ]
    if objective == "pieces of f-":
        alphaMinus = [
            model.add_variable(lb=0.0, ub=1.0, name=f"alphaMinus({j})")
            for j in pieceMinus
        ]
    obj1 = model.add_variable(lb=0.0, name="obj1")
    obj2 = model.add_variable(lb=0.0, name="obj2")

    coeff1 = (objective in {"pieces of f+", "pieces of f-", "pieces of f"}) * 1
    coeff2 = (
        (coeff1 * (0.5 / max_error) + (1 - coeff1))
        * (objective in {"max error", "average error"})
        * 1
    )

    # --- Objective function ---
    model.minimize(coeff1 * obj1 + coeff2 * obj2)

    # --- Constraints ---
    if objective == "average error":
        model.add_linear_constraint(N_points * obj2 >= sum(error[i] for i in points))
    elif objective == "max error":
        for i in points:
            model.add_linear_constraint(obj2 >= error[i])

    if objective == "pieces of f":
        model.add_linear_constraint(
            obj1 >= sum(gamma[j][k] for j in piecePlus for k in pieceMinus)
        )
    elif objective == "pieces of f+":
        model.add_linear_constraint(obj1 >= sum(alphaPlus[j] for j in piecePlus))
    elif objective == "pieces of f-":
        model.add_linear_constraint(obj1 >= sum(alphaMinus[k] for k in pieceMinus))

    for i in points:
        model.add_linear_constraint(zPWL[i] - z[i] <= error[i], name=f"errorPlus({i})")
        model.add_linear_constraint(
            zPWL[i] - z[i] >= -error[i], name=f"errorMinus({i})"
        )
        model.add_linear_constraint(zPWL[i] == zPlus[i] - zMinus[i], name=f"zDC({i})")
        for j in piecePlus:
            model.add_linear_constraint(
                zPlus[i] - sum(aPlus[j][r] * x[i, r] for r in dims) - bPlus[j] >= 0,
                name=f"convexPlus({i,j})",
            )
        for k in pieceMinus:
            model.add_linear_constraint(
                zMinus[i] - sum(aMinus[k][r] * x[i, r] for r in dims) - bMinus[k] >= 0,
                name=f"convexMinus({i,k})",
            )
        model.add_linear_constraint(
            sum(decPlus[i][j] for j in piecePlus) >= 1, name=f"atLeastOnePlus({i})"
        )
        model.add_linear_constraint(
            sum(decMinus[i][k] for k in pieceMinus) >= 1, name=f"atLeastOneMinus({i})"
        )

    if big_M_constraint == "indicator":
        for i in points:
            for j in piecePlus:
                model.add_indicator_constraint(
                    indicator=decPlus[i][j],
                    activate_on_zero=False,
                    implied_constraint=(
                        zPlus[i] - sum([aPlus[j][r] * x[i, r] for r in dims]) - bPlus[j]
                        <= 0.0
                    ),
                    name=f"indicatorPlus({i,j})",
                )
            for k in pieceMinus:
                model.add_indicator_constraint(
                    indicator=decMinus[i][k],
                    activate_on_zero=False,
                    implied_constraint=(
                        zMinus[i]
                        - sum([aMinus[k][r] * x[i, r] for r in dims])
                        - bMinus[k]
                        <= 0.0
                    ),
                    name=f"indicatorMinus({i,k})",
                )
    else:
        bigMnPlus = default_big_M * np.ones(N_points)
        bigMnMinus = default_big_M * np.ones(N_points)
        if big_M_constraint == "tight":
            bigMnPlus = min(N_plus - 1, N_minus) * (
                tight_parameters["max_fx_n"] - tight_parameters["min_fx_n"]
            )
            bigMnMinus = min(N_minus - 1, N_plus) * (
                tight_parameters["max_fx_n"] - tight_parameters["min_fx_n"]
            )
        for i in points:
            for j in piecePlus:
                model.add_linear_constraint(
                    zPlus[i] - sum([aPlus[j][r] * x[i, r] for r in dims]) - bPlus[j]
                    <= bigMnPlus[i] * (1 - decPlus[i][j]),
                    name=f"bigMnPlus({i,j})",
                )
            for k in pieceMinus:
                model.add_linear_constraint(
                    zMinus[i] - sum([aMinus[k][r] * x[i, r] for r in dims]) - bMinus[k]
                    <= bigMnMinus[i] * (1 - decMinus[i][k]),
                    name=f"bigMnMinus({i,k})",
                )

    if sort_affine_pieces:
        for j in piecePlus[1:]:
            model.add_linear_constraint(
                aPlus[j][0] >= aPlus[j - 1][0], name=f"sortPlusPieces({j})"
            )
        for k in pieceMinus[1:]:
            model.add_linear_constraint(
                aMinus[k][0] >= aMinus[k - 1][0], name=f"sortMinusPieces({k})"
            )

    if impose_d_plus_1_points_per_piece == "f+ and f-":
        for j in piecePlus:
            model.add_linear_constraint(sum(decPlus[i][j] for i in points) >= d + 1)
        for k in pieceMinus:
            model.add_linear_constraint(sum(decMinus[i][k] for i in points) >= d + 1)

    if impose_d_plus_1_points_per_piece == "f":
        for j in piecePlus:
            for k in pieceMinus:
                for i in points:
                    model.add_linear_constraint(dec[i][j][k] <= decPlus[i][j])
                    model.add_linear_constraint(dec[i][j][k] <= decMinus[i][k])
                    model.add_linear_constraint(
                        dec[i][j][k] >= decPlus[i][j] + decMinus[i][k] - 1
                    )
                    model.add_linear_constraint(gamma[j][k] >= dec[i][j][k])
                model.add_linear_constraint(
                    gamma[j][k] * (d + 1) <= sum(dec[i][j][k] for i in points)
                )

    if bounded_variables:
        for i in points:
            error[i].lower_bound = 0
            error[i].upper_bound = max_error
            zPWL[i].lower_bound = z[i] - max_error
            zPWL[i].upper_bound = z[i] + max_error
            zMinus[i].lower_bound = 0
            zMinus[i].upper_bound = bigMnMinus[i]
            zPlus[i].lower_bound = z[i] - max_error
            zPlus[i].upper_bound = z[i] + max_error + bigMnMinus[i]
        for k in pieceMinus:
            for r in dims:
                aMinus[k][r].lower_bound = -min(N_minus - 1, N_plus) * (
                    tight_parameters["max_a_b"][r] - tight_parameters["min_a_b"][r]
                )
                aMinus[k][r].upper_bound = min(N_minus - 1, N_plus) * (
                    tight_parameters["max_a_b"][r] - tight_parameters["min_a_b"][r]
                )
            aMinus[k][0].lower_bound = 0
            bMinus[k].lower_bound = -min(N_minus - 1, N_plus) * (
                tight_parameters["max_a_b"][-1] - tight_parameters["min_a_b"][-1]
            )
            bMinus[k].upper_bound = min(N_minus - 1, N_plus) * (
                tight_parameters["max_a_b"][-1] - tight_parameters["min_a_b"][-1]
            )
        for j in piecePlus:
            for r in dims:
                aPlus[j][r].lower_bound = tight_parameters["min_a_b"][r] - min(
                    N_minus - 1, N_plus
                ) * (tight_parameters["max_a_b"][r] - tight_parameters["min_a_b"][r])
                aPlus[j][r].upper_bound = tight_parameters["max_a_b"][r] + min(
                    N_minus - 1, N_plus
                ) * (tight_parameters["max_a_b"][r] - tight_parameters["min_a_b"][r])
            aPlus[j][0].lower_bound = tight_parameters["min_a_b"][0]
            bPlus[j].lower_bound = tight_parameters["min_a_b"][-1] - min(
                N_minus - 1, N_plus
            ) * (tight_parameters["max_a_b"][-1] - tight_parameters["min_a_b"][-1])
            bPlus[j].upper_bound = tight_parameters["max_a_b"][-1] + min(
                N_minus - 1, N_plus
            ) * (tight_parameters["max_a_b"][-1] - tight_parameters["min_a_b"][-1])

    if fix_first_affine_piece:
        for r in dims:
            aMinus[0][r].lower_bound = 0
            aMinus[0][r].upper_bound = 0
        bMinus[0].lower_bound = 0
        bMinus[0].upper_bound = 0

    # --- Use solution hint ---
    model_params = None
    if solution_hint is not None:
        solution_hint_dict = {}
        for i in points:
            solution_hint_dict[zPWL[i]] = solution_hint["zPWL"][i]
            solution_hint_dict[zPlus[i]] = solution_hint["zPlus"][i]
            solution_hint_dict[zMinus[i]] = solution_hint["zMinus"][i]
            solution_hint_dict[error[i]] = solution_hint["error"][i]
        for j in piecePlus:
            solution_hint_dict[bPlus[j]] = solution_hint["bPlus"][j]
            for r in dims:
                solution_hint_dict[aPlus[j][r]] = solution_hint["aPlus"][j, r]
            for i in points:
                solution_hint_dict[decPlus[i][j]] = solution_hint["decPlus"][i, j]
        for k in pieceMinus:
            solution_hint_dict[bMinus[k]] = solution_hint["bMinus"][k]
            for r in dims:
                solution_hint_dict[aMinus[k][r]] = solution_hint["aMinus"][k, r]
            for i in points:
                solution_hint_dict[decMinus[i][k]] = solution_hint["decMinus"][i, k]

        s_hint = model_parameters.SolutionHint(variable_values=solution_hint_dict)
        model_params = model_parameters.ModelSolveParameters(solution_hints=[s_hint])

    params = mathopt.SolveParameters(
        solution_pool_size=solution_pool_size,
        enable_output=enable_output,
        relative_gap_tolerance=relative_gap_tolerance,
        time_limit=timedelta(seconds=time_limit_seconds),
    )

    if solver == "GUROBI":
        params.gurobi.param_values["IntFeasTol"] = str(integer_feasibility_tolerance)
        params.gurobi.param_values["FeasibilityTol"] = str(
            integer_feasibility_tolerance
        )
    elif solver == "HIGHS":
        params.highs.double_options["mip_feasibility_tolerance"] = (
            integer_feasibility_tolerance
        )
        params.highs.double_options["primal_feasibility_tolerance"] = (
            integer_feasibility_tolerance
        )
        params.highs.double_options["dual_feasibility_tolerance"] = (
            integer_feasibility_tolerance
        )
    elif solver == "SCIP":
        params.gscip.real_params["numerics/feastol"] = integer_feasibility_tolerance
        params.gscip.real_params["numerics/dualfeastol"] = integer_feasibility_tolerance

    # --- Solve the problem ---
    result = mathopt.solve(
        model, solver_map[solver], params=params, model_params=model_params
    )

    variables = {
        "zPWL": zPWL,
        "zMinus": zMinus,
        "zPlus": zPlus,
        "error": error,
        "aPlus": aPlus,
        "bPlus": bPlus,
        "aMinus": aMinus,
        "bMinus": bMinus,
        "decPlus": decPlus,
        "decMinus": decMinus,
        "dec": dec,
        "gamma": gamma,
        "alphaPlus": alphaPlus,
        "alphaMinus": alphaMinus,
    }

    return model, variables, result


def extract_values(
    variables: dict,
    result,
    data: np.ndarray = None,
    clean_values: bool = False,
):
    # if result.status() != mathopt.SolveStatus.OPTIMAL:
    #     print(
    #         f"Warning: The optimization problem was not solved to optimality. "
    #         f"Status: {result.status()}. "
    #         f"The extracted variable values may be suboptimal or infeasible."
    #     )

    N_plus = len(variables["bPlus"])
    N_minus = len(variables["bMinus"])
    N_points = len(variables["zPWL"])
    d = len(variables["aPlus"][0])

    aPlus = np.array(
        [
            [result.variable_values()[variables["aPlus"][j][r]] for r in range(d)]
            for j in range(N_plus)
        ]
    )
    aMinus = np.array(
        [
            [result.variable_values()[variables["aMinus"][k][r]] for r in range(d)]
            for k in range(N_minus)
        ]
    )
    bPlus = np.array(
        [result.variable_values()[variables["bPlus"][j]] for j in range(N_plus)]
    )
    bMinus = np.array(
        [result.variable_values()[variables["bMinus"][k]] for k in range(N_minus)]
    )
    decPlus = np.array(
        [
            [
                result.variable_values()[variables["decPlus"][i][j]]
                for j in range(N_plus)
            ]
            for i in range(N_points)
        ]
    )
    decMinus = np.array(
        [
            [
                result.variable_values()[variables["decMinus"][i][k]]
                for k in range(N_minus)
            ]
            for i in range(N_points)
        ]
    )
    zPWL = np.array(
        [result.variable_values()[variables["zPWL"][i]] for i in range(N_points)]
    )
    zPlus = np.array(
        [result.variable_values()[variables["zPlus"][i]] for i in range(N_points)]
    )
    zMinus = np.array(
        [result.variable_values()[variables["zMinus"][i]] for i in range(N_points)]
    )
    error = np.array(
        [result.variable_values()[variables["error"][i]] for i in range(N_points)]
    )

    if clean_values:

        x = data[:, :-1]
        z = data[:, -1]
        d = data.shape[1] - 1
        dims = range(d)

        # round the dec variables
        decPlus = np.round(decPlus)
        decMinus = np.round(decMinus)

        # adjust zPlus and zMinus based on rounded dec values
        for i in range(N_points):
            for j in range(N_plus):
                if decPlus[i, j] == 1.0:
                    zPlus[i] = sum([aPlus[j, r] * x[i, r] for r in dims]) + bPlus[j]
            for k in range(N_minus):
                if decMinus[i, k] == 1.0:
                    zMinus[i] = sum([aMinus[k, r] * x[i, r] for r in dims]) + bMinus[k]
        # correct zPWL and error
        zPWL = zPlus - zMinus
        error = abs(zPWL - z)

    variable_values = {
        "aPlus": aPlus,
        "aMinus": aMinus,
        "bPlus": bPlus,
        "bMinus": bMinus,
        "decPlus": decPlus,
        "decMinus": decMinus,
        "zPWL": zPWL,
        "zPlus": zPlus,
        "zMinus": zMinus,
        "error": error,
    }

    return variable_values
