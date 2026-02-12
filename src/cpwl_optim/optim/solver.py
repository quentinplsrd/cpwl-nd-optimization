# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["axes3d.mouserotationstyle"] = 'azel'
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection

from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linprog, linear_sum_assignment
from scipy.linalg import null_space

from itertools import product, combinations
import time

from typing import Optional, Literal
import numpy as np
import pandas as pd
from datetime import timedelta
from ortools.math_opt.python import mathopt, model_parameters

import os

# os.environ["GRB_LICENSE_FILE"] = r"C:\Users\qploussard\gurobi.lic"
# os.environ["GUROBI_HOME"] = r"C:\gurobi1203\win64"

# os.environ["GUROBI_HOME"] = r"C:\gurobi1300\win64"

#%% functions


def solve_CPWL_approximation(data: np.ndarray, max_error: float,
                             N_plus: int, N_minus: int,
                             objective: 
                                 Literal['max error','average error',
                                 'pieces of f','pieces of f+','pieces of f-'
                                 ] = 'max error',
                             fix_first_affine_piece: bool = False,
                             sort_affine_pieces: bool = False,
                             impose_d_plus_1_points_per_piece: 
                                 Literal['f+ and f-','f','no'] = 'no',
                             big_M_constraint: Literal[
                                 'indicator','default','tight'
                                 ] = 'default',
                             bounded_variables: bool = False,
                             default_big_M: Optional[float] = 1e5, 
                             tight_parameters: Optional[dict] = None,
                             solver: Literal['GUROBI','HIGHS','SCIP'] = 'HIGHS',
                             enable_output: bool = True,
                             relative_gap_tolerance: float = 1e-4,
                             integer_feasibility_tolerance: float = 1e-9,
                             time_limit_seconds: int = 120,
                             solution_hint: Optional[dict] = None,
                             solution_pool_size: Optional[int] = None):
    
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
        "SCIP":   mathopt.SolverType.GSCIP,
        "HIGHS":  mathopt.SolverType.HIGHS,
        "GUROBI": mathopt.SolverType.GUROBI
        }
    objective_set = {
        'max error',
        'average error',
        'pieces of f',
        'pieces of f+',
        'pieces of f-'
        }
    impose_d_plus_1_points_per_piece_set = {
        'no',
        'f+ and f-',
        'f'
        }
    big_M_constraint_set = {
        'indicator',
        'default',
        'tight'
        }
    
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
        raise TypeError(f"`default_big_M` must be a float, got {type(default_big_M).__name__}.")
    if tight_parameters is not None:
        if not isinstance(tight_parameters, dict):
            raise TypeError(f"`tight_parameters` must be a dictionary, got {type(tight_parameters).__name__}.")
    if not isinstance(solver, str):
        raise TypeError("`solver` must be a string.")
    if not isinstance(enable_output, bool):
        raise TypeError("`enable_output` must be a boolean.")
    if not isinstance(relative_gap_tolerance, (float, np.floating)):
        raise TypeError(f"`relative_gap_tolerance` must be a float, got {type(relative_gap_tolerance).__name__}.")
    if not isinstance(time_limit_seconds, int):
        raise TypeError(f"`time_limit_seconds` must be a float, got {type(time_limit_seconds).__name__}.")
    if solution_hint is not None:
        if not isinstance(solution_hint, dict):
            raise TypeError(f"`solution_hint` must be a dictionary, got {type(solution_hint).__name__}.")
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
        if big_M_constraint=='default':
            raise ValueError("You must provide a value for `default_big_M`.")
    if solver not in solver_map:
        supported = ", ".join(solver_map.keys())
        raise ValueError(
            f"The solver '{solver}' is not a supported MathOpt MILP solver. "
            f"Please use one of the following: {supported}."
        )
    if (big_M_constraint=='indicator') and (solver!='GUROBI'):
        raise ValueError(
            "The indicator constraint is only implemented in Gurobi. "
        )
    if relative_gap_tolerance < 0:
        raise ValueError("`relative_gap_tolerance` must be nonnegative.")
    if time_limit_seconds <= 0:
        raise ValueError("`time_limit_seconds` must be positive.")
    if solution_pool_size is not None:
        if solver not in {'SCIP','GUROBI'}:
            raise ValueError(
                "To return multiple feasible solutions, you must use the SCIP or Gurobi solver. "
                )
        if solution_pool_size < 1:
            raise ValueError("`solution_pool_size` must be >= 1.")
    if tight_parameters is None:
        if (big_M_constraint=='tight') or bounded_variables:
            raise ValueError(
                "You must provide a valid `tight_parameters`. "
                )
    if solution_hint is not None:
        if solver not in {'GUROBI','SCIP'}:
            print(f"Warning: The solver {solver} does not support warm-start, the solution hint was discarded.")
        
    
    # --- Build model ---
    x = data[:,:-1]
    z = data[:,-1]
    piecePlus = range(N_plus)
    pieceMinus = range(N_minus)
    points = range(N_points)
    dims = range(d)
        
    model = mathopt.Model(name='CPWL')
    
    # --- Create variables ---
    zPWL = [model.add_variable(name=f"zPWL({i})") for i in points]
    zPlus = [model.add_variable(name=f"zPlus({i})") for i in points]
    zMinus = [model.add_variable(name=f"zMinus({i})") for i in points]
    aPlus = [[model.add_variable(name=f"aPlus({j},{r})") for r in dims] for j in piecePlus]
    bPlus = [model.add_variable(name=f"bPlus({j})") for j in piecePlus]
    aMinus = [[model.add_variable(name=f"aMinus({j},{r})") for r in dims] for j in pieceMinus]
    bMinus = [model.add_variable(name=f"bMinus({j})") for j in pieceMinus]
    decPlus = [[model.add_binary_variable(name=f"decPlus({i},{j})") for j in piecePlus] for i in points]
    decMinus = [[model.add_binary_variable(name=f"decMinus({i},{j})") for j in pieceMinus] for i in points]
    error = [model.add_variable(lb=0.,name=f"error({i})") for i in points]
    dec=None
    gamma=None
    alphaPlus=None
    alphaMinus=None
    if impose_d_plus_1_points_per_piece=='f':
        dec = [[[model.add_variable(lb=0.0, ub=1.0, name=f"dec({i},{j},{k})") for k in pieceMinus] for j in piecePlus] for i in points]
        gamma = [[model.add_variable(lb=0.0, ub=1.0, name=f"gamma({j},{k})") for k in pieceMinus] for j in piecePlus]
    if objective == 'pieces of f+':
        alphaPlus = [model.add_variable(lb=0.0, ub=1.0, name=f"alphaPlus({j})") for j in piecePlus]
    if objective == 'pieces of f-':
        alphaMinus = [model.add_variable(lb=0.0, ub=1.0, name=f"alphaMinus({j})") for j in pieceMinus]
    obj1 = model.add_variable(lb=0.0, name="obj1")
    obj2 = model.add_variable(lb=0.0, name="obj2")

    coeff1 = (objective in {'pieces of f+','pieces of f-','pieces of f'})*1
    coeff2 = (coeff1*(0.5/max_error) + (1-coeff1))*(objective in {'max error','average error'})*1
    
    # --- Objective function ---
    model.minimize(coeff1*obj1 + coeff2*obj2)
    
    # --- Constraints ---
    if objective == 'average error':
        model.add_linear_constraint(N_points*obj2 >= sum(error[i] for i in points))
    elif objective == 'max error':
        for i in points:
            model.add_linear_constraint(obj2 >= error[i])
            
    if objective == 'pieces of f':
        model.add_linear_constraint(obj1 >= sum(gamma[j][k] for j in piecePlus for k in pieceMinus))
    elif objective == 'pieces of f+':
        model.add_linear_constraint(obj1 >= sum(alphaPlus[j] for j in piecePlus))
    elif objective == 'pieces of f-':
        model.add_linear_constraint(obj1 >= sum(alphaMinus[k] for k in pieceMinus))
        
    for i in points:
        model.add_linear_constraint(zPWL[i] - z[i] <= error[i], name=f"errorPlus({i})")
        model.add_linear_constraint(zPWL[i] - z[i] >= - error[i], name=f"errorMinus({i})")
        model.add_linear_constraint(zPWL[i] == zPlus[i] - zMinus[i], name=f"zDC({i})")
        for j in piecePlus:
            model.add_linear_constraint(zPlus[i] - sum(aPlus[j][r]*x[i,r] for r in dims) - bPlus[j] >= 0, name=f"convexPlus({i,j})")
        for k in pieceMinus:
            model.add_linear_constraint(zMinus[i] - sum(aMinus[k][r]*x[i,r] for r in dims) - bMinus[k] >= 0, name=f"convexMinus({i,k})")
        model.add_linear_constraint(sum(decPlus[i][j] for j in piecePlus) >= 1, name=f"atLeastOnePlus({i})")
        model.add_linear_constraint(sum(decMinus[i][k] for k in pieceMinus) >= 1, name=f"atLeastOneMinus({i})")
    
    if big_M_constraint=='indicator':
        for i in points:
            for j in piecePlus:
                model.add_indicator_constraint(indicator=decPlus[i][j],
                                               activate_on_zero=False,
                                               implied_constraint=(zPlus[i] - sum([aPlus[j][r]*x[i,r] for r in dims]) - bPlus[j] <= 0.), 
                                               name=f"indicatorPlus({i,j})")
            for k in pieceMinus:
                model.add_indicator_constraint(indicator=decMinus[i][k],
                                               activate_on_zero=False,
                                               implied_constraint=(zMinus[i] - sum([aMinus[k][r]*x[i,r] for r in dims]) - bMinus[k] <= 0.), 
                                               name=f"indicatorMinus({i,k})")
    else:
        bigMnPlus = default_big_M*np.ones(N_points)
        bigMnMinus = default_big_M*np.ones(N_points)
        if big_M_constraint=='tight':
            bigMnPlus = min(N_plus-1,N_minus)*(tight_parameters['max_fx_n'] - tight_parameters['min_fx_n'])
            bigMnMinus = min(N_minus-1,N_plus)*(tight_parameters['max_fx_n'] - tight_parameters['min_fx_n'])
        for i in points:
            for j in piecePlus:
                model.add_linear_constraint(zPlus[i] - sum([aPlus[j][r]*x[i,r] for r in dims]) - bPlus[j] <= bigMnPlus[i]*(1-decPlus[i][j]), name=f"bigMnPlus({i,j})")
            for k in pieceMinus:
                model.add_linear_constraint(zMinus[i] - sum([aMinus[k][r]*x[i,r] for r in dims]) - bMinus[k] <= bigMnMinus[i]*(1-decMinus[i][k]), name=f"bigMnMinus({i,k})")
                 
    if sort_affine_pieces:
        for j in piecePlus[1:]:
            model.add_linear_constraint(aPlus[j][0] >= aPlus[j-1][0], name=f"sortPlusPieces({j})")
        for k in pieceMinus[1:]:
            model.add_linear_constraint(aMinus[k][0] >= aMinus[k-1][0], name=f"sortMinusPieces({k})")
    
    if impose_d_plus_1_points_per_piece=='f+ and f-':
        for j in piecePlus:
            model.add_linear_constraint(sum(decPlus[i][j] for i in points) >= d+1)
        for k in pieceMinus:
            model.add_linear_constraint(sum(decMinus[i][k] for i in points) >= d+1)
            
    if impose_d_plus_1_points_per_piece=='f':
        for j in piecePlus:
            for k in pieceMinus:
                for i in points:
                    model.add_linear_constraint(dec[i][j][k] <= decPlus[i][j])
                    model.add_linear_constraint(dec[i][j][k] <= decMinus[i][k])
                    model.add_linear_constraint(dec[i][j][k] >= decPlus[i][j] + decMinus[i][k] - 1)
                    model.add_linear_constraint(gamma[j][k] >= dec[i][j][k])
                model.add_linear_constraint(gamma[j][k]*(d+1) <= sum(dec[i][j][k] for i in points)) 
        

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
                aMinus[k][r].lower_bound = - min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][r] - tight_parameters['min_a_b'][r])
                aMinus[k][r].upper_bound = min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][r] - tight_parameters['min_a_b'][r])
            aMinus[k][0].lower_bound = 0
            bMinus[k].lower_bound = - min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][-1] - tight_parameters['min_a_b'][-1])
            bMinus[k].upper_bound = min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][-1] - tight_parameters['min_a_b'][-1])
        for j in piecePlus:
            for r in dims:
                aPlus[j][r].lower_bound = tight_parameters['min_a_b'][r] - min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][r] - tight_parameters['min_a_b'][r])
                aPlus[j][r].upper_bound = tight_parameters['max_a_b'][r] + min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][r] - tight_parameters['min_a_b'][r])
            aPlus[j][0].lower_bound = tight_parameters['min_a_b'][0]
            bPlus[j].lower_bound = tight_parameters['min_a_b'][-1] - min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][-1] - tight_parameters['min_a_b'][-1])
            bPlus[j].upper_bound = tight_parameters['max_a_b'][-1] + min(N_minus-1,N_plus)*(tight_parameters['max_a_b'][-1] - tight_parameters['min_a_b'][-1])

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
            solution_hint_dict[zPWL[i]] = solution_hint['zPWL'][i]
            solution_hint_dict[zPlus[i]] = solution_hint['zPlus'][i]
            solution_hint_dict[zMinus[i]] = solution_hint['zMinus'][i]
            solution_hint_dict[error[i]] = solution_hint['error'][i]
        for j in piecePlus:
            solution_hint_dict[bPlus[j]] = solution_hint['bPlus'][j]
            for r in dims:
                solution_hint_dict[aPlus[j][r]] = solution_hint['aPlus'][j,r]
            for i in points:
                solution_hint_dict[decPlus[i][j]] = solution_hint['decPlus'][i,j]
        for k in pieceMinus:
            solution_hint_dict[bMinus[k]] = solution_hint['bMinus'][k]
            for r in dims:
                solution_hint_dict[aMinus[k][r]] = solution_hint['aMinus'][k,r]
            for i in points:
                solution_hint_dict[decMinus[i][k]] = solution_hint['decMinus'][i,k]

        s_hint = model_parameters.SolutionHint(variable_values=solution_hint_dict)
        model_params = model_parameters.ModelSolveParameters(solution_hints=[s_hint])
        
    params = mathopt.SolveParameters(solution_pool_size=solution_pool_size, 
                                     enable_output=enable_output, 
                                     relative_gap_tolerance=relative_gap_tolerance, 
                                     time_limit=timedelta(seconds=time_limit_seconds))
    
    if solver=='GUROBI':
        params.gurobi.param_values["IntFeasTol"] = str(integer_feasibility_tolerance)
        params.gurobi.param_values["FeasibilityTol"] = str(integer_feasibility_tolerance)
    elif solver=='HIGHS':
        params.highs.double_options["mip_feasibility_tolerance"] = integer_feasibility_tolerance
        params.highs.double_options["primal_feasibility_tolerance"] = integer_feasibility_tolerance
        params.highs.double_options["dual_feasibility_tolerance"] = integer_feasibility_tolerance
    elif solver=='SCIP':
        params.gscip.real_params['numerics/feastol'] = integer_feasibility_tolerance
        params.gscip.real_params['numerics/dualfeastol'] = integer_feasibility_tolerance
    
    # --- Solve the problem ---
    result = mathopt.solve(model, solver_map[solver], 
                           params=params, model_params=model_params)
    
    variables = {'zPWL': zPWL, 'zMinus': zMinus, 'zPlus': zPlus, 'error': error,
                 'aPlus': aPlus, 'bPlus': bPlus, 'aMinus': aMinus, 'bMinus': bMinus,
                 'decPlus': decPlus, 'decMinus': decMinus, 'dec': dec, 'gamma': gamma,
                 'alphaPlus': alphaPlus, 'alphaMinus': alphaMinus}
    
    return model, variables, result


def extract_values(variables, result, data=None, clean_values=False):
    
    N_plus = len(variables['bPlus'])
    N_minus = len(variables['bMinus'])
    N_points = len(variables['zPWL'])
    d = len(variables['aPlus'][0])
    
    aPlus = np.array([[result.variable_values()[variables['aPlus'][j][r]] for r in range(d)] for j in range(N_plus)])
    aMinus = np.array([[result.variable_values()[variables['aMinus'][k][r]] for r in range(d)] for k in range(N_minus)])
    bPlus = np.array([result.variable_values()[variables['bPlus'][j]] for j in range(N_plus)])
    bMinus = np.array([result.variable_values()[variables['bMinus'][k]] for k in range(N_minus)])
    decPlus = np.array([[result.variable_values()[variables['decPlus'][i][j]] for j in range(N_plus)] for i in range(N_points)])
    decMinus = np.array([[result.variable_values()[variables['decMinus'][i][k]] for k in range(N_minus)] for i in range(N_points)])
    zPWL = np.array([result.variable_values()[variables['zPWL'][i]] for i in range(N_points)])
    zPlus = np.array([result.variable_values()[variables['zPlus'][i]] for i in range(N_points)])
    zMinus = np.array([result.variable_values()[variables['zMinus'][i]] for i in range(N_points)])
    error = np.array([result.variable_values()[variables['error'][i]] for i in range(N_points)])
    
    if clean_values:
        
        x = data[:,:-1]
        z = data[:,-1]
        d = data.shape[1]-1
        dims = range(d)
        
        # round the dec variables
        decPlus = np.round(decPlus)
        decMinus = np.round(decMinus)
        
        # adjust zPlus and zMinus based on rounded dec values
        for i in range(N_points):
            for j in range(N_plus):
                if decPlus[i,j] == 1.:
                    zPlus[i] = sum([aPlus[j,r]*x[i,r] for r in dims]) + bPlus[j]
            for k in range(N_minus):
                if decMinus[i,k] == 1.:
                    zMinus[i] = sum([aMinus[k,r]*x[i,r] for r in dims]) + bMinus[k]
        # correct zPWL and error
        zPWL = zPlus - zMinus
        error = abs(zPWL-z)
        
    variable_values = {'aPlus': aPlus,
                       'aMinus': aMinus,
                       'bPlus': bPlus,
                       'bMinus': bMinus,
                       'decPlus': decPlus,
                       'decMinus': decMinus,
                       'zPWL': zPWL,
                       'zPlus': zPlus,
                       'zMinus': zMinus,
                       'error': error}

    return variable_values


def illustrate_CPWL(data, variable_values, faces,
                    ax=None, size=5, colormap=None, alpha=0.4,
                    exploded_factor=0., show_tick=True,
                    show_points=True, show_error=True):
    d = data.shape[1]-1
    N = data.shape[0]
    if d==2:
        if ax is None:
            plt.figure(figsize = (8,8),facecolor="w")
            ax = plt.axes(projection="3d")
        if show_points:
            ax_points = ax.scatter(*zip(*data), c='r', s=size)
        if show_error:
            z_PWL = evaluate_DC_CPWL_function(variable_values,data[:,:2])
            for i in range(N):
                ax.plot([data[i,0]]*2, [data[i,1]]*2, [data[i,2], z_PWL[i]],'k')
        if colormap is None:
            face_colors='C0'
        else:
            cmap = matplotlib.colormaps[colormap]
            num_faces = len(faces)
            face_colors = face_colors = [cmap(i / num_faces) for i in range(num_faces)]
        faceCollection = Poly3DCollection(faces,shade=False,facecolors=face_colors,edgecolors='k',alpha=alpha)
        ax.add_collection3d(faceCollection)
        if not show_tick:
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('z')
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        plt.draw()
        return ax_points
    elif d==3:
        list_list_faces = []
        list_centroids = []
        data_allocation_face = np.zeros(N, dtype=int)
        for i, f in enumerate(faces):
            list_face = []
            hull = ConvexHull(f[:,:3])
            points = hull.points
            simplices = hull.simplices
            equations = np.round(hull.equations,9)
            # merge simplices with the same 3d equations
            _, idx_uniques = np.unique(equations,axis=0,return_inverse=True)
            nber_unique_eq = max(idx_uniques)+1
            for eq_id in range(nber_unique_eq):
                vertices_id_from_eq = np.unique(simplices[idx_uniques==eq_id,:])
                face_points = points[vertices_id_from_eq,:]
                # Sort vertices Counter-Clockwise
                # Project to 2D by dropping the coord with max normal component
                n = equations[np.where(idx_uniques == eq_id)[0][0], :3] # Face normal
                drop_axis = np.argmax(np.abs(n))
                proj = np.delete(face_points, drop_axis, axis=1) # Project to 2D
                list_face.append(face_points[ConvexHull(proj).vertices])
            list_list_faces.append(list_face)
            list_centroids.append(points.mean(axis=0))
            is_inside_hull = np.all(data[:,:3] @ equations[:,:-1].T + equations[:,-1] <= 1e-9, axis=1)
            data_allocation_face[is_inside_hull] = i
        # list_vector_spread = [c - np.full(3,0.5) for c in list_centroids]
        center_centroid = np.array(list_centroids).mean(axis=0)
        list_vector_spread = [c - center_centroid for c in list_centroids]
        list_list_faces_exploded = []
        data_exploded = data[:,:3]*1
        for k in range(len(list_centroids)):
            list_list_faces_exploded.append([face + exploded_factor*list_vector_spread[k].reshape(-1,3) for face in list_list_faces[k]])
            data_exploded[data_allocation_face==k,:] = data_exploded[data_allocation_face==k,:] + exploded_factor*list_vector_spread[k]
        N_domains = len(list_list_faces_exploded)
        if colormap is None:
            face_colors=['C0']*N_domains
        else:
            cmap = matplotlib.colormaps[colormap]
            face_colors = [cmap(i / N_domains) for i in range(N_domains)]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if show_points:
            ax_points = ax.scatter(*zip(*data_exploded), c='r', s=size)
        if show_error:
            z_PWL = evaluate_DC_CPWL_function(variable_values,data[:,:3])
            errors = 500*abs(z_PWL - data[:,-1])
            ax.scatter(*zip(*data_exploded), s=errors, marker='o', edgecolor='k', facecolor='none')
        for k in range(N_domains):
            domain = list_list_faces_exploded[k]
            ax.add_collection3d(Poly3DCollection(domain, alpha=alpha, facecolors=face_colors[k], edgecolor='k'))        
        if not show_tick:
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('x₃')
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        return list_list_faces



#%% Main execution

if __name__ == "__main__":

    print("Create data set")
    path = "./data/.xlsx"
    Naxis=9
    xy = np.array(np.meshgrid(np.linspace(-1,1,Naxis),np.linspace(-1,1,Naxis))).reshape(2,-1).T
    xy = xy + (0.5/(Naxis-1))*np.random.rand(xy.shape[0],xy.shape[1])
    # z = xy[:,0]**2 + xy[:,1]**2
    z = xy[:,0]*np.sin(np.pi*xy[:,1]/2 + 0.5*np.pi)
    # z = xy[:,0]**2 - xy[:,1]**2
    data = np.c_[xy,z]
    
    # CY data
    data = pd.read_excel(path ,index_col=0)
    data = (data.values)[:,-3:]
    data = data[~np.isnan(data).any(axis=1),:]
    
    # 3D data
    Naxis=4
    x = np.array(np.meshgrid(*tuple([np.linspace(-1,1,Naxis)]*3))).reshape(3,-1).T
    x = x + (0.5/(Naxis-1))*np.random.rand(x.shape[0],x.shape[1])
    z = (x**2).sum(axis=1)
    data = np.c_[x,z]
    
    print("Define max error")
    max_error = 0.5
    
    print("Rescale data set to the space [1,2]^(d+1)")
    print("This particular rescaling is important to keep the MILP coefficients in a good range")
    rescaled_data, slopes, intercepts = rescale_data(data,between_one_and_two=True)
    rescaled_error = max_error/slopes[-1]
    
    print("Plot rescaled data")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*zip(*rescaled_data))
    
    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data, rescaled_error)
    tight_parameters = get_tight_parameters(rescaled_data, affine_set, max_slope=100)
    
    print("Solve the MILP model")
    model, variables, result = solve_CPWL_approximation(
        rescaled_data, max_error=rescaled_error, N_plus=6, N_minus=1,
        objective='max error',
        big_M_constraint='tight', integer_feasibility_tolerance=1e-9,
        solver='GUROBI', default_big_M=1e6, tight_parameters=tight_parameters,
        fix_first_affine_piece=True, impose_d_plus_1_points_per_piece='f+ and f-',
        sort_affine_pieces=False,
        bounded_variables=True,
        time_limit_seconds=60)
    
    print("Extract and clean the CPWL results")
    variable_values = extract_values(variables, result, data=rescaled_data, clean_values=True)
    
    print("Rescale the CPWL results to the space [0,1]^(d+1)")
    d = data.shape[1]-1
    slopes2 = np.ones(d+1)
    intercepts2 = -np.ones(d+1)
    rescaled_variable_values = rescale_variable_values(variable_values, slopes2, intercepts2)
    
    print("Calculate the affine pieces")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)
    
    print("Test that equations are consistent with faces")
    verify_dc = [max(abs(np.matmul(np.insert(affine_pieces['dc']['faces'][k],d+1,1,axis=1),
                                   np.insert(affine_pieces['dc']['equations'][k],-1,-1))))
                 for k in range(len(affine_pieces['dc']['faces']))]
    verify_minus = [max(abs(np.matmul(np.insert(affine_pieces['convex_minus']['faces'][k],d+1,1,axis=1),
                                      np.insert(affine_pieces['convex_minus']['equations'][k],-1,-1))))
                 for k in range(len(affine_pieces['convex_minus']['faces']))]
    verify_plus = [max(abs(np.matmul(np.insert(affine_pieces['convex_plus']['faces'][k],d+1,1,axis=1),
                                     np.insert(affine_pieces['convex_plus']['equations'][k],-1,-1))))
                 for k in range(len(affine_pieces['convex_plus']['faces']))]

    print("Verify the convex hull of domain")
    list_list_faces = illustrate_CPWL(rescaled_data-1, rescaled_variable_values, affine_pieces["dc"]["faces"],
                                      ax=None, size=5, colormap='tab20', alpha=0.4,
                                      exploded_factor=0.4, show_tick=False)
    # equations from polytopes
    verify_domains_dc = []
    for k in range(len(affine_pieces['dc']['polytopes'])):
        mat1 = affine_pieces['dc']['polytopes'][k]
        mat2 = np.unique(
            ConvexHull(np.unique(np.vstack(list_list_faces[k]),
                                 axis=0)).equations,axis=0)
        diff = mat1-mat2
        verify_domains_dc.append(abs(diff).max())

    
    illustrate_CPWL(rescaled_data-1, rescaled_variable_values, affine_pieces["dc"]["faces"],
                        ax=None, size=5, colormap='tab20', alpha=0.4,
                        exploded_factor=0.4, show_tick=False)
    
    
    print("Rescale the CPWL results to the original data scale")
    variable_values_original_scale = rescale_variable_values(variable_values, slopes, intercepts)
    
    print("Plot the pieces in their original scaling")
    _, slopes3, intercepts3 = rescale_data(data,between_one_and_two=False)
    rescaled_faces = rescale_faces(affine_pieces["dc"]["faces"], slopes3, intercepts3)
    illustrate_CPWL(data, variable_values_original_scale, rescaled_faces,
                        ax=None, size=5, colormap='tab20', alpha=0.4,
                        exploded_factor=0.4, show_tick=True)

    
    zPWL_eval = evaluate_DC_CPWL_function(variable_values_original_scale,data[:,:-1])
    
    plt.plot(data[:,-1])
    plt.plot(zPWL_eval)
    error_eval = abs(zPWL_eval - data[:,-1])
    

#%% Iterative test, add planes

# N_plus, N_minus = 1,1
max_error = 0.5
extended_variable_values = None

N_plus_minus_list = np.array([[1,1],
                              [1,2],
                              [1,3],
                              [2,3],
                              [2,4],
                              [3,4],
                              [3,5]])


for k in range(len(N_plus_minus_list)):
    
    N_plus = int(N_plus_minus_list[k][0])
    N_minus = int(N_plus_minus_list[k][1])
    
    if k>0:
        print('Add planes and warm-start')
        add_plus = N_plus_minus_list[k][0] - N_plus_minus_list[k-1][0]
        add_minus = N_plus_minus_list[k][1] - N_plus_minus_list[k-1][1]
        extended_variable_values = add_pieces_to_solution(variable_values, 
                                                          add_plus = add_plus, 
                                                          add_minus = add_minus)
    
    print(f"Iteration: {k}")
    
    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data, max_error)
    tight_parameters = get_tight_parameters(rescaled_data, affine_set, max_slope=10)
    
    print("Solve the MILP model")
    model, variables, result = solve_CPWL_approximation(
        rescaled_data, max_error=max_error, N_plus=N_plus, N_minus=N_minus,
        objective='average error',
        big_M_constraint='tight', integer_feasibility_tolerance=1e-9,
        solver='HIGHS', default_big_M=1e6, tight_parameters=tight_parameters,
        fix_first_affine_piece=True, impose_d_plus_1_points_per_piece='f+ and f-',
        sort_affine_pieces=False,
        bounded_variables=True,
        solution_hint=extended_variable_values,
        time_limit_seconds=30)
    

    print("Extract and clean the CPWL results")
    variable_values = extract_values(variables, result, data=rescaled_data, clean_values=True)
    
    max_error = variable_values['error'].max() + 1e-5

    print("Rescale the CPWL results to the space [0,1]^(d+1)")
    d = data.shape[1]-1
    slopes2 = np.ones(d+1)
    intercepts2 = -np.ones(d+1)
    rescaled_variable_values = rescale_variable_values(variable_values, slopes2, intercepts2)
    
    print("Calculate the affine pieces")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)
    
    print("Plot the affine pieces")
    face_colors='C0'
    plt.figure(figsize = (8,8), facecolor="w")
    ax = plt.axes(projection="3d")
    faceCollection = Poly3DCollection(affine_pieces["dc"]["faces"],
                                      shade=False,
                                      facecolors=face_colors,
                                      edgecolors='k',
                                      alpha=0.4)
    ax.add_collection3d(faceCollection)
    ax.scatter(*zip(*rescaled_data-1))
    
#%% Iterative test, add points

    extended_variable_values = None
    
    size_subset = 40
    
    farthest_point_indices = find_all_farthest_point_sampling(rescaled_data, 
                                                              from_domain=True,
                                                              index_initial_point=0)
    
    rescaled_data1 = rescaled_data[farthest_point_indices[:size_subset],:]
    
    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data1, rescaled_error)
    tight_parameters = get_tight_parameters(rescaled_data1, affine_set, max_slope=100)
    
    print("Solve the MILP model")
    model, variables, result = solve_CPWL_approximation(
        rescaled_data1, max_error=rescaled_error, N_plus=4, N_minus=4,
        objective='max error',
        big_M_constraint='tight', integer_feasibility_tolerance=1e-9,
        solver='SCIP', default_big_M=1e6, tight_parameters=tight_parameters,
        fix_first_affine_piece=True, impose_d_plus_1_points_per_piece='f+ and f-',
        sort_affine_pieces=False,
        bounded_variables=True,
        time_limit_seconds=300)

    print("Extract and clean the CPWL results")
    variable_values = extract_values(variables, result, data=rescaled_data1, clean_values=True)
    
    print("Rescale the CPWL results to the space [0,1]^(d+1)")
    d = data.shape[1]-1
    slopes2 = np.ones(d+1)
    intercepts2 = -np.ones(d+1)
    rescaled_variable_values = rescale_variable_values(variable_values, slopes2, intercepts2)
    
    print("Calculate the affine pieces")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)
    
    illustrate_CPWL(rescaled_data1-1, rescaled_variable_values, affine_pieces["dc"]["faces"],
                        ax=None, size=5, colormap='tab20', alpha=0.4,
                        exploded_factor=0.4, show_tick=False)

    extended_variable_values = add_points_to_solution(variable_values, rescaled_data[farthest_point_indices[size_subset:],:])
    rescaled_data2 = rescaled_data[farthest_point_indices,:]

    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data2, rescaled_error)
    tight_parameters = get_tight_parameters(rescaled_data2, affine_set, max_slope=100)
    
    
    print("Solve the MILP model")
    model, variables, result = solve_CPWL_approximation(
        rescaled_data2, max_error=rescaled_error, N_plus=4, N_minus=4,
        objective='max error',
        big_M_constraint='tight', integer_feasibility_tolerance=1e-9,
        solver='SCIP', default_big_M=1e6, tight_parameters=tight_parameters,
        fix_first_affine_piece=True, impose_d_plus_1_points_per_piece='f+ and f-',
        sort_affine_pieces=False,
        bounded_variables=True,
        solution_hint=extended_variable_values,
        time_limit_seconds=60)
    
    print("Extract and clean the CPWL results")
    variable_values = extract_values(variables, result, data=rescaled_data2, clean_values=True)
    
    print("Rescale the CPWL results to the space [0,1]^(d+1)")
    d = data.shape[1]-1
    slopes2 = np.ones(d+1)
    intercepts2 = -np.ones(d+1)
    rescaled_variable_values = rescale_variable_values(variable_values, slopes2, intercepts2)
    
    print("Calculate the affine pieces")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)
    
    illustrate_CPWL(rescaled_data2-1, rescaled_variable_values, affine_pieces["dc"]["faces"],
                        ax=None, size=5, colormap='tab20', alpha=0.4,
                        exploded_factor=0.4, show_tick=False)
