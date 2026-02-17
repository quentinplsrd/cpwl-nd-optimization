# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""
# %%
from cpwl_optim.optim.solver import solve_CPWL_model, extract_values
from cpwl_optim.cpwl.tight_regions import find_affine_set, get_tight_parameters
from cpwl_optim.cpwl.data_scaler import (
    rescale_data,
    rescale_variable_values,
)

from cpwl_optim.cpwl.cpwl_functions import (
    find_affine_pieces,
    illustrate_CPWL,
    find_all_farthest_point_sampling,
)
from cpwl_optim.data_io.parse_data import *

import numpy as np


def add_points_to_solution(variable_values, new_points):

    aPlus = variable_values["aPlus"] * 1
    aMinus = variable_values["aMinus"] * 1
    bPlus = variable_values["bPlus"] * 1
    bMinus = variable_values["bMinus"] * 1
    decPlus = variable_values["decPlus"] * 1 if "decPlus" in variable_values else None
    decMinus = (
        variable_values["decMinus"] * 1 if "decMinus" in variable_values else None
    )
    zPWL = variable_values["zPWL"] * 1 if "zPWL" in variable_values else None
    zPlus = variable_values["zPlus"] * 1 if "zPlus" in variable_values else None
    zMinus = variable_values["zMinus"] * 1 if "zMinus" in variable_values else None
    error = variable_values["error"] * 1 if "error" in variable_values else None

    N_plus = len(bPlus)
    N_minus = len(bMinus)

    x = new_points[:, :-1]
    z = new_points[:, -1]

    new_zPlus_disag = np.matmul(aPlus, x.T) + bPlus.reshape(-1, 1)
    new_zMinus_disag = np.matmul(aMinus, x.T) + bMinus.reshape(-1, 1)

    new_zPlus = new_zPlus_disag.max(axis=0)
    new_zMinus = new_zMinus_disag.max(axis=0)
    new_zPWL = new_zPlus - new_zMinus
    new_error = abs(new_zPWL - z)
    new_decPlus = np.eye(N_plus)[new_zPlus_disag.argmax(axis=0)]
    new_decMinus = np.eye(N_minus)[new_zMinus_disag.argmax(axis=0)]

    zPlus = np.r_[zPlus, new_zPlus]
    zMinus = np.r_[zMinus, new_zMinus]
    zPWL = np.r_[zPWL, new_zPWL]
    error = np.r_[error, new_error]
    decPlus = np.r_[decPlus, new_decPlus]
    decMinus = np.r_[decMinus, new_decMinus]

    extended_variable_values = {
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

    return extended_variable_values


# %%

data, descr = load_case1_data()

max_error = 0.5
print(f"Max approximation error (predefined): {max_error}\n")

# print("Rescale data set to the space [1,2]^(d+1)")
# print("This particular rescaling is important to keep the MILP coefficients in a good range")
rescaled_data, slopes, intercepts = rescale_data(data)
rescaled_error = max_error / slopes[-1]

max_error = 0.5
extended_variable_values = None

extended_variable_values = None

size_subset = 40

farthest_point_indices = find_all_farthest_point_sampling(
    rescaled_data, from_domain=True, index_initial_point=0
)

rescaled_data1 = rescaled_data[farthest_point_indices[:size_subset], :]

print("Calculate all combinations of affine functions and the tight parameters")
affine_set = find_affine_set(rescaled_data1, rescaled_error)
tight_parameters = get_tight_parameters(rescaled_data1, affine_set, max_slope=100)

print("Solve the MILP model")
model, variables, result = solve_CPWL_model(
    rescaled_data1,
    max_error=rescaled_error,
    N_plus=4,
    N_minus=4,
    objective="max error",
    big_M_constraint="tight",
    integer_feasibility_tolerance=1e-9,
    solver="SCIP",
    default_big_M=1e6,
    tight_parameters=tight_parameters,
    fix_first_affine_piece=True,
    impose_d_plus_1_points_per_piece="f+ and f-",
    sort_affine_pieces=False,
    bounded_variables=True,
    time_limit_seconds=300,
)

print("Extract and clean the CPWL results")
variable_values = extract_values(
    variables, result, data=rescaled_data1, clean_values=True
)

print("Rescale the CPWL results to the space [0,1]^(d+1)")
d = data.shape[1] - 1
slopes2 = np.ones(d + 1)
intercepts2 = -np.ones(d + 1)
rescaled_variable_values = rescale_variable_values(
    variable_values, slopes2, intercepts2
)

print("Calculate the affine pieces")
affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)

illustrate_CPWL(
    rescaled_data1 - 1,
    rescaled_variable_values,
    affine_pieces["dc"]["faces"],
    ax=None,
    size=5,
    colormap="tab20",
    alpha=0.4,
    exploded_factor=0.4,
    show_tick=False,
)

extended_variable_values = add_points_to_solution(
    variable_values, rescaled_data[farthest_point_indices[size_subset:], :]
)
rescaled_data2 = rescaled_data[farthest_point_indices, :]

print("Calculate all combinations of affine functions and the tight parameters")
affine_set = find_affine_set(rescaled_data2, rescaled_error)
tight_parameters = get_tight_parameters(rescaled_data2, affine_set, max_slope=100)

print("Solve the MILP model")
model, variables, result = solve_CPWL_model(
    rescaled_data2,
    max_error=rescaled_error,
    N_plus=4,
    N_minus=4,
    objective="max error",
    big_M_constraint="tight",
    integer_feasibility_tolerance=1e-9,
    solver="SCIP",
    default_big_M=1e6,
    tight_parameters=tight_parameters,
    fix_first_affine_piece=True,
    impose_d_plus_1_points_per_piece="f+ and f-",
    sort_affine_pieces=False,
    bounded_variables=True,
    solution_hint=extended_variable_values,
    time_limit_seconds=60,
)

print("Extract and clean the CPWL results")
variable_values = extract_values(
    variables, result, data=rescaled_data2, clean_values=True
)

print("Rescale the CPWL results to the space [0,1]^(d+1)")
d = data.shape[1] - 1
slopes2 = np.ones(d + 1)
intercepts2 = -np.ones(d + 1)
rescaled_variable_values = rescale_variable_values(
    variable_values, slopes2, intercepts2
)

print("Calculate the affine pieces")
affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)

illustrate_CPWL(
    rescaled_data2 - 1,
    rescaled_variable_values,
    affine_pieces["dc"]["faces"],
    ax=None,
    size=5,
    colormap="tab20",
    alpha=0.4,
    exploded_factor=0.4,
    show_tick=False,
    savefig=True,
    path_savefig=f"../output/data_warm_start_{descr}",
)
