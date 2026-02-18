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
from cpwl_optim.cpwl.cpwl_functions import find_affine_pieces
from cpwl_optim.data_io.parse_data import *

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes3d.mouserotationstyle"] = "azel"


def add_pieces_to_solution(variable_values, add_plus=0, add_minus=0):

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

    aPlus = np.r_[aPlus, np.repeat(aPlus[[-1], :], add_plus, axis=0)]
    aMinus = np.r_[aMinus, np.repeat(aMinus[[-1], :], add_minus, axis=0)]
    bPlus = np.r_[bPlus, np.repeat(bPlus[-1], add_plus)]
    bMinus = np.r_[bMinus, np.repeat(bMinus[-1], add_minus)]
    decPlus = np.c_[decPlus, np.repeat(decPlus[:, [-1]], add_plus, axis=1)]
    decMinus = np.c_[decMinus, np.repeat(decMinus[:, [-1]], add_minus, axis=1)]

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


data, descr = load_case1_data()

max_error = 0.5
print(f"Max approximation error (predefined): {max_error}\n")

# print("Rescale data set to the space [1,2]^(d+1)")
# print("This particular rescaling is important to keep the MILP coefficients in a good range")
rescaled_data, slopes, intercepts = rescale_data(data)
rescaled_error = max_error / slopes[-1]

max_error = 0.5
extended_variable_values = None

N_plus_minus_list = np.array([[1, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5]])


for k in range(len(N_plus_minus_list)):

    N_plus = int(N_plus_minus_list[k][0])
    N_minus = int(N_plus_minus_list[k][1])

    if k > 0:
        print("Add planes and warm-start")
        add_plus = N_plus_minus_list[k][0] - N_plus_minus_list[k - 1][0]
        add_minus = N_plus_minus_list[k][1] - N_plus_minus_list[k - 1][1]
        extended_variable_values = add_pieces_to_solution(
            variable_values, add_plus=add_plus, add_minus=add_minus
        )

    print(f"Iteration: {k}")

    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data, max_error)
    tight_parameters = get_tight_parameters(rescaled_data, affine_set, max_slope=10)

    print("Solve the MILP model")
    model, variables, result = solve_CPWL_model(
        rescaled_data,
        max_error=max_error,
        N_plus=N_plus,
        N_minus=N_minus,
        objective="average error",
        big_M_constraint="tight",
        integer_feasibility_tolerance=1e-9,
        solver="HIGHS",
        default_big_M=1e6,
        tight_parameters=tight_parameters,
        fix_first_affine_piece=True,
        impose_d_plus_1_points_per_piece="f+ and f-",
        sort_affine_pieces=False,
        bounded_variables=True,
        solution_hint=extended_variable_values,
        time_limit_seconds=30,
    )

    print("Extract and clean the CPWL results")
    variable_values = extract_values(
        variables, result, data=rescaled_data, clean_values=True
    )

    max_error = variable_values["error"].max() + 1e-5

    print("Rescale the CPWL results to the space [0,1]^(d+1)")
    d = data.shape[1] - 1
    slopes2 = np.ones(d + 1)
    intercepts2 = -np.ones(d + 1)
    rescaled_variable_values = rescale_variable_values(
        variable_values, slopes2, intercepts2
    )

    print("Calculate the affine pieces")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)

    print("Plot the affine pieces")
    face_colors = "C0"
    plt.figure(figsize=(8, 8), facecolor="w")
    ax = plt.axes(projection="3d")
    faceCollection = Poly3DCollection(
        affine_pieces["dc"]["faces"],
        shade=False,
        facecolors=face_colors,
        edgecolors="k",
        alpha=0.4,
    )
    ax.add_collection3d(faceCollection)
    ax.scatter(*zip(*rescaled_data - 1))

    plt.savefig(f"../output/planes_warm_start_{N_plus}_{N_minus}.png", dpi=300)
