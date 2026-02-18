# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""
# %%
import os
from cpwl_optim.optim.solver import solve_CPWL_model, extract_values
from cpwl_optim.cpwl.tight_regions import find_affine_set, get_tight_parameters
from cpwl_optim.cpwl.data_scaler import (
    rescale_data,
    rescale_faces,
    rescale_variable_values,
)
from cpwl_optim.data_io.parse_data import *
from cpwl_optim.cpwl.cpwl_functions import (
    find_affine_pieces, 
    illustrate_CPWL, 
    evaluate_DC_CPWL_function
)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes3d.mouserotationstyle"] = "azel"


# %% Main execution
# Create output folder if it does not exist
if not os.path.exists("../output"):
    os.makedirs("../output")

print("Create data set")
path = "../data/crystal_hydro.xlsx"

for data_loader, arg, n_affine_pieces in [
    (load_case1_data, None, [2, 3]),
    (load_case2_data, path, [2, 3]),
    (load_case3_data, None, [6, 1]),
]:
    if arg is not None:
        data, descr = data_loader(arg)
    else:
        data, descr = data_loader()

    max_error = 0.5
    n_plus, n_minus = n_affine_pieces
    print(f"Max approximation error (predefined): {max_error}\n")

    # print("Rescale data set to the space [1,2]^(d+1)")
    # print("This particular rescaling is important to keep the MILP coefficients in a good range")
    rescaled_data, slopes, intercepts = rescale_data(data)
    rescaled_error = max_error / slopes[-1]

    print("Plot rescaled data")
    # Create data visualization function
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*zip(*rescaled_data))
    plt.savefig(f"../output/rescaled_data_{descr}.png", dpi=300, bbox_inches="tight")

    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data, rescaled_error)
    tight_parameters = get_tight_parameters(
        rescaled_data, affine_set, max_slope=100
    )

    print("Solve the MILP model")
    model, variables, result = solve_CPWL_model(
        rescaled_data,
        max_error=rescaled_error,
        N_plus=n_plus,
        N_minus=n_minus,
        big_M_constraint="tight",
        solver="HIGHS",
        default_big_M=1e6,
        tight_parameters=tight_parameters,
        fix_first_affine_piece=True,
        impose_d_plus_1_points_per_piece="f+ and f-",
        sort_affine_pieces=False,
        bounded_variables=True,
        time_limit_seconds=60,
    )

    print("Extract and clean the CPWL results")
    variable_values = extract_values(
        variables, result, data=rescaled_data, clean_values=True
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

    print("Verify the convex hull of domain")
    list_list_faces = illustrate_CPWL(
        rescaled_data - 1,
        rescaled_variable_values,
        affine_pieces["dc"]["faces"],
        ax=None,
        size=5,
        colormap="tab20",
        alpha=0.4,
        exploded_factor=0.4,
        show_tick=False,
        savefig=True,
        path_savefig=f"../output/rescaled_{descr}",
    )

    print("Rescale the CPWL results to the original data scale")
    variable_values_original_scale = rescale_variable_values(
        variable_values, slopes, intercepts
    )

    print("Plot the pieces in their original scaling")
    _, slopes3, intercepts3 = rescale_data(data, between_one_and_two=False)
    rescaled_faces = rescale_faces(
        affine_pieces["dc"]["faces"], slopes3, intercepts3
    )
    illustrate_CPWL(
        data,
        variable_values_original_scale,
        rescaled_faces,
        ax=None,
        size=5,
        colormap="tab20",
        alpha=0.4,
        exploded_factor=0.4,
        show_tick=True,
        savefig=True,
        path_savefig=f"../output/original_{descr}",
    )

    zPWL_eval = evaluate_DC_CPWL_function(
        variable_values_original_scale, data[:, :-1]
    )
    error_eval = abs(zPWL_eval - data[:, -1])

    plt.figure(figsize=(6, 6), facecolor="w")
    plt.plot(data[:, -1], zPWL_eval, "o", label="CPWL evaluation")
    plt.savefig(f"../output/evaluation_{descr}.png", dpi=300, bbox_inches="tight")

