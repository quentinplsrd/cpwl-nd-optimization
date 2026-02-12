# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""

from src.cpwl_optim.optim.solver import solve_CPWL_model, extract_values
from src.cpwl_optim.cpwl.tight_regions import find_affine_set, get_tight_parameters
from src.cpwl_optim.cpwl.data_scaler import (
    rescale_data,
    rescale_equations,
    rescale_faces,
    rescale_polytopes,
    rescale_variable_values,
)
from src.cpwl_optim.data_io.parse_data import *

import numpy as np
from typing import Optional, Literal
import time
from scipy.optimize import linprog, linear_sum_assignment
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import ConvexHull, HalfspaceIntersection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes3d.mouserotationstyle"] = "azel"


# %% functions


def calculate_CPWL_approximation(
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
    solver: Literal["GUROBI", "HIGHS", "SCIP"] = "HIGHS",
    enable_output: bool = True,
    relative_gap_tolerance: float = 1e-4,
    integer_feasibility_tolerance: float = 1e-9,
    time_limit_seconds: int = 120,
    plot_results=True,
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

    # check that the data is valid
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise TypeError(f"`data` must be 2D, got shape {data.shape}.")
    N_points, d_plus_1 = data.shape
    d = d_plus_1 - 1
    if d < 1:
        raise TypeError(
            f"Invalid dimension {d+1}: "
            "the data set should have at least 1 input and 1 output dimension."
        )
    if N_points < d + 1:
        raise TypeError(
            f"Invalid data shape {data.shape}: "
            f"number of points ({N_points}) must be >= dimension ({d+1})."
        )
    if np.isnan(data).any():
        raise ValueError("Invalid data: " "the data contains NaN values.")
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    constant_value = data_max == data_min
    if constant_value[:-1].any():
        raise ValueError(
            "Invalid data: " "at least one input dimension is constant-valued."
        )

    # check the max error is valid
    if not isinstance(max_error, (float, np.floating)):
        raise TypeError(
            f"`max_error` must be a float, got {type(max_error).__name__}.")
    if max_error < 0:
        raise ValueError("`max_error` must be nonnegative.")

    # check the N_plus and N_minus are valid
    if not isinstance(N_plus, int) or not isinstance(N_minus, int):
        raise TypeError("`N_plus` and `N_minus` must be integers.")
    if N_plus < 1 or N_minus < 1:
        raise ValueError("`N_plus` and `N_minus` must be >= 1.")

    # ...

    print("Rescaling data set to the space [1,2]^(d+1)")
    print(
        "This particular rescaling is important to keep the MILP coefficients in a good range"
    )
    rescaled_data, slopes, intercepts = rescale_data(
        data, between_one_and_two=True)
    rescaled_error = max_error / slopes[-1]

    tight_parameters = None
    if big_M_constraint == "tight":
        print(
            "Calculating all combinations of affine functions and the tight parameters"
        )
        affine_set = find_affine_set(rescaled_data, rescaled_error)
        tight_parameters = get_tight_parameters(
            rescaled_data, affine_set, max_slope=100
        )

    print("Solving the MILP model")
    # TODO: print a summary of all tightening strategies used and solver parameters
    time.sleep(0.5)
    _, variables, result = solve_CPWL_model(
        rescaled_data,
        max_error=rescaled_error,
        N_plus=N_plus,
        N_minus=N_minus,
        objective=objective,
        big_M_constraint=big_M_constraint,
        integer_feasibility_tolerance=integer_feasibility_tolerance,
        enable_output=enable_output,
        solver=solver,
        default_big_M=default_big_M,
        tight_parameters=tight_parameters,
        fix_first_affine_piece=fix_first_affine_piece,
        impose_d_plus_1_points_per_piece=impose_d_plus_1_points_per_piece,
        sort_affine_pieces=sort_affine_pieces,
        bounded_variables=bounded_variables,
        time_limit_seconds=time_limit_seconds,
    )

    print("Extracting and cleaning the CPWL results")
    variable_values = extract_values(
        variables, result, data=rescaled_data, clean_values=True
    )

    print("Rescaling the CPWL results to the original data scale")
    variable_values_original_scale = rescale_variable_values(
        variable_values, slopes, intercepts
    )

    print("Rescaling the CPWL results to the space [0,1]^(d+1)")
    slopes2 = np.ones(d + 1)
    intercepts2 = -np.ones(d + 1)
    rescaled_variable_values = rescale_variable_values(
        variable_values, slopes2, intercepts2
    )

    print("Calculating the affine pieces in [0,1]^(d+1)")
    affine_pieces = find_affine_pieces(rescaled_variable_values, max_z=1e4)

    print("Calculating the affine pieces in their original scaling")
    _, slopes3, intercepts3 = rescale_data(data, between_one_and_two=False)
    original_affine_pieces = transform_affine_pieces(
        affine_pieces, slopes3, intercepts3
    )

    if plot_results:
        print("Plot the pieces in their original scaling")
        rescaled_faces = original_affine_pieces["dc"]["faces"]
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
        )

    variable_values_original_scale["affine pieces"] = original_affine_pieces

    # print(check_validity_affine_pieces(original_affine_pieces))

    return variable_values_original_scale


def evaluate_DC_CPWL_function(CPWL_parameters, x):

    aPlus = CPWL_parameters["aPlus"] * 1
    aMinus = CPWL_parameters["aMinus"] * 1
    bPlus = CPWL_parameters["bPlus"] * 1
    bMinus = CPWL_parameters["bMinus"] * 1

    zPlus = (np.matmul(aPlus, x.T) + bPlus.reshape(-1, 1)).max(axis=0)
    zMinus = (np.matmul(aMinus, x.T) + bMinus.reshape(-1, 1)).max(axis=0)
    zPWL = zPlus - zMinus

    return zPWL


def add_pieces_to_solution(variable_values, add_plus=0, add_minus=0):

    aPlus = variable_values["aPlus"] * 1
    aMinus = variable_values["aMinus"] * 1
    bPlus = variable_values["bPlus"] * 1
    bMinus = variable_values["bMinus"] * 1
    decPlus = variable_values["decPlus"] * \
        1 if "decPlus" in variable_values else None
    decMinus = (
        variable_values["decMinus"] *
        1 if "decMinus" in variable_values else None
    )
    zPWL = variable_values["zPWL"] * 1 if "zPWL" in variable_values else None
    zPlus = variable_values["zPlus"] * \
        1 if "zPlus" in variable_values else None
    zMinus = variable_values["zMinus"] * \
        1 if "zMinus" in variable_values else None
    error = variable_values["error"] * \
        1 if "error" in variable_values else None

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


def find_all_farthest_point_sampling(data, from_domain=True, index_initial_point=0):

    N = data.shape[0]
    if from_domain:
        data = data[:, :-1]

    if index_initial_point is None:
        index_initial_point = np.random.randint(N)
    index_initial_point %= N

    # calculate pair-wise distance
    matrix_dist = squareform(pdist(data, metric="euclidean"))

    farthest_point_indices = [index_initial_point]
    number_of_visited_points = 1

    while number_of_visited_points < N:
        distance_from_closest_point = matrix_dist[farthest_point_indices, :].min(
            axis=0)
        new_farthest_point = distance_from_closest_point.argmax()
        farthest_point_indices.append(new_farthest_point)
        number_of_visited_points += 1

    return farthest_point_indices


def add_points_to_solution(variable_values, new_points):

    aPlus = variable_values["aPlus"] * 1
    aMinus = variable_values["aMinus"] * 1
    bPlus = variable_values["bPlus"] * 1
    bMinus = variable_values["bMinus"] * 1
    decPlus = variable_values["decPlus"] * \
        1 if "decPlus" in variable_values else None
    decMinus = (
        variable_values["decMinus"] *
        1 if "decMinus" in variable_values else None
    )
    zPWL = variable_values["zPWL"] * 1 if "zPWL" in variable_values else None
    zPlus = variable_values["zPlus"] * \
        1 if "zPlus" in variable_values else None
    zMinus = variable_values["zMinus"] * \
        1 if "zMinus" in variable_values else None
    error = variable_values["error"] * \
        1 if "error" in variable_values else None

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


def find_affine_pieces(variable_values, max_z=1e4):

    aPlus = variable_values["aPlus"] * 1
    aMinus = variable_values["aMinus"] * 1
    bPlus = variable_values["bPlus"] * 1
    bMinus = variable_values["bMinus"] * 1

    N_plus = len(bPlus)
    N_minus = len(bMinus)
    d = aPlus.shape[1]

    half_space_plus = np.c_[aPlus, -np.ones(N_plus), bPlus]
    half_space_minus = np.c_[aMinus, -np.ones(N_minus), bMinus]

    # we assume the domain is [0,1]^d
    boundary_constraints = np.zeros((2 * d + 1, d + 2))
    boundary_constraints[: (d + 1), : (d + 1)] = np.eye(d + 1)
    boundary_constraints[:d, d + 1] = -1.0
    boundary_constraints[d, d + 1] = -max_z
    boundary_constraints[(d + 1): 2 * d + 1, :d] = -np.eye(d)

    feasible_point = np.append(np.full(d, 0.5), 0.5 * max_z)

    # find affine pieces for each convex components
    # "polytopes" is the list of piece's domain equations
    # "faces" is the list of piece's list of points
    # "equations" is the list of piece's linear coeff (A,b in z = A*x + b)
    convex_component = []
    for half_space in [half_space_plus, half_space_minus]:
        half_space = np.r_[half_space, boundary_constraints]
        half_space_intersection = HalfspaceIntersection(
            half_space, feasible_point)
        vertices = half_space_intersection.intersections
        convex_hull = ConvexHull(vertices)
        equation_faces, face_id = np.unique(
            convex_hull.equations, return_inverse=True, axis=0
        )
        N_faces = len(equation_faces)
        vertices_to_discard = np.where(vertices[:, -1] > 0.5 * max_z)[0]
        simplices = convex_hull.simplices
        simplices_to_discard = np.isin(
            simplices, vertices_to_discard).any(axis=1)
        faces_to_discard = np.unique(face_id[simplices_to_discard])
        faces_to_keep = np.where(
            ~np.isin(np.arange(N_faces), faces_to_discard))[0]
        set_polytopes, set_faces, set_equations = [], [], []
        for f in faces_to_keep:
            simplex_ids = np.where(face_id == f)[0]
            point_ids = np.unique(simplices[simplex_ids])
            convex_hull_domain = ConvexHull(vertices[point_ids, :-1])
            equations_domains = np.unique(convex_hull_domain.equations, axis=0)
            set_polytopes.append(equations_domains)
            # set_polytopes.append(convex_hull_domain.equations)
            set_faces.append(
                vertices[point_ids[convex_hull_domain.vertices], :])
            # A,b in z = A*x + b
            linear_coeffs = np.delete(-equation_faces[f], -2) / \
                equation_faces[f][-2]
            set_equations.append(linear_coeffs)
            # set_equations.append(equation_faces[f])
        convex_component.append(
            {"polytopes": set_polytopes, "faces": set_faces,
                "equations": set_equations}
        )

    # find for the DC function
    set_plus, set_minus = convex_component
    N_plus = len(set_plus["polytopes"])
    N_minus = len(set_minus["polytopes"])

    set_polytopes, set_faces, set_equations = [], [], []
    # static parameters to find the interior point via LP
    c = np.zeros(d + 1)
    c[-1] = -1
    bounds = [(None, None)] * d + [(0, None)]
    # iterate through all combinations of plus and minus planes
    for j in range(N_plus):
        for k in range(N_minus):
            polytope_plus = set_plus["polytopes"][j]
            polytope_minus = set_minus["polytopes"][k]
            half_space = np.r_[polytope_plus, polytope_minus]
            # find a strictly interior point via LP:  A_ub x ≤ b_ub
            norm_vector = np.reshape(
                np.linalg.norm(half_space[:, :-1],
                               axis=1), (half_space.shape[0], 1)
            )
            A = np.c_[(half_space[:, :-1], norm_vector)]
            b = -half_space[:, -1:]
            sol = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            if not sol.success:
                continue
            interior_point = (sol.x)[:-1]
            half_space_intersection = HalfspaceIntersection(
                half_space, interior_point)
            vertices = half_space_intersection.intersections
            if vertices.size > 0:
                # order vertices counterclockwise via their convex hull
                convex_hull = ConvexHull(vertices)
                # reorder the vertices
                vertices = vertices[convex_hull.vertices, :]
                polytope = convex_hull.equations
                equations_domains = np.unique(polytope, axis=0)
                set_polytopes.append(equations_domains)
                # set_polytopes.append(polytope)
                z_values = evaluate_DC_CPWL_function(variable_values, vertices)
                face = np.c_[vertices, z_values]
                set_faces.append(face)
                set_equations.append(
                    set_plus["equations"][j] - set_minus["equations"][k]
                )

    affine_pieces = {
        "convex_plus": set_plus,
        "convex_minus": set_minus,
        "dc": {
            "polytopes": set_polytopes,
            "faces": set_faces,
            "equations": set_equations,
        },
    }

    return affine_pieces


def transform_affine_pieces(affine_pieces, slopes, intercepts):

    transformed_affine_pieces = {}

    for type_cpwl in ["convex_plus", "convex_minus", "dc"]:
        transformed_affine_pieces[type_cpwl] = {}
        # rescale faces
        rescaled_faces = rescale_faces(
            affine_pieces[type_cpwl]["faces"], slopes, intercepts
        )
        transformed_affine_pieces[type_cpwl]["faces"] = rescaled_faces
        # rescale equations (linear coeffs)
        rescaled_equations = rescale_equations(
            affine_pieces[type_cpwl]["equations"], slopes, intercepts
        )
        transformed_affine_pieces[type_cpwl]["equations"] = rescaled_equations
        # rescale polytopes (domain equations)
        rescaled_polytopes = rescale_polytopes(
            affine_pieces[type_cpwl]["polytopes"], slopes, intercepts
        )
        transformed_affine_pieces[type_cpwl]["polytopes"] = rescaled_polytopes

    return transformed_affine_pieces


def check_validity_affine_pieces(affine_pieces):

    validity_report = {}
    d = affine_pieces["dc"]["faces"][0].shape[1] - 1
    for type_cpwl in ["convex_plus", "convex_minus", "dc"]:
        validity_report[type_cpwl] = {"polytopes": [], "equations": []}
        number_pieces = len(affine_pieces[type_cpwl]["faces"])
        for k in range(number_pieces):
            face = affine_pieces[type_cpwl]["faces"][k]
            polytope = affine_pieces[type_cpwl]["polytopes"][k]
            polytope = polytope / np.linalg.norm(polytope[:, :-1], axis=1).reshape(
                -1, 1
            )
            polytope = np.unique(polytope, axis=0)
            equation = affine_pieces[type_cpwl]["equations"][k]
            equation_domain = np.unique(
                ConvexHull(face[:, :-1]).equations, axis=0)
            dists = cdist(polytope, equation_domain, metric="cityblock")
            row_ind, col_ind = linear_sum_assignment(dists)
            equation_domain = equation_domain[col_ind, :]
            diff_polytope = abs(polytope - equation_domain).max()
            diff_equation = np.insert(face, d + 1, 1, axis=1) @ np.insert(
                equation, -1, -1
            ).reshape(-1, 1)
            diff_equation = abs(diff_equation).max()
            validity_report[type_cpwl]["polytopes"].append(diff_polytope)
            validity_report[type_cpwl]["equations"].append(diff_equation)

    return validity_report


def illustrate_CPWL(
    data,
    variable_values,
    faces,
    ax=None,
    size=5,
    colormap=None,
    alpha=0.4,
    exploded_factor=0.0,
    show_tick=True,
    show_points=True,
    show_error=True,
):
    d = data.shape[1] - 1
    N = data.shape[0]
    if d == 2:
        if ax is None:
            plt.figure(figsize=(8, 8), facecolor="w")
            ax = plt.axes(projection="3d")
        if show_points:
            ax_points = ax.scatter(*zip(*data), c="r", s=size)
        if show_error:
            z_PWL = evaluate_DC_CPWL_function(variable_values, data[:, :2])
            for i in range(N):
                ax.plot([data[i, 0]] * 2, [data[i, 1]]
                        * 2, [data[i, 2], z_PWL[i]], "k")
        if colormap is None:
            face_colors = "C0"
        else:
            cmap = matplotlib.colormaps[colormap]
            num_faces = len(faces)
            face_colors = face_colors = [
                cmap(i / num_faces) for i in range(num_faces)]
        faceCollection = Poly3DCollection(
            faces, shade=False, facecolors=face_colors, edgecolors="k", alpha=alpha
        )
        ax.add_collection3d(faceCollection)
        if not show_tick:
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_zlabel("z")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        plt.draw()
        return ax_points
    elif d == 3:
        list_list_faces = []
        list_centroids = []
        data_allocation_face = np.zeros(N, dtype=int)
        for i, f in enumerate(faces):
            list_face = []
            hull = ConvexHull(f[:, :3])
            points = hull.points
            simplices = hull.simplices
            equations = np.round(hull.equations, 9)
            # merge simplices with the same 3d equations
            _, idx_uniques = np.unique(equations, axis=0, return_inverse=True)
            nber_unique_eq = max(idx_uniques) + 1
            for eq_id in range(nber_unique_eq):
                vertices_id_from_eq = np.unique(
                    simplices[idx_uniques == eq_id, :])
                face_points = points[vertices_id_from_eq, :]
                # Sort vertices Counter-Clockwise
                # Project to 2D by dropping the coord with max normal component
                n = equations[np.where(idx_uniques == eq_id)[
                    0][0], :3]  # Face normal
                drop_axis = np.argmax(np.abs(n))
                proj = np.delete(face_points, drop_axis,
                                 axis=1)  # Project to 2D
                list_face.append(face_points[ConvexHull(proj).vertices])
            list_list_faces.append(list_face)
            list_centroids.append(points.mean(axis=0))
            is_inside_hull = np.all(
                data[:, :3] @ equations[:, :-1].T + equations[:, -1] <= 1e-9, axis=1
            )
            data_allocation_face[is_inside_hull] = i
        # list_vector_spread = [c - np.full(3,0.5) for c in list_centroids]
        center_centroid = np.array(list_centroids).mean(axis=0)
        list_vector_spread = [c - center_centroid for c in list_centroids]
        list_list_faces_exploded = []
        data_exploded = data[:, :3] * 1
        for k in range(len(list_centroids)):
            list_list_faces_exploded.append(
                [
                    face + exploded_factor *
                    list_vector_spread[k].reshape(-1, 3)
                    for face in list_list_faces[k]
                ]
            )
            data_exploded[data_allocation_face == k, :] = (
                data_exploded[data_allocation_face == k, :]
                + exploded_factor * list_vector_spread[k]
            )
        N_domains = len(list_list_faces_exploded)
        if colormap is None:
            face_colors = ["C0"] * N_domains
        else:
            cmap = matplotlib.colormaps[colormap]
            face_colors = [cmap(i / N_domains) for i in range(N_domains)]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        if show_points:
            ax_points = ax.scatter(*zip(*data_exploded), c="r", s=size)
        if show_error:
            z_PWL = evaluate_DC_CPWL_function(variable_values, data[:, :3])
            errors = 500 * abs(z_PWL - data[:, -1])
            ax.scatter(
                *zip(*data_exploded),
                s=errors,
                marker="o",
                edgecolor="k",
                facecolor="none",
            )
        for k in range(N_domains):
            domain = list_list_faces_exploded[k]
            ax.add_collection3d(
                Poly3DCollection(
                    domain, alpha=alpha, facecolors=face_colors[k], edgecolor="k"
                )
            )
        if not show_tick:
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_zlabel("x₃")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        return list_list_faces


# %% Main execution
if __name__ == "__main__":

    print("Create data set")
    path = "./data/crystal_hydro.xlsx"

    # data = load_case1_data()
    # CY data
    # data = load_case2_data(path)
    # 3D data
    data = load_case3_data()

    max_error = 0.5
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
    plt.show(block = False)

    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data, rescaled_error)
    tight_parameters = get_tight_parameters(
        rescaled_data, affine_set, max_slope=100)

    print("Solve the MILP model")
    model, variables, result = solve_CPWL_model(
        rescaled_data,
        max_error=rescaled_error,
        N_plus=6,
        N_minus=1,
        big_M_constraint="tight",
        solver="GUROBI",
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

    # print("Test that equations are consistent with faces")
    # Where do we need these?
    # verify_dc = [
    #     max(
    #         abs(
    #             np.matmul(
    #                 np.insert(affine_pieces["dc"]
    #                           ["faces"][k], d + 1, 1, axis=1),
    #                 np.insert(affine_pieces["dc"]["equations"][k], -1, -1),
    #             )
    #         )
    #     )
    #     for k in range(len(affine_pieces["dc"]["faces"]))
    # ]
    # verify_minus = [
    #     max(
    #         abs(
    #             np.matmul(
    #                 np.insert(
    #                     affine_pieces["convex_minus"]["faces"][k], d + 1, 1, axis=1
    #                 ),
    #                 np.insert(affine_pieces["convex_minus"]
    #                           ["equations"][k], -1, -1),
    #             )
    #         )
    #     )
    #     for k in range(len(affine_pieces["convex_minus"]["faces"]))
    # ]
    # verify_plus = [
    #     max(
    #         abs(
    #             np.matmul(
    #                 np.insert(
    #                     affine_pieces["convex_plus"]["faces"][k], d + 1, 1, axis=1
    #                 ),
    #                 np.insert(affine_pieces["convex_plus"]
    #                           ["equations"][k], -1, -1),
    #             )
    #         )
    #     )
    #     for k in range(len(affine_pieces["convex_plus"]["faces"]))
    # ]

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
    )
    # equations from polytopes
    verify_domains_dc = []
    for k in range(len(affine_pieces["dc"]["polytopes"])):
        mat1 = affine_pieces["dc"]["polytopes"][k]
        mat2 = np.unique(
            ConvexHull(
                np.unique(np.vstack(list_list_faces[k]), axis=0)).equations,
            axis=0,
        )
        diff = mat1 - mat2
        verify_domains_dc.append(abs(diff).max())

    # illustrate_CPWL(
    #     rescaled_data - 1,
    #     rescaled_variable_values,
    #     affine_pieces["dc"]["faces"],
    #     ax=None,
    #     size=5,
    #     colormap="tab20",
    #     alpha=0.4,
    #     exploded_factor=0.4,
    #     show_tick=False,
    # ) # No update in parameter or change in code

    print("Rescale the CPWL results to the original data scale")
    variable_values_original_scale = rescale_variable_values(
        variable_values, slopes, intercepts
    )

    print("Plot the pieces in their original scaling")
    _, slopes3, intercepts3 = rescale_data(data, between_one_and_two=False)
    rescaled_faces = rescale_faces(
        affine_pieces["dc"]["faces"], slopes3, intercepts3)
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
    )

    zPWL_eval = evaluate_DC_CPWL_function(
        variable_values_original_scale, data[:, :-1])

    plt.plot(data[:, -1], zPWL_eval, "o", label="CPWL evaluation")
    error_eval = abs(zPWL_eval - data[:, -1])

    plt.show() # Plot figures without interupting code execution

# %% Iterative test, add planes

# N_plus, N_minus = 1,1
max_error = 0.5
extended_variable_values = None

N_plus_minus_list = np.array(
    [[1, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5]])


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
    tight_parameters = get_tight_parameters(
        rescaled_data, affine_set, max_slope=10)

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


    # %% Iterative test, add points

    extended_variable_values = None

    size_subset = 40

    farthest_point_indices = find_all_farthest_point_sampling(
        rescaled_data, from_domain=True, index_initial_point=0
    )

    rescaled_data1 = rescaled_data[farthest_point_indices[:size_subset], :]

    print("Calculate all combinations of affine functions and the tight parameters")
    affine_set = find_affine_set(rescaled_data1, rescaled_error)
    tight_parameters = get_tight_parameters(
        rescaled_data1, affine_set, max_slope=100)

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
    tight_parameters = get_tight_parameters(
        rescaled_data2, affine_set, max_slope=100)

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
    )

plt.show()