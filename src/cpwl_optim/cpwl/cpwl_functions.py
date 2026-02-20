# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""


from typing import Literal
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linprog, linear_sum_assignment
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt
from .data_scaler import rescale_equations, rescale_faces, rescale_polytopes
import numpy as np


def evaluate_DC_CPWL_function(cpwl_param: dict, x: np.ndarray) -> np.ndarray:

    if not isinstance(cpwl_param, dict):
        raise TypeError(
            f"`cpwl_param` must be a dictionary, got {type(cpwl_param).__name__}."
        )
    # if not isinstance(x, np.ndarray):
    #     raise TypeError(f"`x` must be a numpy.ndarray, got {type(x).__name__}.")

    aPlus = cpwl_param["aPlus"] * 1
    aMinus = cpwl_param["aMinus"] * 1
    bPlus = cpwl_param["bPlus"] * 1
    bMinus = cpwl_param["bMinus"] * 1

    zPlus = (np.matmul(aPlus, x.T) + bPlus.reshape(-1, 1)).max(axis=0)
    zMinus = (np.matmul(aMinus, x.T) + bMinus.reshape(-1, 1)).max(axis=0)
    zPWL = zPlus - zMinus

    return zPWL


def find_all_farthest_point_sampling(
    data: np.ndarray, from_domain: bool = True, index_initial_point: int = 0
) -> list:
    # Type checks
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    # if not isinstance(from_domain, bool):
    #     raise TypeError(
    #         f"`from_domain` must be a boolean, got {type(from_domain).__name__}."
    #     )
    if not isinstance(index_initial_point, (int, np.integer)):
        raise TypeError(
            f"`index_initial_point` must be an integer, got {type(index_initial_point).__name__}."
        )

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
        distance_from_closest_point = matrix_dist[farthest_point_indices, :].min(axis=0)
        new_farthest_point = distance_from_closest_point.argmax()
        farthest_point_indices.append(new_farthest_point)
        number_of_visited_points += 1

    return farthest_point_indices


def find_affine_pieces(var_dict: dict, max_z=1e4):
    if not isinstance(var_dict, dict):
        raise TypeError(
            f"`var_dict` must be a dictionary, got {type(var_dict).__name__}."
        )

    aPlus = var_dict["aPlus"] * 1
    aMinus = var_dict["aMinus"] * 1
    bPlus = var_dict["bPlus"] * 1
    bMinus = var_dict["bMinus"] * 1

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
    boundary_constraints[(d + 1) : 2 * d + 1, :d] = -np.eye(d)

    feasible_point = np.append(np.full(d, 0.5), 0.5 * max_z)

    # find affine pieces for each convex components
    # "polytopes" is the list of piece's domain equations
    # "faces" is the list of piece's list of points
    # "equations" is the list of piece's linear coeff (A,b in z = A*x + b)
    convex_component = []
    for half_space in [half_space_plus, half_space_minus]:
        half_space = np.r_[half_space, boundary_constraints]
        half_space_intersection = HalfspaceIntersection(half_space, feasible_point)
        vertices = half_space_intersection.intersections
        convex_hull = ConvexHull(vertices)
        equation_faces, face_id = np.unique(
            convex_hull.equations, return_inverse=True, axis=0
        )
        N_faces = len(equation_faces)
        vertices_to_discard = np.where(vertices[:, -1] > 0.5 * max_z)[0]
        simplices = convex_hull.simplices
        simplices_to_discard = np.isin(simplices, vertices_to_discard).any(axis=1)
        faces_to_discard = np.unique(face_id[simplices_to_discard])
        faces_to_keep = np.where(~np.isin(np.arange(N_faces), faces_to_discard))[0]
        set_polytopes, set_faces, set_equations = [], [], []
        for f in faces_to_keep:
            simplex_ids = np.where(face_id == f)[0]
            point_ids = np.unique(simplices[simplex_ids])
            convex_hull_domain = ConvexHull(vertices[point_ids, :-1])
            equations_domains = np.unique(convex_hull_domain.equations, axis=0)
            set_polytopes.append(equations_domains)
            # set_polytopes.append(convex_hull_domain.equations)
            set_faces.append(vertices[point_ids[convex_hull_domain.vertices], :])
            # A,b in z = A*x + b
            linear_coeffs = np.delete(-equation_faces[f], -2) / equation_faces[f][-2]
            set_equations.append(linear_coeffs)
            # set_equations.append(equation_faces[f])
        convex_component.append(
            {"polytopes": set_polytopes, "faces": set_faces, "equations": set_equations}
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
                np.linalg.norm(half_space[:, :-1], axis=1), (half_space.shape[0], 1)
            )
            A = np.c_[(half_space[:, :-1], norm_vector)]
            b = -half_space[:, -1:]
            sol = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            if not sol.success:
                continue
            interior_point = (sol.x)[:-1]
            half_space_intersection = HalfspaceIntersection(half_space, interior_point)
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
                z_values = evaluate_DC_CPWL_function(var_dict, vertices)
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


def transform_affine_pieces(affine_pieces: dict, slopes, intercepts):
    if not isinstance(affine_pieces, dict):
        raise TypeError(
            f"`affine_pieces` must be a dictionary, got {type(affine_pieces).__name__}."
        )

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


def check_validity_affine_pieces(affine_pieces: dict):
    if not isinstance(affine_pieces, dict):
        raise TypeError(
            f"`affine_pieces` must be a dictionary, got {type(affine_pieces).__name__}."
        )

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
            equation_domain = np.unique(ConvexHull(face[:, :-1]).equations, axis=0)
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
    data: np.ndarray,
    var_dict: dict,
    faces: list,
    ax: matplotlib.axes.Axes = None,
    size: float = 5,
    colormap: str | None = None,
    alpha: float = 0.4,
    exploded_factor: float = 0.0,
    show_tick: bool = True,
    show_points: bool = True,
    show_error: bool = True,
    savefig: bool = False,
    path_savefig: str = "function_description",
    fig_format: Literal["jpg", "png", "jpeg", "pdf", "svg"] = "png",
):
    # Type checks
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if not isinstance(var_dict, dict):
        raise TypeError(
            f"`var_dict` must be a dictionary, got {type(var_dict).__name__}."
        )
    if not isinstance(faces, list) or not all(
        isinstance(face, np.ndarray) for face in faces
    ):
        raise TypeError(
            f"`faces` must be a list of (N, 3) arrays, got {type(faces).__name__} with elements of type {[type(face).__name__ for face in faces]}."
        )
    if alpha < 0 or alpha > 1:
        raise ValueError(f"`alpha` must be between 0 and 1, got {alpha}.")
    if exploded_factor < 0:
        raise ValueError(
            f"`exploded_factor` must be non-negative, got {exploded_factor}."
        )

    d = data.shape[1] - 1
    N = data.shape[0]
    if d == 2:
        if ax is None:
            plt.figure(figsize=(8, 8), facecolor="w")
            ax = plt.axes(projection="3d")
        if show_points:
            ax_points = ax.scatter(*zip(*data), c="r", s=size)
        if show_error:
            z_PWL = evaluate_DC_CPWL_function(var_dict, data[:, :2])
            for i in range(N):
                ax.plot([data[i, 0]] * 2, [data[i, 1]] * 2, [data[i, 2], z_PWL[i]], "k")
        if colormap is None:
            face_colors = "C0"
        else:
            cmap = matplotlib.colormaps[colormap]
            num_faces = len(faces)
            face_colors = face_colors = [cmap(i / num_faces) for i in range(num_faces)]
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
        if savefig:
            plt.savefig(
                f"{path_savefig}_cpwl_approximation.{fig_format}",
                dpi=300,
                bbox_inches="tight",
            )
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
                vertices_id_from_eq = np.unique(simplices[idx_uniques == eq_id, :])
                face_points = points[vertices_id_from_eq, :]
                # Sort vertices Counter-Clockwise
                # Project to 2D by dropping the coord with max normal component
                n = equations[np.where(idx_uniques == eq_id)[0][0], :3]  # Face normal
                drop_axis = np.argmax(np.abs(n))
                proj = np.delete(face_points, drop_axis, axis=1)  # Project to 2D
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
                    face + exploded_factor * list_vector_spread[k].reshape(-1, 3)
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
            z_PWL = evaluate_DC_CPWL_function(var_dict, data[:, :3])
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

        if savefig:
            plt.savefig(
                f"{path_savefig}_cpwl_approximation.{fig_format}",
                dpi=300,
                bbox_inches="tight",
            )

        return list_list_faces
