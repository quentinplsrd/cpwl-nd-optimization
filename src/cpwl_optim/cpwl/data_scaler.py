import numpy as np
from typing import Iterable


def rescale_data(data: np.ndarray, between_one_and_two: bool = True):
    """Rescale the data set.

    Args:
      data: The data set to rescale.

      between_one_and_two: If the data should be rescaled to the range [1,2] instead of [0,1].

    Returns:
      The rescaled data, the transformation coefficients (slopes and intercepts).
      `data` can be recovered by applying: slopes*rescaled_data + intercepts

    """

    # --- Type checks ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"`data` must be a 2D array, got shape {data.shape}.")
    if not isinstance(between_one_and_two, bool):
        raise TypeError("`between_one_and_two` must be a boolean.")

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    rescaled_data = (data - data_min) / (data_max - data_min)
    rescaled_data += 1 * between_one_and_two

    slopes = data_max - data_min
    intercepts = data_min - between_one_and_two * slopes

    return rescaled_data, slopes, intercepts


def invert_transform(slopes, intercepts):
    return 1 / slopes, -intercepts / slopes


def rescale_faces(list_faces: Iterable[np.ndarray], slopes, intercepts):
    # --- Type and value checks ---
    if not isinstance(list_faces, Iterable):
        raise TypeError(
            f"`list_faces` must be a list, got {type(list_faces).__name__}."
        )
    if len(list_faces) < 1:
        raise ValueError("`list_faces` must contain at least one face.")

    list_rescaled_faces = []
    for face in list_faces:
        list_rescaled_faces.append(face * slopes + intercepts)

    return list_rescaled_faces


def rescale_equations(list_equations: Iterable, slopes, intercepts):
    if len(list_equations) < 1:
        raise ValueError("`list_equations` must contain at least one equation.")

    list_rescaled_equations = []
    a_z = slopes[-1]
    a_x = slopes[:-1]
    b_z = intercepts[-1]
    b_x = intercepts[:-1]

    for equation in list_equations:
        coeff_piece = equation[:-1]
        intercept_piece = equation[-1]
        new_coeff_piece = coeff_piece * (a_z / a_x)
        new_intercept_piece = (
            intercept_piece * a_z + b_z - sum(coeff_piece * b_x / a_x) * a_z
        )
        list_rescaled_equations.append(np.r_[new_coeff_piece, new_intercept_piece])

    return list_rescaled_equations


def rescale_polytopes(list_polytopes: Iterable[np.ndarray], slopes, intercepts):
    if len(list_polytopes) < 1:
        raise ValueError("`list_polytopes` must contain at least one polytope.")

    list_rescaled_polytopes = []
    a_x = slopes[:-1].reshape(1, -1)
    b_x = intercepts[:-1].reshape(1, -1)

    for polytope in list_polytopes:
        polytope_coeff = polytope[:, :-1]
        polytope_intercept = polytope[:, [-1]]
        new_polytope_coeff = (1 / a_x) * polytope_coeff
        new_polytope_intercept = (polytope_intercept - polytope_coeff) @ (b_x / a_x).T
        list_rescaled_polytopes.append(
            np.c_[new_polytope_coeff, new_polytope_intercept]
        )

    return list_rescaled_polytopes


def rescale_variable_values(variable_values: dict, slopes, intercepts):
    if not isinstance(variable_values, dict):
        raise TypeError(
            f"`variable_values` must be a dictionary, got {type(variable_values).__name__}."
        )

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

    aPlus = aPlus * slopes[-1] / slopes[:-1]
    aMinus = aMinus * slopes[-1] / slopes[:-1]
    bPlus = bPlus * slopes[-1] + intercepts[-1] - (aPlus * intercepts[:-1]).sum(axis=1)
    bMinus = bMinus * slopes[-1] - (aMinus * intercepts[:-1]).sum(axis=1)

    error = slopes[-1] * error if "error" in variable_values else None
    zPWL = slopes[-1] * zPWL + intercepts[-1] if "zPWL" in variable_values else None
    # TODO: to verify
    zPlus = slopes[-1] * zPlus + intercepts[-1] if "zPlus" in variable_values else None
    zMinus = slopes[-1] * zMinus if "zMinus" in variable_values else None

    rescaled_variable_values = {
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

    return rescaled_variable_values
