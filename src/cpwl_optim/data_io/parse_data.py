import numpy as np
import pandas as pd


def load_case1_data(naxis: int = 9):
    """
    Data for z = f(x, y) = y*sin(x)

    naxis: number of points along each axis (naxis^ndim total points)
    data: input and output data in the form of (x, y, z)
    """

    x = np.linspace(-1, 1, naxis)
    xy = np.array(np.meshgrid(x, x)).reshape(2, -1).T
    xy = xy + (0.5 / (naxis - 1)) * np.random.rand(xy.shape[0], xy.shape[1])
    z = xy[:, 0] * np.sin(np.pi * xy[:, 1] / 2 + 0.5 * np.pi)
    data = np.c_[xy, z]
    descr = "ysinx"

    return data, descr


def load_case2_data(path: str = "data/crystalhydro.xlsx"):
    """
    Read data for Crystal HydroPower Plant

    path: path to data file
    data: input and output data in the form of (x, y, z)
    """

    try:
        data = pd.read_excel(path, index_col=0)
    except Exception as e:
        print(f"Error reading data from {path}: {e}")
        raise
    # data = pd.read_excel(path, index_col=0)
    data = (data.values)[:, -3:]
    data = data[~np.isnan(data).any(axis=1), :]

    if len(data) == 0:
        raise ValueError(
            f"No valid data found in {path} after removing rows with NaN values."
        )
    descr = "crystalhydro"

    return data, descr


def load_case3_data(naxis: int = 4):
    """
    Data for z = f(w, x, y) = w^2 + x^2 + y^2

    :param naxis: Description
    """

    x = np.array(np.meshgrid(*tuple([np.linspace(-1, 1, naxis)] * 3))).reshape(3, -1).T
    x = x + (0.5 / (naxis - 1)) * np.random.rand(x.shape[0], x.shape[1])
    z = (x**2).sum(axis=1)
    data = np.c_[x, z]
    descr = "sum_of_squares3d"

    return data, descr
