# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:40:10 2025

@author: qploussard
"""

import time
from itertools import product, combinations
from typing import Optional, Literal
import numpy as np


#%% functions

def find_plane_equation(points_array):
    """
    Calculates the equation of a plane (ax + by + cz = d) given at least
    three non-collinear points.

    Args:
        points_array (np.array): An Nx3 numpy array where N >= 3,
                                 representing the points (x, y, z).

    Returns:
        tuple: A tuple (a, b, c, d) representing the plane equation coefficients.
               Returns (None, None, None, None) if fewer than 3 points are provided.
    """
    if points_array.shape[0] < 3:
        print("Error: At least 3 points are required to define a plane.")
        return None, None, None, None

    # Use the first three points for calculation
    P1 = points_array[0]
    P2 = points_array[1]
    P3 = points_array[2]

    # 1. Form two vectors in the plane
    v1 = P3 - P1
    v2 = P2 - P1

    # 2. Calculate the normal vector (a, b, c) using the cross product
    # The normal vector is perpendicular to the plane.
    normal_vector = np.cross(v1, v2)
    a, b, c = normal_vector

    # Check for collinearity (cross product will be a zero vector)
    if np.allclose(normal_vector, [0, 0, 0]):
        print("Error: The first three points are collinear (lie on a line). Cannot define a unique plane.")
        return None, None, None, None

    # 3. Calculate the constant 'd' using the dot product with the first point P1
    # d is equal to the dot product of the normal vector and any point on the plane.
    d = np.dot(normal_vector, P1)

    return np.array([a, b, c, -d])


# careful, the number of affine sets is equal to 'N choose d+1'*2^(d+1)
def find_affine_set(data: np.ndarray,
                    max_error: float):
    
    """Find the set of affine functions associated to every combination of d+1 points and every point-wise distance.

    Args:
      data: The data set. 
      
      max_error: The distance of each affine function to its associated points.
      
    Returns:
      The list of affine functions as a 2D array. 
      The rows represent the affine functions and the columns represent the linear coefficients of the affine functions.
      The last column is the bias term.
      
    """    

    # --- Type checks ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {data.shape}.")
    if not isinstance(max_error, (float, np.floating)):
        raise TypeError(f"`max_error` must be a float, got {type(max_error).__name__}.")

    # --- Value checks ---
    if max_error < 0:
        raise ValueError("`max_error` must be nonnegative.")
    
    N_points, d_plus_1 = data.shape
    d = d_plus_1 - 1
    x = data[:,:-1]
    x_1 = np.c_[x,np.ones(N_points)]
    z = data[:,-1]
    simplex_comb = np.array(list(combinations(range(N_points), d+1)))
    list_matrix = x_1[simplex_comb,:]
    list_z = z[simplex_comb]
    list_inv_matrix = np.linalg.inv(list_matrix)
    error_comb = max_error*np.array(list(product([-1, 1], repeat=(d+1))))
    z_plus_errors = list_z[:, np.newaxis, :] + error_comb[np.newaxis, :, :]  # (n_simplex, n_error, d+1)
    affine_set = np.einsum('sij,sej->sei', list_inv_matrix, z_plus_errors).reshape(-1, d+1)
    
    return affine_set


def get_tight_parameters(data: np.ndarray,
                         affine_set: np.ndarray,
                         max_slope: Optional[float] = None) -> dict:
    
    """Calculate the tightening parameters to be used in the MILP CPWL approximation.

    Args:
      data: The data set. 
      
      affine_set: A set of affine functions.
      
      max_slope: A maximum slope to eliminate some affine functions.

    Returns:
      A dictionary of tightening parameters.
      
    """  
    
    # --- Type checks ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {data.shape}.")
    if not isinstance(affine_set, np.ndarray):
        raise TypeError(f"`affine_set` must be a numpy.ndarray, got {type(affine_set).__name__}.")
    if affine_set.ndim != 2:
        raise ValueError(f"`affine_set` must be 2D, got shape {affine_set.shape}.")
    if max_slope is not None:
        if not isinstance(max_slope, (int, float, np.floating)):
            raise TypeError(f"`max_slope` must be a float or int, got {type(max_slope).__name__}.")

    # --- Value checks ---
    if max_slope is not None:
        if max_slope <= 0:
            raise ValueError("`max_slope` must be positive.")
    
    N_points = len(data)
    x = data[:,:-1]
    x_1 = np.c_[x,np.ones(N_points)]
    affine_set_filtered = affine_set*1
    if max_slope:
        filter_slope = np.all(abs(affine_set[:,:-1])<=max_slope,axis=1)
        affine_set_filtered = affine_set[filter_slope,:]
    max_a_b = affine_set_filtered.max(axis=0)
    min_a_b = affine_set_filtered.min(axis=0)
    fx_n = np.matmul(affine_set_filtered,x_1.T)
    max_fx_n = fx_n.max(axis=0)
    min_fx_n = fx_n.min(axis=0)
    tight_parameters = {'max_a_b': max_a_b,
                        'min_a_b': min_a_b,
                        'max_fx_n': max_fx_n,
                        'min_fx_n': min_fx_n}
    
    return tight_parameters