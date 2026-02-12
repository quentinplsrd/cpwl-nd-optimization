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


def evaluate_DC_CPWL_function(CPWL_parameters,x):
    
    aPlus = CPWL_parameters['aPlus']*1
    aMinus = CPWL_parameters['aMinus']*1
    bPlus = CPWL_parameters['bPlus']*1
    bMinus = CPWL_parameters['bMinus']*1
    
    zPlus = (np.matmul(aPlus,x.T) + bPlus.reshape(-1,1)).max(axis=0)
    zMinus = (np.matmul(aMinus,x.T) + bMinus.reshape(-1,1)).max(axis=0)
    zPWL = zPlus - zMinus
    
    return zPWL


def rescale_variable_values(variable_values, slopes, intercepts):
    
    aPlus = variable_values['aPlus']*1
    aMinus = variable_values['aMinus']*1
    bPlus = variable_values['bPlus']*1
    bMinus = variable_values['bMinus']*1
    
    decPlus = variable_values['decPlus']*1 if 'decPlus' in variable_values else None
    decMinus = variable_values['decMinus']*1 if 'decMinus' in variable_values else None
    zPWL = variable_values['zPWL']*1 if 'zPWL' in variable_values else None
    zPlus = variable_values['zPlus']*1 if 'zPlus' in variable_values else None
    zMinus = variable_values['zMinus']*1 if 'zMinus' in variable_values else None
    error = variable_values['error']*1 if 'error' in variable_values else None
    
    aPlus = aPlus*slopes[-1]/slopes[:-1]
    aMinus = aMinus*slopes[-1]/slopes[:-1]
    bPlus = bPlus*slopes[-1] + intercepts[-1] - (aPlus*intercepts[:-1]).sum(axis=1)
    bMinus = bMinus*slopes[-1] - (aMinus*intercepts[:-1]).sum(axis=1)
    
    error = slopes[-1]*error if 'error' in variable_values else None
    zPWL = slopes[-1]*zPWL + intercepts[-1] if 'zPWL' in variable_values else None
    # TODO: to verify
    zPlus = slopes[-1]*zPlus + intercepts[-1] if 'zPlus' in variable_values else None
    zMinus = slopes[-1]*zMinus if 'zMinus' in variable_values else None
    
    rescaled_variable_values = {'aPlus': aPlus,
                                'aMinus': aMinus,
                                'bPlus': bPlus,
                                'bMinus': bMinus,
                                'decPlus': decPlus,
                                'decMinus': decMinus,
                                'zPWL': zPWL,
                                'zPlus': zPlus,
                                'zMinus': zMinus,
                                'error': error}
    
    return rescaled_variable_values


def add_pieces_to_solution(variable_values, add_plus = 0, add_minus = 0):
    
    aPlus = variable_values['aPlus']*1
    aMinus = variable_values['aMinus']*1
    bPlus = variable_values['bPlus']*1
    bMinus = variable_values['bMinus']*1
    decPlus = variable_values['decPlus']*1 if 'decPlus' in variable_values else None
    decMinus = variable_values['decMinus']*1 if 'decMinus' in variable_values else None
    zPWL = variable_values['zPWL']*1 if 'zPWL' in variable_values else None
    zPlus = variable_values['zPlus']*1 if 'zPlus' in variable_values else None
    zMinus = variable_values['zMinus']*1 if 'zMinus' in variable_values else None
    error = variable_values['error']*1 if 'error' in variable_values else None
    
    aPlus = np.r_[aPlus, np.repeat(aPlus[[-1],:], add_plus, axis=0)]
    aMinus = np.r_[aMinus, np.repeat(aMinus[[-1],:], add_minus, axis=0)]
    bPlus = np.r_[bPlus, np.repeat(bPlus[-1], add_plus)]
    bMinus = np.r_[bMinus, np.repeat(bMinus[-1], add_minus)]
    decPlus = np.c_[decPlus, np.repeat(decPlus[:,[-1]], add_plus, axis=1)]
    decMinus = np.c_[decMinus, np.repeat(decMinus[:,[-1]], add_minus, axis=1)]
    
    extended_variable_values = {'aPlus': aPlus,
                                'aMinus': aMinus,
                                'bPlus': bPlus,
                                'bMinus': bMinus,
                                'decPlus': decPlus,
                                'decMinus': decMinus,
                                'zPWL': zPWL,
                                'zPlus': zPlus,
                                'zMinus': zMinus,
                                'error': error}
    
    return extended_variable_values


def find_all_farthest_point_sampling(data, 
                                     from_domain=True,
                                     index_initial_point=0):
    
    N = data.shape[0]
    if from_domain:
        data = data[:,:-1]
        
    if index_initial_point is None:
        index_initial_point = np.random.randint(N)
    index_initial_point %= N
    
    # calculate pair-wise distance
    matrix_dist = squareform(pdist(data, metric='euclidean'))

    farthest_point_indices = [index_initial_point]
    number_of_visited_points = 1
    
    while number_of_visited_points < N:
        distance_from_closest_point = matrix_dist[farthest_point_indices,:].min(axis=0)
        new_farthest_point = distance_from_closest_point.argmax()
        farthest_point_indices.append(new_farthest_point)
        number_of_visited_points += 1
        
    return farthest_point_indices
        

def add_points_to_solution(variable_values, new_points):
    
    aPlus = variable_values['aPlus']*1
    aMinus = variable_values['aMinus']*1
    bPlus = variable_values['bPlus']*1
    bMinus = variable_values['bMinus']*1
    decPlus = variable_values['decPlus']*1 if 'decPlus' in variable_values else None
    decMinus = variable_values['decMinus']*1 if 'decMinus' in variable_values else None
    zPWL = variable_values['zPWL']*1 if 'zPWL' in variable_values else None
    zPlus = variable_values['zPlus']*1 if 'zPlus' in variable_values else None
    zMinus = variable_values['zMinus']*1 if 'zMinus' in variable_values else None
    error = variable_values['error']*1 if 'error' in variable_values else None
    
    N_plus = len(bPlus)
    N_minus = len(bMinus)
    
    x = new_points[:,:-1]
    z = new_points[:,-1]
    
    new_zPlus_disag = np.matmul(aPlus,x.T) + bPlus.reshape(-1,1)
    new_zMinus_disag = np.matmul(aMinus,x.T) + bMinus.reshape(-1,1)
    
    new_zPlus = new_zPlus_disag.max(axis=0)
    new_zMinus = new_zMinus_disag.max(axis=0)
    new_zPWL = new_zPlus - new_zMinus
    new_error = abs(new_zPWL - z)
    new_decPlus = np.eye(N_plus)[new_zPlus_disag.argmax(axis=0)]
    new_decMinus = np.eye(N_minus)[new_zMinus_disag.argmax(axis=0)]
    
    zPlus = np.r_[zPlus,new_zPlus]
    zMinus = np.r_[zMinus,new_zMinus]
    zPWL = np.r_[zPWL,new_zPWL]
    error = np.r_[error,new_error]
    decPlus = np.r_[decPlus,new_decPlus]
    decMinus = np.r_[decMinus,new_decMinus]
    
    extended_variable_values = {'aPlus': aPlus,
                                'aMinus': aMinus,
                                'bPlus': bPlus,
                                'bMinus': bMinus,
                                'decPlus': decPlus,
                                'decMinus': decMinus,
                                'zPWL': zPWL,
                                'zPlus': zPlus,
                                'zMinus': zMinus,
                                'error': error}
    
    return extended_variable_values


def rescale_faces(list_faces, slopes, intercepts):
    
    list_rescaled_faces = []
    for face in list_faces:
        list_rescaled_faces.append(face*slopes + intercepts)
        
    return list_rescaled_faces


def rescale_equations(list_equations, slopes, intercepts):
    
    list_rescaled_equations = []
    a_z = slopes[-1]
    a_x = slopes[:-1]
    b_z = intercepts[-1]
    b_x = intercepts[:-1]
    
    for equation in list_equations:
        coeff_piece = equation[:-1]
        intercept_piece = equation[-1]
        new_coeff_piece = coeff_piece*(a_z/a_x)
        new_intercept_piece = intercept_piece*a_z + b_z - sum(coeff_piece*b_x/a_x)*a_z
        list_rescaled_equations.append(np.r_[new_coeff_piece,new_intercept_piece])

    return list_rescaled_equations


def rescale_polytopes(list_polytopes, slopes, intercepts):
    
    list_rescaled_polytopes = []
    a_x = slopes[:-1].reshape(1,-1)
    b_x = intercepts[:-1].reshape(1,-1)
    
    for polytope in list_polytopes:
        polytope_coeff = polytope[:,:-1]
        polytope_intercept = polytope[:,[-1]]
        new_polytope_coeff = (1/a_x)*polytope_coeff
        new_polytope_intercept = polytope_intercept - polytope_coeff @ (b_x/a_x).T
        list_rescaled_polytopes.append(np.c_[new_polytope_coeff,new_polytope_intercept])
        
    return list_rescaled_polytopes


def find_affine_pieces(variable_values, max_z=1e4):
    
    aPlus = variable_values['aPlus']*1
    aMinus = variable_values['aMinus']*1
    bPlus = variable_values['bPlus']*1
    bMinus = variable_values['bMinus']*1
    
    N_plus = len(bPlus)
    N_minus = len(bMinus)
    d = aPlus.shape[1]
    
    half_space_plus = np.c_[aPlus,-np.ones(N_plus),bPlus]
    half_space_minus = np.c_[aMinus,-np.ones(N_minus),bMinus]
    
    # we assume the domain is [0,1]^d
    boundary_constraints = np.zeros((2*d+1, d+2))
    boundary_constraints[:(d+1), :(d+1)] = np.eye(d+1)
    boundary_constraints[:d, d+1] = -1.
    boundary_constraints[d, d+1] = -max_z
    boundary_constraints[(d+1):2*d + 1, :d] = -np.eye(d)

    feasible_point = np.append(np.full(d, 0.5), 0.5 * max_z)  
    
    # find affine pieces for each convex components
    # "polytopes" is the list of piece's domain equations
    # "faces" is the list of piece's list of points
    # "equations" is the list of piece's linear coeff (A,b in z = A*x + b)
    convex_component = []
    for half_space in [half_space_plus, half_space_minus]:
        half_space = np.r_[half_space,boundary_constraints]
        half_space_intersection = HalfspaceIntersection(half_space, feasible_point)
        vertices = half_space_intersection.intersections
        convex_hull = ConvexHull(vertices)
        equation_faces, face_id = np.unique(convex_hull.equations,return_inverse=True,axis=0)
        N_faces = len(equation_faces)
        vertices_to_discard = np.where(vertices[:,-1] > 0.5 * max_z)[0]
        simplices = convex_hull.simplices
        simplices_to_discard = np.isin(simplices, vertices_to_discard).any(axis=1)
        faces_to_discard = np.unique(face_id[simplices_to_discard])
        faces_to_keep = np.where(~np.isin(np.arange(N_faces),faces_to_discard))[0]
        set_polytopes, set_faces, set_equations = [], [], []
        for f in faces_to_keep:
            simplex_ids = np.where(face_id==f)[0]
            point_ids = np.unique(simplices[simplex_ids])
            convex_hull_domain = ConvexHull(vertices[point_ids,:-1])
            equations_domains = np.unique(convex_hull_domain.equations,axis=0)
            set_polytopes.append(equations_domains)
            # set_polytopes.append(convex_hull_domain.equations)
            set_faces.append(vertices[point_ids[convex_hull_domain.vertices],:])
            # A,b in z = A*x + b
            linear_coeffs = np.delete(-equation_faces[f],-2)/equation_faces[f][-2]
            set_equations.append(linear_coeffs)
            # set_equations.append(equation_faces[f])
        convex_component.append({
            "polytopes": set_polytopes,
            "faces": set_faces,
            "equations": set_equations
            })
    
    # find for the DC function
    set_plus, set_minus = convex_component
    N_plus = len(set_plus["polytopes"])
    N_minus = len(set_minus["polytopes"])
    
    set_polytopes, set_faces, set_equations = [], [], []
    # static parameters to find the interior point via LP
    c = np.zeros(d+1)
    c[-1] = -1
    bounds = [(None, None)]*d + [(0, None)]
    # iterate through all combinations of plus and minus planes
    for j in range(N_plus):
        for k in range(N_minus):
            polytope_plus = set_plus["polytopes"][j]
            polytope_minus = set_minus["polytopes"][k]
            half_space = np.r_[polytope_plus,polytope_minus]
            # find a strictly interior point via LP:  A_ub x ≤ b_ub
            norm_vector = np.reshape(np.linalg.norm(half_space[:, :-1], axis=1),
                                     (half_space.shape[0], 1))
            A = np.c_[(half_space[:, :-1], norm_vector)]
            b = - half_space[:, -1:]
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
                vertices = vertices[convex_hull.vertices,:]
                polytope  = convex_hull.equations
                equations_domains = np.unique(polytope,axis=0)
                set_polytopes.append(equations_domains)
                # set_polytopes.append(polytope)
                z_values = evaluate_DC_CPWL_function(variable_values,vertices)
                face = np.c_[vertices,z_values]
                set_faces.append(face)
                set_equations.append(
                    set_plus["equations"][j] - set_minus["equations"][k]
                    )
    
    affine_pieces = {"convex_plus": set_plus,
                     "convex_minus": set_minus,
                     "dc": {
                         "polytopes": set_polytopes,
                         "faces": set_faces,
                         "equations": set_equations
                         }}
    
    return affine_pieces
    

def transform_affine_pieces(affine_pieces, slopes, intercepts):
    
    transformed_affine_pieces = {}
    
    for type_cpwl in ["convex_plus", "convex_minus", "dc"]:
        transformed_affine_pieces[type_cpwl] = {}
        # rescale faces
        rescaled_faces = rescale_faces(affine_pieces[type_cpwl]["faces"], slopes, intercepts)
        transformed_affine_pieces[type_cpwl]["faces"] = rescaled_faces
        # rescale equations (linear coeffs)
        rescaled_equations = rescale_equations(affine_pieces[type_cpwl]["equations"], slopes, intercepts)
        transformed_affine_pieces[type_cpwl]["equations"] = rescaled_equations
        # rescale polytopes (domain equations)
        rescaled_polytopes = rescale_polytopes(affine_pieces[type_cpwl]["polytopes"], slopes, intercepts)
        transformed_affine_pieces[type_cpwl]["polytopes"] = rescaled_polytopes
        
    return transformed_affine_pieces


def check_validity_affine_pieces(affine_pieces):
    
    validity_report = {}
    d = affine_pieces["dc"]['faces'][0].shape[1]-1
    for type_cpwl in ["convex_plus", "convex_minus", "dc"]:
        validity_report[type_cpwl] = {'polytopes': [],
                                      'equations': []}
        number_pieces = len(affine_pieces[type_cpwl]['faces'])
        for k in range(number_pieces):
            face = affine_pieces[type_cpwl]["faces"][k]
            polytope =  affine_pieces[type_cpwl]["polytopes"][k]
            polytope = polytope/np.linalg.norm(polytope[:,:-1],axis=1).reshape(-1,1)
            polytope = np.unique(polytope,axis=0)
            equation = affine_pieces[type_cpwl]["equations"][k]
            equation_domain = np.unique(ConvexHull(face[:,:-1]).equations,axis=0)
            dists = cdist(polytope, equation_domain, metric='cityblock')
            row_ind, col_ind = linear_sum_assignment(dists)
            equation_domain = equation_domain[col_ind,:]
            diff_polytope = abs(polytope - equation_domain).max()
            diff_equation = np.insert(face,d+1,1,axis=1) @ np.insert(equation,-1,-1).reshape(-1,1)
            diff_equation = abs(diff_equation).max()
            validity_report[type_cpwl]['polytopes'].append(diff_polytope)
            validity_report[type_cpwl]['equations'].append(diff_equation)
            
    return validity_report
            
