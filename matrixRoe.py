# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:17:35 2024

@author: leonardoab
"""

# import numpy as np
# import matrix


# def compute_jacobian(U, alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol):
#     A = derivative.DFduj(alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol)
#     return A

# def compute_roe_matrix(U_L, U_R, alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol):
#     RoeMatrix = np.zeros((3, 3))

#     # Pontos e pesos da quadratura de Gauss-Legendre
#     quad_points = [-0.7745966692, 0.0, 0.7745966692]
#     quad_weights = [0.5555555556, 0.8888888889, 0.5555555556]

#     # Realiza a quadratura de Gauss-Legendre
#     for i in range(3):
#         xi = quad_points[i]
#         dxi = quad_weights[i]

#         # Calcula a matriz Jacobiana A no ponto U(xi)
#         F_U = compute_jacobian(U_L + xi * (U_R - U_L), alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol)

#         # Acumula a contribuição para RoeMatrix
#         RoeMatrix += dxi * F_U

#     return RoeMatrix

import numpy as np
import jacobianMatrix

def calculate_roe_matrix(alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol, num_points=3):
    """
    Calculate the Roe matrix using Gauss-Legendre quadrature.

    Parameters:
    - alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol: Parameters needed for DFduj.
    - num_points: Number of quadrature points (default is 3).

    Returns:
    - Roe matrix.
    """

    # Gauss-Legendre quadrature weights and points
    weights = np.array([5/18, 8/18, 5/18])
    points = np.array([1/2 - np.sqrt(15)/10, 1/2, 1/2 + np.sqrt(15)/10])

    # Initialize the Roe matrix
    Roe_matrix = np.zeros_like(jacobianMatrix.DFduj(alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol))

    # Pre-calculate Jacobian at each grid point
    Jacobians = []
    for xi in points:
        u_interpolated = ul + xi * (ug - ul)
        J = jacobianMatrix.DFduj(alpha, rhol, rhog, u_interpolated, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol)
        Jacobians.append(J)

    # Calculate the Roe matrix
    for i in range(num_points):
        Roe_matrix += weights[i] * Jacobians[i]

    # Verificar se a matriz contém valores finitos e não NaNs
    # if not np.all(np.isfinite(Roe_matrix)):
    #     Roe_matrix = np.zeros_like(Roe_matrix)  # Substituir por uma matriz de zeros se houver valores não finitos

    return Roe_matrix


# # Example usage:
# alpha = 0.5
# rhol = 1.0
# rhog = 0.5
# ul = 10.0
# ug = 20.0
# Cl = 1.0
# Cg = 2.0
# theta = 0.0
# DH = 1.0
# AREA = 1.0
# EPS = 0.1
# G = 9.81
# MUL = 1.0
# MUG = 2.0
# sigma = 0.01
# w_u = 0.1
# w_rho = 0.2
# w_P = 0.3
# tol = 1e-6

# # Calculate the Roe matrix
# roe_matrix = calculate_roe_matrix(alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol)
# print("Roe Matrix:")
# print(roe_matrix)
