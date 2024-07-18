# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 06:57:27 2024

@author: leonardoab
"""

import numpy as np
import math
from math import pi, sinh
import catenary
import find
import matrixRoe
import frictionFactor
import steadyState
import boundaryConditions
import matplotlib.pyplot as plt

def calculate_boundary_conditions(U0, Un, nCl, nCg, nrho_l0, nP_l0, dot_M_l0, dot_M_g0, AREA, Lp, Lr):
    # Condições de entrada (boundary conditions at inlet)
    alpha_start = U0[1] / U0[0]
    rho_l_start = (1 - alpha_start) * U0[0]
    rho_g_start = alpha_start * U0[0]

    # Velocidades específicas na entrada
    u_l_start = dot_M_l0 / ((1 - alpha_start) * rho_l_start * AREA)
    u_g_start = dot_M_g0 / (alpha_start * rho_g_start * AREA)

    # Pressão na entrada (assumindo uma relação de estado linear)
    P_inlet = U0[2] - 0.5 * (rho_l_start * u_l_start**2 + rho_g_start * u_g_start**2)

    # Condições de saída (boundary conditions at outlet)
    alpha_end = Un[1] / Un[0]
    rho_l_end = (1 - alpha_end) * Un[0]
    rho_g_end = alpha_end * Un[0]

    # Velocidades específicas na saída
    u_l_end = dot_M_l0 / ((1 - alpha_end) * rho_l_end * AREA)
    u_g_end = dot_M_g0 / (alpha_end * rho_g_end * AREA)

    # Pressão na saída (assumindo uma relação de estado linear)
    P_outlet = Un[2] - 0.5 * (rho_l_end * u_l_end**2 + rho_g_end * u_g_end**2)

    return alpha_start, rho_l_start, rho_g_start, alpha_end, rho_l_end, rho_g_end, P_inlet, P_outlet

def apply_boundary_conditions(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, nulv_start, nugv_start, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv_end, nugv_end):
    # Condição de contorno na entrada
    u1_start, u2_start, u3_start = compute_conservative_variables(nPs, nCl, nCg, nrho_l0, nP_l0, alpha_start_dim, rho_l_start_dim, nulv_start, rho_g_start_dim, nugv_start)
    U[0, :] = [u1_start, u2_start, u3_start]

    # Condição de contorno na saída
    u1_end, u2_end, u3_end = compute_conservative_variables(nPs, nCl, nCg, nrho_l0, nP_l0, alpha_end_dim, rho_l_end_dim, nulv_end, rho_g_end_dim, nugv_end)
    U[-1, :] = [u1_end, u2_end, u3_end]



def compute_conservative_variables(P, Cl, Cg, rho_l_0, Pl_0, alpha, rho_l, ul, rho_g, ug):
    u1 = (1 - alpha) * (rho_l_0 + (P - Pl_0) / Cl)
    u2 = alpha * P / (Cg**2)
    u3 = (1 - alpha) * rho_l * ul + alpha * rho_g * ug
    # alpha_safe = np.where(alpha != 0, alpha, 1e-10)
    # rho_g_safe = np.where(rho_g != 0, rho_g, 1e-10)
    # u3 = (1 - alpha_safe) * rho_l * ul + alpha_safe * rho_g_safe * ug
    return u1, u2, u3

def compute_catenary(s, z, Lp, theta_0):
    tol = 1e-5
    # Adicionar verificação para s ser zero
    if s == 0:
        return -theta_0 * (pi / 180)  # Convertendo -theta_0 graus para radianos
    
    A = catenary.catenary_constant(s, z, tol)
    LR = A * sinh(s / A)
    VPARAM = np.array([A, LR])
    
    if s < Lp:
        return -theta_0 * (pi / 180)  # Convertendo -theta_0 graus para radianos
    elif s == Lp:
        return 0  # 0 graus é 0 radianos
    else:
        theta_graus = catenary.Dfungeo(s, VPARAM)
        theta_radianos = theta_graus * (pi / 180)
        return theta_radianos 
    
def compute_primitive(u1, u2, u3):
    alpha = u2 / u1
    rho_l = (1 - alpha) * u1
    rho_g = alpha * u1
    u_l = (1 - alpha) * u3 / rho_l if rho_l != 0 else 0
    u_g = alpha * u3 / rho_g if rho_g != 0 else 0
    P = u3 - (1 - alpha) * rho_l * u_l**2 - alpha * rho_g * u_g**2
    return alpha, rho_l, rho_g, u_l, u_g, P

def compute_primitive_variables(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):
   """
   Retorna as variáveis primitivas uma vez conhecidas as variáveis conservativas.

   Parâmetros:
   u1, u2, u3: Variáveis conservativas
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   alpha: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe (positivo upward e negativo downward)
   Cg: Velocidade do som no gás em m/s
   Cl: Velocidade do som no líquido em m/s
   rho_l0: Valor de referência da densidade do líquido
   P_l0: Valor de referência da pressão do líquido
   AREA: Área seccional da tubulação do pipeline e riser
   D: Diâmetro da tubulação
   EPS: Rugosidade do tubo do pipeline
   G: Aceleração da gravidade
   MUL: Viscosidade dinâmica do líquido
   MUG: Viscosidade dinâmica do gás
   sigma: Tensão superficial líquido-gás
   w_u: Escala de velocidade
   w_rho: Escala de densidade
   tol: Tolerância numérica
   tola: Tolerância para decidir se uj é considerado zero
   maxit: Número máximo de iterações

   Retorna:
   alpha: Fração de vazio
   rhol: Densidade do líquido
   rhog: Densidade do gás
   P: Pressão
   ul: Velocidade do líquido
   ug: Velocidade do gás
   index: Indicador de escoamento (0: estratificado, 1: não estratificado)
   """
   # Obter alpha a partir de u1 e u2
   alpha = find.alpha_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0, tol)

   # Obter a pressão a partir de u1 e u2
   P = find.pressure_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0)

   # Densidades do gás e líquido
   rhog = P / (Cg ** 2)
   rhol = rho_l0 + (P - P_l0) / (Cl ** 2)

   # Velocidade do gás e líquido
   ul, ug, index = find.find_ulug_from_uj_genflw_simp(u1, u2, u3, rhol, rhog, alpha, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit)

   return alpha, rhol, rhog, P, ul, ug, index

def compute_average_primitive_variables(alpha_i, alpha_ip1, rho_l_i, rho_l_ip1, rho_g_i, rho_g_ip1, u_l_i, u_l_ip1, u_g_i, u_g_ip1, P_i, P_ip1):
    alpha_i_star = 0.5 * (alpha_i + alpha_ip1)
    rho_l_i_star = 0.5 * (rho_l_i + rho_l_ip1)
    rho_g_i_star = 0.5 * (rho_g_i + rho_g_ip1)
    u_l_i_star = 0.5 * (u_l_i + u_l_ip1)
    u_g_i_star = 0.5 * (u_g_i + u_g_ip1)
    P_i_star = 0.5 * (P_i + P_ip1)
    return alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star

def compute_mixture_viscosity(alpha, mu_g, mu_l):
    return mu_g * alpha + (1 - alpha) * mu_l

def compute_edia(epsilon, D_H):
    return epsilon / D_H

def compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D, s_i, z, edia, mu_g, mu_l, Lp, theta_0):
    mu_m_i = compute_mixture_viscosity(alpha_i, mu_g, mu_l)
    rho_m_i = (1 - alpha_i) * rho_l_i + alpha_i * rho_g_i
    theta = compute_catenary(s_i, z, Lp, theta_0)
    sin_theta_i = np.sin(theta)
    j_i = (1 - alpha_i) * u_l_i + alpha_i * u_g_i
    
    R_e_m_i = (rho_m_i * j_i * D) / mu_m_i if j_i != 0 else 0
    f_m_i = frictionFactor.FatorAtrito(R_e_m_i, edia)
    return rho_m_i, sin_theta_i, f_m_i, j_i, R_e_m_i

def compute_source_term_vector(U_i, alpha_i, rho_m_i, sin_theta_i, f_m_i, j_i):
    S1_i = S2_i = 0
    S3_i = -rho_m_i * sin_theta_i - 0.5 * rho_m_i * f_m_i * np.abs(j_i) * j_i
    return np.array([S1_i, S2_i, S3_i])

def compute_roe_matrix(U_i, U_ip1, alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol):
    Roe_matrix = matrixRoe.calculate_roe_matrix(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol)
    # Verificar se a matriz contém valores finitos e não NaNs
    # if not np.all(np.isfinite(Roe_matrix)):
    #     Roe_matrix = np.zeros_like(Roe_matrix)
    return Roe_matrix

def compute_average_conservative_variables(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star):
    return np.array([(1 - alpha_i_star) * rho_l_i_star, alpha_i_star * rho_g_i_star, (1 - alpha_i_star) * rho_l_i_star * u_l_i_star + alpha_i_star * rho_g_i_star * u_g_i_star])

def compute_average_flux_vector(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star):
    return np.array([(1 - alpha_i_star) * rho_l_i_star * u_l_i_star,
                      alpha_i_star * rho_g_i_star * u_g_i_star,
                      (1 - alpha_i_star) * rho_l_i_star * u_l_i_star**2 + alpha_i_star * rho_g_i_star * u_g_i_star**2 + P_i_star])

def roe_riemann_solver(F_i_star, F_ip1_star, U_i_star, U_ip1_star, Roe_matrix):
    U_diff = U_ip1_star - U_i_star
    return 0.5 * (F_i_star + F_ip1_star) - 0.5 * np.dot(Roe_matrix, U_diff)

def compute_time_step_lax_wendroff(CFL, delta_x, Roe_matrix):
    if not np.all(np.isfinite(Roe_matrix)):
        max_lambda = 1.0  
    else:
        eigenvalues = np.linalg.eigvals(Roe_matrix)
        max_lambda = max(np.abs(eigenvalues))
    if max_lambda == 0:
        max_lambda = 1.0
    
    return CFL * delta_x / (2 * max_lambda)

def update_solution_at_interface(U, numerical_flux, source_term, delta_t, delta_x):
    return U + (delta_t / delta_x) * (numerical_flux - source_term)

def calcular_catenaria(X, Z, tol):
    CA = catenary.catenary_constant(X, Z, tol)
    Lr = CA * np.sinh(X / CA)
    return CA, Lr

def adimensionalizar(Cg, Cl, P_l0, rho_l0, Ps):
    omega_P = P_l0
    omega_rho = rho_l0
    omega_c = 1 / np.sqrt(omega_rho / omega_P)
    nCg = Cg / omega_c
    nCl = Cl / omega_c
    nP_l0 = P_l0 / omega_P
    nrho_l0 = rho_l0 / omega_rho
    nPs = Ps / omega_P
    return nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho

def gerar_vetores_velocidade_superficial(num_cells):
    jl = [0.001]
    jg = [0.001]
    
    # Definindo os limites para cada intervalo
    limites = [0.01, 0.1, 1, 10]
    incrementos = [5 * (10 ** -4), 5 * (10 ** -3), 5 * (10 ** -2), 5 * (10 ** -1)]
    
    i = 0
    for lim, inc in zip(limites, incrementos):
        while jl[-1] < lim:
            i += 1
            if i >= num_cells:
                break
            jl.append(jl[i - 1] + inc)
            jg.append(jg[i - 1] + inc)
    
    # Garantindo que os vetores tenham o tamanho correto
    jl = np.array(jl[:num_cells])
    jg = np.array(jg[:num_cells])
    
    return jl, jg

def compute_F1(alpha, rho_l, rho_g, u_l, u_g):
    return np.array([(1 - alpha) * rho_l * u_l,
                      alpha * rho_g * u_g,
                      (1 - alpha) * rho_l * u_l**2 + alpha * rho_g * u_g**2])


def calculate_residuals(N, dt, dx, residuals, w_conserv, w_source, U_new, U, tol):
    residual_max = np.zeros(3)
    for i in range(1, N):
        residual_i = np.linalg.norm(residuals[i, :])
        if residual_i > tol:
            for j in range(3):
                U_new[i, j] = w_conserv * (U_new[i, j] + dt / dx * residuals[i, j]) + w_source * U[i, j]
                if abs(U_new[i, j]) > tol:
                    residual_max[j] = max(residual_max[j], abs(U_new[i, j]))
    return residual_max

def simulate_pipeline(U, F1, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D_H, Lp, theta_0, delta_x, T, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv):
    n = U.shape[0]
    time = 0
    delta_t_min = 1e-6
    residuals = np.zeros((n, 3))
    U_new = U.copy()
    
    # Armazenar valores de U ao longo do tempo
    time_values = []
    U1_values = []
    U2_values = []
    U3_values = []
    
    while time < T:
        print("Tempo atual:", time)
        # apply_boundary_conditions(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, nulv[0], nugv[0], alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv[-1], nugv[-1])
        for i in range(n - 1):
            theta = compute_catenary(i * delta_x, Z, Lp, theta_0)
            alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, P_i = compute_primitive(*U[i, :])
            alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1 = compute_primitive(*U[i + 1, :])

            alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star = compute_average_primitive_variables(alpha_i, alpha_ip1, rho_l_i, rho_l_ip1, rho_g_i, rho_g_ip1, u_l_i, u_l_ip1, u_g_i, u_g_ip1, P_i, P_ip1)
            U_i_star = compute_average_conservative_variables(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
            F_i_star = compute_average_flux_vector(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
            U_ip1_star = compute_average_conservative_variables(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)
            F_ip1_star = compute_average_flux_vector(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)

            Roe_matrix = compute_roe_matrix(U[i], U[i + 1], alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, nCl, nCg, theta, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, omega_P, tol)
            residuals[i, :] = -np.dot(Roe_matrix, (U[i + 1, :] - U[i, :])) + (F_ip1_star - F_i_star) / delta_x
 			
            delta_t = compute_time_step_lax_wendroff(CFL, delta_x, Roe_matrix)
            delta_t_min = min(delta_t_min, delta_t)
 			
            numerical_flux = roe_riemann_solver(F_i_star, F_ip1_star, U_i_star, U_ip1_star, Roe_matrix)
            rho_m_i, sin_theta_i, f_m_i, j_i, R_e_m_i = compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D_H, i, Z, eps, mu_g, mu_l, Lp, theta_0)
            source_term = compute_source_term_vector(U[i], alpha_i, rho_m_i, np.sin(theta), f_m_i, j_i)
            residuals[i, :] += source_term
 			
            U[i] = update_solution_at_interface(U[i], numerical_flux, source_term, delta_t_min, delta_x)
            print("Celula:",i)
            
        residual_max = calculate_residuals(n, delta_t, delta_x, residuals, w_conserv=1.0, w_source=1.0, U_new=U_new, U=U, tol=tol)
        
        U = U_new.copy()
		
		# Atualiza o tempo
        time += delta_t#_min
        
        # Armazenar valores
        time_values.append(time)
        U1_values.append(U[:, 0].copy())
        U2_values.append(U[:, 1].copy())
        U3_values.append(U[:, 2].copy())

        # Verifica se o tempo ultrapassou T e sai do loop principal, se necessário
        if time >= T:
            break
    
        # if np.all(residual_max < tol):
        #     break
    return U, time_values, U1_values, U2_values, U3_values

# def simulate_pipeline(U, F1, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D_H, Lp, theta_0, delta_x, T, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv):
#     n = U.shape[0]
#     time = 0
#     delta_t_min = 1e-6
#     tola = tol * 100
#     residuals = np.zeros((n, 3))
#     U_new = U.copy()
    
#     # Armazenar valores de U ao longo do tempo
#     time_values = []
#     U1_values = []
#     U2_values = []
#     U3_values = []
    
#     while time < T:
#         print("Tempo atual:", time)
        
#         # Aplicar condições de contorno
#         apply_boundary_conditions(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, nulv[0], nugv[0], alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv[-1], nugv[-1])
        
#         for i in range(n - 1):
#             theta = compute_catenary(i * delta_x, Z, Lp, theta_0)
#             alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, P_i = compute_primitive_variables(*U[i, :], theta, nCg, nCl, nrho_l0, nP_l0, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, tol, tola, n)
#             alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1 = compute_primitive_variables(*U[i + 1, :], theta, nCg, nCl, nrho_l0, nP_l0, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, tol, tola, n)

#             alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star = compute_average_primitive_variables(alpha_i, alpha_ip1, rho_l_i, rho_l_ip1, rho_g_i, rho_g_ip1, u_l_i, u_l_ip1, u_g_i, u_g_ip1, P_i, P_ip1)
#             U_i_star = compute_average_conservative_variables(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
#             F_i_star = compute_average_flux_vector(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
#             U_ip1_star = compute_average_conservative_variables(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)
#             F_ip1_star = compute_average_flux_vector(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)

#             theta = compute_catenary(i * delta_x, Z, Lp, theta_0)
#             Roe_matrix = compute_roe_matrix(U[i], U[i + 1], alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, nCl, nCg, theta, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, omega_P, tol)
#             residuals[i, :] = -np.dot(Roe_matrix, (U[i + 1, :] - U[i, :])) + (F_ip1_star - F_i_star) / delta_x
            
#             delta_t = compute_time_step_lax_wendroff(CFL, delta_x, Roe_matrix)
#             delta_t_min = min(delta_t_min, delta_t)
            
#             numerical_flux = roe_riemann_solver(F_i_star, F_ip1_star, U_i_star, U_ip1_star, Roe_matrix)
#             rho_m_i, sin_theta_i, f_m_i, j_i, R_e_m_i = compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D_H, i, Z, eps, mu_g, mu_l, Lp, theta_0)
#             source_term = compute_source_term_vector(U[i], alpha_i, rho_m_i, np.sin(theta), f_m_i, j_i)
#             residuals[i, :] += source_term
            
#             U_new[i] = update_solution_at_interface(U[i], numerical_flux, source_term, delta_t_min, delta_x)
#             print("Celula:", i)
        
#         residual_max = calculate_residuals(n, delta_t, delta_x, residuals, w_conserv=1.0, w_source=1.0, U_new=U_new, U=U, tol=tol)
        
#         U = U_new.copy()
        
#         # Atualiza o tempo
#         time += delta_t_min
        
#         # Armazenar valores
#         time_values.append(time)
#         U1_values.append(U[:, 0].copy())
#         U2_values.append(U[:, 1].copy())
#         U3_values.append(U[:, 2].copy())

#         # Verifica se o tempo ultrapassou T e sai do loop principal, se necessário
#         if time >= T:
#             break
        
#         # Verificação de convergência (descomente se necessário)
#         # if np.all(residual_max < tol):
#         #     break
    
#     return U, time_values, U1_values, U2_values, U3_values


# Plotar gráficos
def plot_results(time_values, U_values, label):
    plt.figure(figsize=(10, 6))
    for i in range(len(time_values)):
        plt.plot(U_values[i], label=f'Time {time_values[i]:.2f}')
    plt.xlabel('Spatial Position')
    plt.ylabel(label)
    plt.title(f'{label} over Time')
    plt.legend()
    plt.show()
    
# Parâmetros de entrada e condições iniciais
n = 1001                        # número de pontos da malha
X = 6.435                       # comprimento do tubo
Z = 9.886                       # altura do tubo
Lp = 10.0                       # comprimento da porção inclinada
theta_0 = 3.0                   # ângulo inicial em graus
Cg = 343.0                      # velocidade do som no gás
Cl = 1498.0                     # velocidade do som no líquido
P_l0 = 101325.0                 # pressão de referência
rho_l0 = 998.0                  # densidade do líquido de referência
rho_g0 = 1.2                    # densidade do gás de referência
Ps = 1.5*101325.0                 # pressão no tubo
mu_g = 1.81e-5                  # viscosidade do gás
mu_l = 1e-3                     # viscosidade do líquido
eps = 4.6e-5                    # rugosidade
D_H = 0.1016                    # diâmetro hidráulico
T = 2.0                          # tempo total de simulação
CFL = 0.6                       # número de Courant-Friedrichs-Lewy
tol = 1e-15
tola = tol * 100
AREA = math.pi * (D_H**2)/4.0
sigma = 7.28 * 10 ** (-2)
G = 9.81

# Parâmetro da catenária
CA, Lr = calcular_catenaria(X, Z, tol)
delta_x = ((Lp + Lr)/ Lr) / (n - 1)
S = Lp + X
U = np.zeros((n, 3))
F1 = np.zeros((n, 3))

# Adimensionalização
nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho = adimensionalizar(Cg, Cl, P_l0, rho_l0, Ps)
    
# Vazão de massa de líquido e gás na pressão de referência
Pb = Ps + rho_l0 * G * Z
rhog0 = Pb / (Cg ** 2)
rhol0 = rho_l0 + (Pb - P_l0) / (Cl ** 2)

# Adimensionalização dos valores de referência
nrhog0 = rhog0 / omega_rho
nrhol0 = rhol0 / omega_rho
    
# Geração dos vetores de velocidade superficial
vjl, vjg = gerar_vetores_velocidade_superficial(n)
jl = 0.01
jg = 0.001
omega_u = np.maximum(jl, jg)
nmul = rhol0 * jl * AREA
nmug = rhog0 * jg * AREA 
vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav = steadyState.EstadoEstacionario_ndim_simp(n, nmul, nmug, Ps, Lp, Lr, CA, np.radians(theta_0), D_H, AREA, eps, G, Cl, Cg, rho_l0, P_l0, mu_l, mu_g, sigma, omega_P, omega_u, omega_rho, tol)

dot_M_l0 = nmul 
dot_M_g0 = nmug 
alpha_start, rho_l_start, rho_g_start, alpha_end, rho_l_end, rho_g_end, P_inlet, P_outlet = boundaryConditions.calculate_boundary_conditions(U[0, 0], U[0, 1], U[0, 2], nCl, nCg, nrho_l0, nP_l0, dot_M_l0, dot_M_g0, AREA, Lp, Lr)
alpha_start_dim = alpha_start
rho_l_start_dim = rho_l_start / omega_rho
rho_g_start_dim = rho_g_start / omega_rho
alpha_end_dim = alpha_end
rho_l_end_dim = rho_l_end / omega_rho
rho_g_end_dim = rho_g_end / omega_rho

for i in range(n):
    # Condições iniciais de U
    u1, u2, u3 = compute_conservative_variables(nPs, nCl, nCg, nrho_l0, nP_l0, alphav[i], nrholv[i], nulv[i], nrhogv[i], nugv[i])
    U[i, 0] = u1
    U[i, 1] = u2
    U[i, 2] = u3
    # Condições iniciais de F1
    alpha, rho_l, rho_g, u_l, u_g, P = compute_primitive_variables(*U[i, :], theta_0, nCg, nCl, nrho_l0, nP_l0, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, tol, tola, n)
    print("alpha:",alpha)
    # alpha, rho_l, rho_g, u_l, u_g, P = compute_primitive(*U[i, :])
    F1[i, :] = compute_F1(alpha, rho_l, rho_g, u_l, u_g)
    
# Simulação
U_final, time_values, U1_values, U2_values, U3_values = simulate_pipeline(U, F1, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D_H, Lp, theta_0, delta_x, T, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv)

# Plotando os resultados
plt.figure(1)
plt.plot(np.linspace(0, S, n), U_final[:, 0])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U1')
plt.title('Distribuição de U1 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(2)
plt.plot(np.linspace(0, S, n), U_final[:, 1])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U2')
plt.title('Distribuição de U2 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(3)
plt.plot(np.linspace(0, S, n), U_final[:, 2])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U3')
plt.title('Distribuição de U3 ao longo do tubo')
plt.grid(True)
plt.show()

# plot_results(time_values, U1_values, 'U1')
# plot_results(time_values, U2_values, 'U2')
# plot_results(time_values, U3_values, 'U3')