import math

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

def calculate_primitive_variables(u1, u2, u3, Cl, Cg, rho_l0, P_l0):
    # Calculando alpha
    alpha0 = (u1 - rho_l0 + P_l0 / (Cl ** 2) + u2 * (Cg / Cl) ** 2) / (2 * (P_l0 / (Cl ** 2 - rho_l0)))
    alpha1 = math.sqrt((u1 - rho_l0 + P_l0 / (Cl ** 2) + u2 * (Cg / Cl) ** 2) ** 2 - 4 * u2 * (Cg / Cl) ** 2 * (P_l0 / (Cl ** 2) - rho_l0)) / (2 * (P_l0 / (Cl ** 2 - rho_l0)))
    alpha = min(alpha0, alpha1)
    
    # Calculando P
    P = -Cl ** 2 / 2 * ((rho_l0 - P_l0 / (Cl ** 2)) - u2 * (Cg / Cl) ** 2 - u1)
    
    return alpha, P


# def calculate_boundary_conditions(u1, u2, u3, Cl, Cg, rho_l0, P_l0, dot_M_l0, dot_M_g0, A, L_p, L_r):
#     # Calculate primitive variables at the inlet and outlet
#     alpha_inlet, P_inlet = calculate_primitive_variables(u1, u2, u3, Cl, Cg, rho_l0, P_l0)
#     # jl_inlet = dot_M_l0 / (A * u1)
#     # jg_inlet = dot_M_g0 / (A * u2)
#     alpha_outlet, P_outlet = alpha_inlet, P_l0  # Assuming the outlet pressure is P_l0
    
#     # Calculate conservative variables in the real cells
#     alpha_start, rho_l_start, rho_g_start = alpha_inlet, (u1 - alpha_inlet) * (1 / (1 - alpha_inlet)), u2 * (1 / alpha_inlet)
#     alpha_end, rho_l_end, rho_g_end = alpha_outlet, (u1 - alpha_outlet) * (1 / (1 - alpha_outlet)), u2 * (1 / alpha_outlet)
    
#     return alpha_start, rho_l_start, rho_g_start, alpha_end, rho_l_end, rho_g_end, P_inlet, P_outlet#, jl_inlet, jg_inlet


# def calculate_boundary_conditions(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, AREA, dot_M_l0, dot_M_g0, L_p, L_r, D, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):
#     # Calculating primitive variables at real boundary points
#     alpha0, rhol0, rhog0, P0, ul0, ug0, index0 = find.find_primvar_from_cvar_simp(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit)
#     alphaL, rholL, rhogL, PL, ulL, ugL, indexL = find.find_primvar_from_cvar_simp(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit)
    
#     # Calculate primitive variables at fictitious boundary points using linear extrapolation
#     alpha_dummy_start = 2 * alpha0 - alphaL
#     P_dummy_start = 2 * P0 - PL
#     alpha_dummy_end = 2 * alphaL - alpha0
#     P_dummy_end = 2 * PL - P0
    
#     # Calculate conservative variables at real boundary points
#     rho_l_start = (u1 - alpha0) * (1 / (1 - alpha0))
#     rho_g_start = u2 * (1 / alpha0)
#     rho_l_end = (u1 - alphaL) * (1 / (1 - alphaL))
#     rho_g_end = u2 * (1 / alphaL)
    
#     # Calculate conservative variables at fictitious boundary points using linear extrapolation
#     rho_l_dummy_start = 2 * rho_l_start - rho_l_end
#     rho_g_dummy_start = 2 * rho_g_start - rho_g_end
#     rho_l_dummy_end = 2 * rho_l_end - rho_l_start
#     rho_g_dummy_end = 2 * rho_g_end - rho_g_start
    
#     # Return boundary conditions at real and fictitious boundary points
#     return (alpha_dummy_start, rho_l_dummy_start, rho_g_dummy_start, alpha0, rho_l_start, rho_g_start, alpha_dummy_end, rho_l_dummy_end, rho_g_dummy_end), (P_dummy_start, P0, P_dummy_end)

# def calculate_primitive_variables_bc(u1, u2, u3, dot_M_l0, dot_M_g0, A, c_l, c_g, P_l0, rho_l_top, rho_g_top):
#     # Calculate alpha and P from u1, u2, u3
#     alpha0 = (u1 - rho_l_top + P_l0 / c_l**2 + u2 * (c_g / c_l)**2) / (2 * (P_l0 / (c_l**2 - rho_l_top)))
#     alpha1 = ((u1 - rho_l_top + P_l0 / c_l**2 + u2 * (c_g / c_l)**2)**2 - 4 * u2 * (c_g / c_l)**2 * (P_l0 / c_l**2 - rho_l_top))**0.5 / (2 * (P_l0 / (c_l**2 - rho_l_top)))
#     alpha = min(alpha0, alpha1)
#     P = (-c_l**2 / 2) * ((rho_l_top - P_l0 / c_l**2) - u2 * (c_g / c_l)**2 - u1 + alpha)
    
#     # Calculate u_l and u_g at the inlet
#     u_l_inlet = dot_M_l0 / (A * u1)
#     u_g_inlet = dot_M_g0 / (A * u2)
    
#     # Calculate u_l and u_g at the outlet (top of the riser)
#     u_l_outlet = dot_M_l0 / (A * u1)
#     u_g_outlet = dot_M_g0 / (A * u2)
    
#     # Calculate conservative variables in dummy cells using linear extrapolation
#     alpha_dummy_start = 2 * alpha_inlet - alpha_outlet
#     rho_l_dummy_start = 2 * rho_l_top - (u1 - alpha_inlet) * (1 / (1 - alpha_inlet))
#     rho_g_dummy_start = 2 * rho_g_top - u2 * (1 / alpha_inlet)
#     alpha_dummy_end = 2 * alpha_outlet - alpha_inlet
#     rho_l_dummy_end = 2 * rho_l_top - (u1 - alpha_outlet) * (1 / (1 - alpha_outlet))
#     rho_g_dummy_end = 2 * rho_g_top - u2 * (1 / alpha_outlet)
    
#     return (alpha_dummy_start, rho_l_dummy_start, rho_g_dummy_start, alpha_inlet, rho_l_top, rho_g_top, alpha_dummy_end, rho_l_dummy_end, rho_g_dummy_end), (P_inlet, P_l0, P_outlet)

# def calculate_boundary_conditions_conservative(u1, u2, u3, Cl, Cg, rho_l0, rho_g0, P_l0, dot_M_l0, dot_M_g0, A, L_p, L_r):
#     # Calculate primitive variables at inlet and outlet
#     alpha_inlet, P_inlet = calculate_primitive_variables(u1, u2, u3, Cl, Cg, rho_l0, P_l0)
#     u_l_inlet = dot_M_l0 / (A * u1)
#     u_g_inlet = dot_M_g0 / (A * u2)
#     alpha_outlet, P_outlet = alpha_inlet, P_l0  # Assuming the outlet pressure is P_l0
    
#     # Calculate conservative variables in real cells
#     alpha_real_start, rho_l_real_start, rho_g_real_start = alpha_inlet, rho_l0, rho_g0
#     alpha_real_end, rho_l_real_end, rho_g_real_end = alpha_outlet, rho_l0, rho_g0
    
#     # Calculate conservative variables in dummy cells using linear extrapolation
#     alpha_dummy_start = 2 * alpha_inlet - alpha_outlet
#     rho_l_dummy_start = 2 * rho_l0 - (u1 - alpha_inlet) * (1 / (1 - alpha_inlet))
#     rho_g_dummy_start = 2 * rho_g0 - u2 * (1 / alpha_inlet)
#     alpha_dummy_end = 2 * alpha_outlet - alpha_inlet
#     rho_l_dummy_end = 2 * rho_l0 - (u1 - alpha_outlet) * (1 / (1 - alpha_outlet))
#     rho_g_dummy_end = 2 * rho_g0 - u2 * (1 / alpha_outlet)
    
#     return (alpha_real_start, rho_l_real_start, rho_g_real_start, alpha_real_end, rho_l_real_end, rho_g_real_end), (P_inlet, P_l0, P_outlet)


# Exemplo de uso
# u1 = 0.5
# u2 = 0.3
# u3 = 0.7
# rho_l0 = 1000
# c_l = 1500
# c_g = 500
# P_l0 = 1e5
# A = 1
# dot_M_l0 = 100
# dot_M_g0 = 50
# L_p = 10
# L_r = 5

# boundary_start, boundary_end = calculate_boundary_conditions(u1, u2, u3, rho_l0, c_l, c_g, P_l0, A, dot_M_l0, dot_M_g0, L_p, L_r)
# print("Condições de contorno no início do pipeline (dummy cells):", boundary_start)
# print("Condições de contorno no topo do riser (dummy cells):", boundary_end)