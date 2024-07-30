import numpy as np
import math
import catenary
import find
import matrixRoe
import frictionFactor
import steadyState
import boundaryConditions
import matplotlib.pyplot as plt

# def apply_boundary_conditions(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, nulv_start, nugv_start, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv_end, nugv_end):
#     # Condição de contorno na entrada
#     u1_start, u2_start, u3_start = compute_conservative_variables(nPs, nCl, nCg, nrho_l0, nP_l0, alpha_start_dim, rho_l_start_dim, nulv_start, rho_g_start_dim, nugv_start)
#     U[0, :] = [u1_start, u2_start, u3_start]

#     # Condição de contorno na saída
#     u1_end, u2_end, u3_end = compute_conservative_variables(nPs, nCl, nCg, nrho_l0, nP_l0, alpha_end_dim, rho_l_end_dim, nulv_end, rho_g_end_dim, nugv_end)
#     U[-1, :] = [u1_end, u2_end, u3_end]

def apply_boundary_conditions(U, alpha_in, P_in, rho_l_in, u_l_in, u_g_in, alpha_out, P_separator, A, nCg, nCl, rho_l0, P_l0, tol):
    """
    Apply boundary conditions at the pipeline entrance and riser top.
    
    Parameters:
        U (numpy.ndarray): Array of conservative variables.
        m_dot_l0 (float): Mass flow rate of the liquid at the pipeline entrance.
        m_dot_g0 (float): Mass flow rate of the gas at the pipeline entrance.
        P_separator (float): Pressure at the separator (riser top).
        A (float): Cross-sectional area of the pipeline.
        nCg (float): Dimensionless sound speed in gas.
        nCl (float): Dimensionless sound speed in liquid.
        rho_l0 (float): Reference liquid density.
        P_l0 (float): Reference pressure.
        tol (float): Tolerance for convergence.
    """
    # Boundary condition at the entrance
    # rho_l_in = m_dot_l0 / A
    # u_l_in = m_dot_l0 / (rho_l_in * A)
    # u_g_in = m_dot_g0 / (P_separator / (nCg ** 2) * A)

    # alpha_in = find.alpha_fun_mass_flow(m_dot_l0, m_dot_g0, u_l_in, u_g_in, nCl, nCg, rho_l0, P_l0, tol)
    # P_in = find.pressure_fun_mass_flow(m_dot_l0, m_dot_g0, u_l_in, u_g_in, nCl, nCg, rho_l0, P_l0)

    U[0, 0] = (1 - alpha_in) * (rho_l0 + (P_in - P_l0) / nCl)
    U[0, 1] = alpha_in * P_in / (nCg ** 2)
    U[0, 2] = (1 - alpha_in) * rho_l_in * u_l_in + alpha_in * (P_separator / (nCg ** 2)) * u_g_in

    # Boundary condition at the riser top
    # alpha_out = find.alpha_fun_mass_flow(m_dot_l0, m_dot_g0, u_l_in, u_g_in, nCl, nCg, rho_l0, P_l0, tol)
    # P_out = P_separator

    U[-1, 0] = (1 - alpha_out) * (rho_l0 + (P_separator - P_l0) / nCl)
    U[-1, 1] = alpha_out * P_separator / (nCg ** 2)
    U[-1, 2] = (1 - alpha_out) * rho_l_in * u_l_in + alpha_out * (P_separator / (nCg ** 2)) * u_g_in

def compute_conservative_variables(P, Cl, Cg, rho_l_0, Pl_0, alpha, rho_l, ul, rho_g, ug):
    u1 = (1 - alpha) * (rho_l_0 + (P - Pl_0) / Cl)
    u2 = alpha * P / (Cg**2)
    u3 = (1 - alpha) * rho_l * ul + alpha * rho_g * ug
    return u1, u2, u3

def compute_primitive_variables(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):

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

def compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D, s_i, z, edia, mu_g, mu_l, Lp, theta):
    mu_m_i = mu_g * alpha_i + (1 - alpha_i) * mu_l
    rho_m_i = (1 - alpha_i) * rho_l_i + alpha_i * rho_g_i
    sin_theta_i = np.sin(theta)
    j_i = (1 - alpha_i) * u_l_i + alpha_i * u_g_i
    
    R_e_m_i = (rho_m_i * j_i * D) / mu_m_i if j_i != 0 else 0
    f_m_i = frictionFactor.FatorAtrito(R_e_m_i, edia)
    return rho_m_i, sin_theta_i, f_m_i, j_i, R_e_m_i

def compute_source_term_vector(U_i, alpha_i, rho_m_i, sin_theta_i, f_m_i, j_i):
    S1_i = S2_i = 0
    S3_i = -rho_m_i * sin_theta_i - 0.5 * rho_m_i * f_m_i * np.abs(j_i) * j_i
    return np.array([S1_i, S2_i, S3_i])
    
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

def adimensionalizar(Cg, Cl, P_l0, rho_l0, Ps, jl, jg, mul, mug):
    omega_P = P_l0
    omega_rho = rho_l0
    omega_c = 1 / np.sqrt(omega_rho / omega_P)
    omega_u = np.maximum(jl, jg)
    nCg = Cg / omega_c
    nCl = Cl / omega_c
    nP_l0 = P_l0 / omega_P
    nrho_l0 = rho_l0 / omega_rho
    nPs = Ps / omega_P
    njl = jl/omega_u
    njg = jg/omega_u
    nmul = mul/(rho_l0*omega_u)
    nmug = mug/(rho_l0*omega_u)
    return nCg, nCl, nP_l0, nrho_l0, nPs, njl, njg, nmul, nmug, omega_c, omega_P, omega_rho, omega_u

def calculate_residuals(U, U_new):
    residuals = U_new - U
    residual_squared = np.sum(residuals**2, axis=1)
    residual_standard_error = np.sqrt(np.sum(residual_squared) / U.shape[0])
    return residual_standard_error


def simulate_pipeline(U, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D_H, Lp, theta_0, delta_x, T, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv, njl, njg):
    # n = U.shape[0]
    time = 0
    delta_t_min = 1e-2
    residuals = np.zeros((n, 3))
    U_new = U.copy()
    # Armazenar valores ao longo do tempo
    delta_time_values = []
    numerical_flux_values = []
    source_term_values = []
    time_values = []
    U_values = []
    while time < T:
        # apply_boundary_conditions(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, nulv[0], nugv[0], alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv[-1], nugv[-1])
        for i in range(n - 1):
            theta = catenary.compute_catenary(i * delta_x, Z, Lp, theta_0)
            alpha_i, rho_l_i, rho_g_i, P_i, u_l_i, u_g_i, index = compute_primitive_variables(U[i,0], U[i,1], U[i,2], theta, nCg, nCl, nrho_l0, nP_l0, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, tol, tola, n)
            alpha_ip1, rho_l_ip1, rho_g_ip1, P_ip1, u_l_ip1, u_g_ip1, index = compute_primitive_variables(U[i+1,0], U[i+1,1], U[i+1,2], theta, nCg, nCl, nrho_l0, nP_l0, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, tol, tola, n)
            
            alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star = compute_average_primitive_variables(alpha_i, alpha_ip1, rho_l_i, rho_l_ip1, rho_g_i, rho_g_ip1, u_l_i, u_l_ip1, u_g_i, u_g_ip1, P_i, P_ip1)
            U_i_star = compute_average_conservative_variables(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
            F_i_star = compute_average_flux_vector(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, P_i_star)
            U_ip1_star = compute_average_conservative_variables(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)
            F_ip1_star = compute_average_flux_vector(alpha_ip1, rho_l_ip1, rho_g_ip1, u_l_ip1, u_g_ip1, P_ip1)

            Roe_matrix = matrixRoe.calculate_roe_matrix(alpha_i_star, rho_l_i_star, rho_g_i_star, u_l_i_star, u_g_i_star, nCl, nCg, theta, D_H, AREA, eps, G, mu_l, mu_g, sigma, omega_u, omega_rho, omega_P, tol)
            residuals[i, :] = -np.dot(Roe_matrix, (U[i + 1, :] - U[i, :])) + (F_ip1_star - F_i_star) / delta_x
            
            delta_t = compute_time_step_lax_wendroff(CFL, delta_x, Roe_matrix)
            delta_time_values.append(delta_t)
            
            numerical_flux = roe_riemann_solver(F_i_star, F_ip1_star, U_i_star, U_ip1_star, Roe_matrix)
            numerical_flux_values.append(numerical_flux)
            rho_m_i, sin_theta_i, f_m_i, j_i, R_e_m_i = compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D_H, i, Z, eps, mu_g, mu_l, Lp, theta)
            source_term = compute_source_term_vector(U[i], alpha_i, rho_m_i, sin_theta_i, f_m_i, j_i)
            source_term_values.append(source_term)
            residuals[i, :] += source_term
        
        apply_boundary_conditions(U, alpha_start_dim, P_i, rho_l_start_dim, njl, njg, alpha_end_dim, nPs, AREA, nCg, nCl, nrho_l0, nP_l0, tol)
        delta_t_min = min(delta_time_values)
        for i in range(n - 1):
            U[i] = update_solution_at_interface(U[i], numerical_flux_values[i], source_term_values[i], delta_t_min, delta_x)
            #U_new[i] = update_solution_at_interface(U[i], numerical_flux, source_term, delta_t_min, delta_x)
        print(f"Tempo atual:{time}")
        
        # residual_error = calculate_residuals(U, U_new)
        # print(f"Tempo atual:{time} :: Erro residual máximo:{residual_error}")
        # if np.all(residual_error < tol):
        #     print("Condição de parada atingida: erro residual menor que a tolerância.")
        #     break
        #U = U_new.copy()
        time += delta_t_min
        time_values.append(time)
        U_values.append(U.copy())

        if time >= T:
            break
        # Condição de parada baseada em erro residual (comentada)
        
    
    return U, time_values, U_values


# Plotar gráficos
def plot_results(time_values, U_values, label):
    plt.figure(figsize=(10, 6))
    for i in range(len(time_values)):
        plt.plot(U_values[i], label=f"Time {time_values[i]:.2f}")
    plt.xlabel("Spatial Position")
    plt.ylabel(label)
    plt.title(f"{label} over Time")
    plt.legend()
    plt.show()
    
# Parâmetros de entrada e condições iniciais
n = 101                         # número de pontos da malha
X = 6.435                       # comprimento do tubo
Z = 9.886                       # altura do tubo
Lp = 10.0                       # comprimento da porção inclinada
theta_0 = 5.0                   # ângulo inicial em graus
Cg = 343.0                      # velocidade do som no gás
Cl = 1498.0                     # velocidade do som no líquido
P_l0 = 101325.0                 # pressão de referência
rho_l0 = 998.0                  # densidade do líquido de referência
rho_g0 = 1.2                    # densidade do gás de referência
Ps = 1.5*101325.0               # pressão no tubo
mu_g = 1.81e-5                  # viscosidade do gás
mu_l = 1e-3                     # viscosidade do líquido
eps = 4.6e-5                    # rugosidade
D = 0.0254                      # diâmetro
time = 0.002                    # tempo total de simulação
CFL = 0.9                       # número de Courant-Friedrichs-Lewy
tol = 1e-15
tola = tol * 100
AREA = math.pi * (D**2)/4.0
sigma = 7.28e-2
G = 9.81
jl = 6.0
jg = 1.0


# Parâmetro da catenária
CA = catenary.catenary_constant(X, Z, tol)
Lr = CA * np.sinh(X / CA)
delta_x = ((Lp + Lr)/ Lr) / (n - 1)
S = Lp + Lr
nS = ((Lp + Lr) / Lr)
U = np.zeros((n, 3))
F = np.zeros((n, 3))

# Vazão de massa de líquido e gás na pressão de referência
Pb = Ps + rho_l0 * G * Z
rhog0 = Pb / (Cg ** 2)
rhol0 = rho_l0 + (Pb - P_l0) / (Cl ** 2)
mul = rhol0 * jl * AREA
mug = rhog0 * jg * AREA

# Adimensionalização
nCg, nCl, nP_l0, nrho_l0, nPs, njl, njg, nmul, nmug, omega_c, omega_P, omega_rho, omega_u = adimensionalizar(Cg, Cl, P_l0, rho_l0, Ps, jl, jg, mul, mug)
nrhog0 = rhog0 / omega_rho
nrhol0 = rhol0 / omega_rho
ntime = time*(jl/Lr)

# Calculo do estado estacionário
vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav = steadyState.EstadoEstacionario_ndim_simp(n, nmul, nmug, Ps, Lp, Lr, CA, np.radians(theta_0), D, AREA, eps, G, Cl, Cg, rho_l0, P_l0, mu_l, mu_g, sigma, omega_P, omega_u, omega_rho, tol)
for i in range(n):
    # Condições iniciais de U
    u1, u2, u3 = compute_conservative_variables(vnp[i], nCl, nCg, nrho_l0, nP_l0, alphav[i], nrholv[i], nulv[i], nrhogv[i], nugv[i])
    U[i, 0] = u1
    U[i, 1] = u2
    U[i, 2] = u3
#alpha_start, rho_l_start, rho_g_start, alpha_end, rho_l_end, rho_g_end, P_inlet, P_outlet = boundaryConditions.calculate_boundary_conditions(U[0, :], U[-1, :], nCl, nCg, nrho_l0, nP_l0, nmul, nmug, AREA, Lp, Lr)

alpha_start_dim = alphav[0]
rho_l_start_dim = nrholv[0] / omega_rho
rho_g_start_dim = nrhogv[0] / omega_rho
alpha_end_dim = alphav[-1]
rho_l_end_dim = nrholv[-1] / omega_rho
rho_g_end_dim = nrhogv[-1] / omega_rho 
# Simulação
U_final, time_values, U_values = simulate_pipeline(U, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D, Lp, theta_0, delta_x, ntime, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv, njl, njg)

# Plotando os resultados
plt.figure(1)
plt.plot(np.linspace(0, nS, n), U_final[:, 0])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U1')
plt.title('Distribuição de U1 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(2)
plt.plot(np.linspace(0, nS, n), U_final[:, 1])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U2')
plt.title('Distribuição de U2 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(3)
plt.plot(np.linspace(0, nS, n), U_final[:, 2])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U3')
plt.title('Distribuição de U3 ao longo do tubo')
plt.grid(True)
plt.show()

# plot_results(time_values, U1_values, 'U1')
# plot_results(time_values, U2_values, 'U2')
# plot_results(time_values, U3_values, 'U3')