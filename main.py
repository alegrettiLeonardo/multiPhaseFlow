import numpy as np
import math
import catenary
import find
import matrixRoe
import frictionFactor
import steadyState
import boundaryConditions
import matplotlib.pyplot as plt

# def apply_boundary_conditions(U, alpha_in, rho_l_in, rho_g_in, u_l_in, u_g_in, rho_l_out, rho_g_out, u_l_out, u_g_out, alpha_out):
#     """
#     Apply boundary conditions at the pipeline entrance and riser top.
    
#     Parameters:
#         U (numpy.ndarray): Array of conservative variables.
#         m_dot_l0 (float): Mass flow rate of the liquid at the pipeline entrance.
#         m_dot_g0 (float): Mass flow rate of the gas at the pipeline entrance.
#         P_separator (float): Pressure at the separator (riser top).
#         A (float): Cross-sectional area of the pipeline.
#         nCg (float): Dimensionless sound speed in gas.
#         nCl (float): Dimensionless sound speed in liquid.
#         rho_l0 (float): Reference liquid density.
#         P_l0 (float): Reference pressure.
#         tol (float): Tolerance for convergence.
#     """
    
#     # inlet
#     U[0, 0] = (1 - alpha_in) * rho_l_in
#     U[0, 1] = alpha_in * rho_g_in
#     U[0, 2] = (1 - alpha_in) * rho_l_in * u_l_in + alpha_in * rho_g_in * u_g_in
    
#     # outlet
#     U[-1, 0] = (1 - alpha_out) * rho_l_out
#     U[-1, 1] = alpha_out * rho_g_out
#     U[-1, 2] = (1 - alpha_out) * rho_l_out * u_l_out + alpha_out * rho_g_out * u_g_out

def apply_boundary_conditions_with_dummy_cells(U, alpha_in, rho_l_in, rho_g_in, u_l_in, u_g_in, rho_l_out, rho_g_out, u_l_out, u_g_out, alpha_out, num_dummy_cells):
    """
    Apply boundary conditions at the pipeline entrance and riser top with dummy cells.
    
    Parameters:
        U (numpy.ndarray): Array of conservative variables including dummy cells.
        alpha_in, alpha_out (float): Void fractions at inlet and outlet.
        rho_l_in, rho_g_in, u_l_in, u_g_in (float): Properties at inlet.
        rho_l_out, rho_g_out, u_l_out, u_g_out (float): Properties at outlet.
    """
    
    # inlet dummy cell (extrapolate from first interior cell)
    U[num_dummy_cells-1, 0] = (1 - alpha_in) * rho_l_in
    U[num_dummy_cells-1, 1] = alpha_in * rho_g_in
    U[num_dummy_cells-1, 2] = (1 - alpha_in) * rho_l_in * u_l_in + alpha_in * rho_g_in * u_g_in
    
    # Apply the actual inlet boundary conditions (in first physical cell)
    U[num_dummy_cells, 0] = (1 - alpha_in) * rho_l_in
    U[num_dummy_cells, 1] = alpha_in * rho_g_in
    U[num_dummy_cells, 2] = (1 - alpha_in) * rho_l_in * u_l_in + alpha_in * rho_g_in * u_g_in
    
    # Apply the actual outlet boundary conditions (in last physical cell)
    U[-(num_dummy_cells+1), 0] = (1 - alpha_out) * rho_l_out
    U[-(num_dummy_cells+1), 1] = alpha_out * rho_g_out
    U[-(num_dummy_cells+1), 2] = (1 - alpha_out) * rho_l_out * u_l_out + alpha_out * rho_g_out * u_g_out
    
    # outlet dummy cell (extrapolate from last interior cell)
    U[-num_dummy_cells, 0] = (1 - alpha_out) * rho_l_out
    U[-num_dummy_cells, 1] = alpha_out * rho_g_out
    U[-num_dummy_cells, 2] = (1 - alpha_out) * rho_l_out * u_l_out + alpha_out * rho_g_out * u_g_out

def compute_conservative_variables(P, Cl, Cg, rho_l_0, Pl_0, alpha, rho_l, ul, rho_g, ug):
    u1 = (1 - alpha) * (rho_l_0 + (P - Pl_0) / Cl)
    u2 = alpha * P / (Cg**2)
    u3 = (1 - alpha) * rho_l * ul + alpha * rho_g * ug
    return float(u1), float(u2), float(u3)

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

    return float(alpha), float(rhol), float(rhog), float(P), float(ul), float(ug), index

def compute_average_primitive_variables(alpha_i, alpha_ip1, rho_l_i, rho_l_ip1, rho_g_i, rho_g_ip1, u_l_i, u_l_ip1, u_g_i, u_g_ip1, P_i, P_ip1):
    alpha_i_star = 0.5 * (alpha_i + alpha_ip1)
    rho_l_i_star = 0.5 * (rho_l_i + rho_l_ip1)
    rho_g_i_star = 0.5 * (rho_g_i + rho_g_ip1)
    u_l_i_star = 0.5 * (u_l_i + u_l_ip1)
    u_g_i_star = 0.5 * (u_g_i + u_g_ip1)
    P_i_star = 0.5 * (P_i + P_ip1)
    return float(alpha_i_star), float(rho_l_i_star), float(rho_g_i_star), float(u_l_i_star), float(u_g_i_star), float(P_i_star)

def compute_mixture_parameters(alpha_i, rho_l_i, rho_g_i, u_l_i, u_g_i, D, s_i, z, edia, mu_g, mu_l, Lp, theta):
    mu_m_i = mu_g * alpha_i + (1 - alpha_i) * mu_l
    rho_m_i = (1 - alpha_i) * rho_l_i + alpha_i * rho_g_i
    sin_theta_i = np.sin(theta)
    j_i = (1 - alpha_i) * u_l_i + alpha_i * u_g_i
    
    R_e_m_i = (rho_m_i * j_i * D) / mu_m_i if j_i != 0 else 0
    f_m_i = frictionFactor.FatorAtrito(R_e_m_i, edia)
    return float(rho_m_i), float(sin_theta_i), float(f_m_i), float(j_i), float(R_e_m_i)

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
    
    return float(CFL * delta_x / (2 * max_lambda))

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
    return float(nCg), float(nCl), float(nP_l0), float(nrho_l0), float(nPs), float(njl), float(njg), float(nmul), float(nmug), float(omega_c), float(omega_P), float(omega_rho), float(omega_u)

def calculate_residuals(U, U_new):
    residuals = U_new - U
    residual_squared = np.sum(residuals**2, axis=1)
    residual_standard_error = np.sqrt(np.sum(residual_squared) / U.shape[0])
    return residual_standard_error


def simulate_pipeline(U, tol, n, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D_H, Lp, theta_0, delta_x, T, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv, njl, njg, p_start_dim, ul_start_dim, ug_start_dim, ul_end_dim, ug_end_dim, num_dummy_cells):
    # n = U.shape[0]
    time = 0
    delta_t_min = 1e-10
    residuals = np.zeros((n, 3))
    U_new = U.copy()
    # Armazenar valores ao longo do tempo
    delta_time_values = []
    numerical_flux_values = []
    source_term_values = []
    time_values = []
    U_values = []
    CA = catenary.catenary_constant(X, Z, tol)
    while time < T:
        apply_boundary_conditions_with_dummy_cells(U, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, ul_start_dim, ug_start_dim, rho_l_end_dim, rho_g_end_dim, ul_end_dim, ug_end_dim, alpha_end_dim, num_dummy_cells)
        
        # Modificar os laços for para levar em conta as dummy cells
        for i in range(num_dummy_cells, n - num_dummy_cells - 1):
            theta = catenary.fun_or_geo(i * delta_x, Lp, theta_0, CA)
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
        
        delta_t_min = min(delta_time_values)
        
        # Atualizar solução ignorando as dummy cells
        for i in range(num_dummy_cells, n - num_dummy_cells - 1):
            U[i] = update_solution_at_interface(U[i], numerical_flux_values[i - num_dummy_cells], source_term_values[i - num_dummy_cells], delta_t_min, delta_x)
        
        print(f"Tempo atual:{time}")
        
        time += delta_t_min
        time_values.append(time)
        U_values.append(U.copy())

        if time >= T:
            break
    
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
num_dummy_cells = 1             # Number of dummy cells
X = 6.435                       # comprimento do riser
Z = 9.886                       # altura do tubo
Lp = 5.0                        # comprimento do oleoduto
beta = -5.0                      # ângulo inicial em graus
Cg = 343.0                      # velocidade do som no gás
Cl = 1498.0                     # velocidade do som no líquido
P_l0 = 101325.0                 # pressão de referência
rho_l0 = 998.0                  # densidade do líquido de referência
rho_g0 = 1.2                    # densidade do gás de referência
Ps = 1.5*P_l0                   # pressão no tubo
mu_g = 1.81e-5                  # viscosidade do gás
mu_l = 1.0e-3                   # viscosidade do líquido
eps = 4.6e-5                    # rugosidade
D = 0.0254                      # diâmetro
time = 0.01                     # tempo total de simulação
CFL = 0.9                       # número de Courant-Friedrichs-Lewy
tol = 1e-15
tola = tol * 100
AREA = math.pi * (D**2)/4.0
sigma = 7.28e-2
G = 9.81
jl = 3.0e-2
jg = 1.0e-2
N = n + num_dummy_cells * 2

# Parâmetro da catenária
CA = catenary.catenary_constant(X, Z, tol)
Lr = CA * np.sinh(X / CA)
delta_x = ((Lp + Lr)/ Lr) / (N - 1)
S = Lp + Lr
nS = ((Lp + Lr) / Lr)
U = np.zeros((N, 3))
F = np.zeros((N, 3))

# Vazão de massa de líquido e gás na pressão de referência
Pb = Ps + rho_l0 * G * Z
rhog0 = Pb / (Cg ** 2)
rhol0 = rho_l0 + (Pb - P_l0) / (Cl ** 2)
mul = rhol0 * jl 
mug = rhog0 * jg 

# Adimensionalização
nCg, nCl, nP_l0, nrho_l0, nPs, njl, njg, nmul, nmug, omega_c, omega_P, omega_rho, omega_u = adimensionalizar(Cg, Cl, P_l0, rho_l0, Ps, jl, jg, mul, mug)
nrhog0 = rhog0 / omega_rho                                                                  
nrhol0 = rhol0 / omega_rho
ntime = time*(jl/Lr)

# Calculo do estado estacionário
vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav = steadyState.EstadoEstacionario_ndim_simp(
   N, nmul, nmug, nPs, Lp, Lr, CA, np.radians(beta), D, AREA, eps, G, nCl, nCg, nrho_l0, nP_l0, mu_l, mu_g, sigma, omega_P, omega_u, omega_rho, tol
)
for i in range(N):
    # Condições iniciais de U
    u1, u2, u3 = compute_conservative_variables(vnp[i], nCl, nCg, nrho_l0, nP_l0, alphav[i], nrholv[i], nulv[i], nrhogv[i], nugv[i])
    U[i, 0] = u1
    U[i, 1] = u2
    U[i, 2] = u3

alpha_start_dim = alphav[0]
rho_l_start_dim = nrholv[0] 
rho_g_start_dim = nrhogv[0] 
alpha_end_dim = alphav[-1]
rho_l_end_dim = nrholv[-1] 
rho_g_end_dim = nrhogv[-1] 
p_start_dim = vnp[0]
ul_start_dim = nulv[0]
ug_start_dim = nugv[0]
ul_end_dim = nulv[-1]
ug_end_dim = nugv[-1]

# Simulação
U_final, time_values, U_values = simulate_pipeline(U, tol, N, nCg, nCl, nP_l0, nrho_l0, nPs, omega_c, omega_P, omega_rho, X, Z, mu_g, mu_l, eps, D, Lp, beta, delta_x, ntime, CFL, sigma, omega_u, AREA, G, alpha_start_dim, rho_l_start_dim, rho_g_start_dim, alpha_end_dim, rho_l_end_dim, rho_g_end_dim, nulv, nugv, njl, njg, p_start_dim, ul_start_dim, ug_start_dim, ul_end_dim, ug_end_dim, num_dummy_cells)

# Plotando os resultados
plt.figure(1)
plt.plot(np.linspace(0, nS, N), U_final[:, 0])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U1')
plt.title('Distribuição de U1 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(2)
plt.plot(np.linspace(0, nS, N), U_final[:, 1])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U2')
plt.title('Distribuição de U2 ao longo do tubo')
plt.grid(True)
plt.show()
    
plt.figure(3)
plt.plot(np.linspace(0, nS, N), U_final[:, 2])
plt.xlabel('Comprimento do tubo')
plt.ylabel('Variável conservativa U3')
plt.title('Distribuição de U3 ao longo do tubo')
plt.grid(True)
plt.show()

# plot_results(time_values, U1_values, 'U1')
# plot_results(time_values, U2_values, 'U2')
# plot_results(time_values, U3_values, 'U3')
