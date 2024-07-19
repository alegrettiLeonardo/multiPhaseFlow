# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:48:44 2024

@author: leonardoab
"""
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import voidFraction
import closureLaw
from scipy.integrate import solve_ivp

def safe_divide(numerator, denominator, fallback=np.inf):
    return np.where(denominator != 0, numerator / denominator, fallback)

def fun_dpds(s, p, mul, mug, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol):
    # Numeros adimensionais
    PIP = w_p / (w_rho * (w_u ** 2))
    PIG = G * Lr / (w_u ** 2)
    PIF = Lr / DH

    # Densidades adimensionais do líquido e gás
    nrhog = p / (Cg ** 2)
    nrhol = rho_l0 + (p - P_l0) / (Cl ** 2)

    # Velocidades superficiais adimensionais
    njl = mul / nrhol
    njg = mug / nrhog

    # Fração de vazio
    theta = fun_or_geo(s * Lr, Lp, beta, CA)

    if theta <= 0:
        alpha = voidFraction.FracaoVazio_comp(njl, njg, nrhog, nrhol, theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    else:
        alpha, Cd, Ud = voidFraction.FracaoVazio_swananda_ndim(njl, njg, nrhog, nrhol, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    # Velocidades adimensionais
    nug = njg / alpha
    nul = njl / (1 - alpha)

    # Derivadas da lei de fechamento
    dfda, dfdrhog, dfdrhol, dfdug, dfdul = closureLaw.Dlei_fechamento_or_ndim_simp(alpha, nrhol, nrhog, nul, nug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    # Elementos Aij, i,j = 1,2
    A11 = (1 - alpha) * nrhol + nrhol * nul * dfdul / dfda
    A12 = nrhol * nul * dfdug / dfda
    A21 = -nrhog * nug * dfdug / dfda
    A22 = alpha * nrhog - nrhog * nug * dfdug / dfda

    # Elementos B1 e B2
    B1 = -((1 - alpha) * nul / (Cl ** 2) + nrhol * nul * (dfdrhol / (Cl ** 2) + dfdrhog / (Cg ** 2)) / dfda)
    B2 = -(alpha * nug / (Cg ** 2) - nrhog * nug * (dfdrhol / (Cl ** 2) + dfdrhog / (Cg ** 2)) / dfda)

    # Termo que multiplica dp/ds
    auxa = (B1 * A22 - B2 * A12) / (A22 * A11 - A21 * A12)
    auxb = (B2 * A11 - B1 * A21) / (A22 * A11 - A21 * A12)

    auxc = mul * auxa + mug * auxb + PIP

    # Termo gravitacional e de atrito
    rhom = ((1 - alpha) * nrhol + alpha * nrhog)
    auxa = -PIG * np.sin(theta) * rhom

    jt = njl + njg
    dmu = MUG / MUL
    Rem = (w_rho * w_u * DH / MUL) * rhom * abs(jt) / ((1 - alpha) + dmu * alpha)

    fm = voidFraction.ffan(EPS / DH, Rem)

    auxb = -PIF * rhom * fm * jt * abs(jt) / 2.0

    dpds = (auxa + auxb) / auxc
    print(f"s: {s}, theta: {theta}, p: {p}, dpds: {dpds}")
    return dpds


def fun_or_geo(s, Lp, beta, CA):
    """
    Função que devolve o ângulo de inclinação local da tubulação em um sistema oleoduto-riser
    com riser em catenária e oleoduto reto com ângulo de inclinação beta.
    
    Args:
    s: posição ao longo do sistema oleoduto riser
    Lp: comprimento do oleoduto
    beta: ângulo de inclinação do oleoduto
    CA: constante da catenária
    
    Returns:
    theta: ângulo de inclinação local da tubulação
    """
    theta = np.where(s < Lp, -beta, np.arctan(np.sinh(np.arcsinh((s - Lp) / CA))))
    return theta

# Função para resolver o estado estacionário
def EstadoEstacionario_ndim_simp(N, mul, mug, Ps, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol):
    # Discretização do sistema oleoduto riser com Lr como escala de comprimento
    ds = ((Lp + Lr) / Lr) / (N - 1)
    tspan = np.linspace(0, (Lp + Lr) / Lr, N)

    # Defina as opções de controle do integrador solve_ivp
    options = {
        'method': 'DOP853', # Método de integração (pode ser 'BDF', 'DOP853', 'LSODA')
        'rtol': 100 * tol,
        'atol': tol,
        'max_step': ds,
        'first_step': ds / 10000.0,  # Ajuste o valor inicial do passo conforme necessário
    }
        
    # Integração da pressão do topo do riser até o início do oleoduto
    # sol = solve_ivp(fun_dpds_wrapper, [tspan[0], tspan[-1]], [Ps], **options)
    sol = solve_ivp(fun_dpds, [tspan[0], tspan[-1]], [Ps], args=(mul, mug, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol), t_eval=tspan, **options)
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Inverter os vetores de saída para corresponder ao MATLAB
    vns = sol.t[::-1]
    vnp = sol.y[0][::-1]
    
    # Determinar densidades do gás e do líquido, velocidades superficiais, fração de vazio e velocidades do líquido e gás
    nrhogv = np.zeros(N)
    nrholv = np.zeros(N)
    alphav = np.zeros(N)
    nugv = np.zeros(N)
    nulv = np.zeros(N)
    thetav = np.zeros(N)
    for i in range(N):
        nrhogv[i] = vnp[i] / (Cg ** 2)
        nrholv[i] = (rho_l0 + (vnp[i] - P_l0) / (Cl ** 2))
        njg = mug / nrhogv[i]
        njl = mul / nrholv[i]
        theta = fun_or_geo(vns[i] * Lr, Lp, beta, CA)
        thetav[i] = theta
        if theta <= 0:
            alphav[i] = voidFraction.FracaoVazio_comp(njl, njg, nrhogv[i], nrholv[i], -theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        else:
            alphav[i], Cd, Ud = voidFraction.FracaoVazio_swananda_ndim(njl, njg, nrhogv[i], nrholv[i], theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
            
        nugv[i] = njg / alphav[i]
        nulv[i] = njl / (1 - alphav[i])

    return vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav

# Exemplo de chamada da função (valores fictícios)
# N = 1000
# mul = 1.0
# mug = 0.1
# Ps = 201325.0
# Lp = 15.0
# Lr = 10.0
# CA = 1.0
# beta = 0.1
# DH = 0.1
# AREA = 1.0
# EPS = 0.001
# G = 9.81
# Cl = 1.0
# Cg = 1.0
# rho_l0 = 1000.0
# P_l0 = 101325.0
# MUL = 0.001
# MUG = 0.0001
# sigma = 0.072
# w_p = 1.0
# w_u = 1.0
# w_rho = 1.0
# tol = 1e-3

#vns, vnp, nrhogv, nrholv, alphav, nugv, nulv = EstadoEstacionario_ndim_simp(1001, 0.08091637746259303, 1.714336751138821e-05, 151987.5, 10.0, 12.54625399319707, 3.018181937174462, 0.05235987755982989, 0.1016, 0.008107319665559963, 4.6e-05, 9.81, 1498.0, 343.0, 998.0, 101325.0, 0.001, 1.81e-05, 0.0728, 101325.0, 0.01, 998.0, 1e-15)
# print(vns)
# print(vnp)
# print(nrhogv)
# print(nrholv)
# print(alphav)
# print(nugv)
# print(nulv)