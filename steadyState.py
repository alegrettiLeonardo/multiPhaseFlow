# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:48:44 2024

@author: leonardoab
"""
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.integrate import solve_ivp
import voidFraction
import catenary
import closureLaw


def safe_divide(numerator, denominator, fallback=np.inf):
    return np.where(denominator != 0, numerator / denominator, fallback)

def fun_dpds(s, p, mul, mug, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol):
   """
   Função utilizada para avaliar dp/ds ao longo do sistema oleoduto-riser para
   avaliação do estado estacionário.
   
   Parâmetros:
   s: posição ao longo do sistema oleoduto riser (0 < s < L_p+L_r)
   p: pressão adimensional
   mul: densidade do líquido vezes a velocidade superficial do liquido (adimensional)
   mug: densidade do gas vezes a velocidade superficial do gas (adimensional)
   Lp: comprimento do oleoduto
   Lr: comprimento do riser
   CA: constante da catenária      
   beta: angulo de inclinação do oleoduto
   AREA: área seccional da tubulação do pipeline e riser
   DH: diâmetro hidráulico da tubulação
   EPS: rugosidade do tubo do pipeline
   G: aceleração da gravidade
   Cl: velocidade do som do líquido (adimensional)
   Cg: velocidade do som do gás (adimensional)
   rho_l0: valor de referência da densidade do líquido (adimensional)
   P_l0: valor de referência da pressão do líquido (adimensional)
   MUL: viscosidade dinâmica do líquido
   MUG: viscosidade dinâmica do gás
   sigma: tensão superficial
   w_p: escala de pressão
   w_u: escala de velocidade
   w_rho: escala de densidade
   tol: tolerância numérica
   
   Retorna:
   dpds: derivada da pressão em relação à posição
   """
   
   # Números adimensionais
   PIP = w_p / (w_rho * (w_u**2))
   PIG = G * Lr / (w_u**2)
   PIF = Lr / DH
   
   # Densidades adimensionais do líquido e gás
   nrhog = p / (Cg**2)
   nrhol = rho_l0 + (p - P_l0) / (Cl**2)
   
   # Velocidades superficiais adimensionais
   njl = mul / nrhol
   njg = mug / nrhog
   
   # Fração de vazio
   theta = catenary.fun_or_geo(s * Lr, Lp, beta, CA)
   
   if theta <= 0:
       alpha = voidFraction.FracaoVazio_comp(njl, njg, nrhog, nrhol, -theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
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
   B1 = -((1 - alpha) * nul / (Cl**2) + nrhol * nul * (dfdrhol / (Cl**2) + dfdrhog / (Cg**2)) / dfda)
   B2 = -(alpha * nug / (Cg**2) - nrhog * nug * (dfdrhol / (Cl**2) + dfdrhog / (Cg**2)) / dfda)
   
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

# Função para resolver o estado estacionário
def EstadoEstacionario_ndim_simp(N, mul, mug, Ps, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol):
    # Discretização do sistema oleoduto riser com Lr como escala de comprimento
    ds = ((Lp + Lr) / Lr) / (N - 1)
    tspan = np.linspace((Lp + Lr) / Lr, 0, N)
    
    # Defina as opções de controle do integrador solve_ivp
    options = {
        'method': 'LSODA', # Método de integração (pode ser 'BDF', 'DOP853', 'LSODA')
        'rtol': 100 * tol,
        'atol': tol,
        'max_step': ds,
        'first_step': ds / 10000.0,  
    }
        
    # Integração da pressão do topo do riser até o início do oleoduto
    sol = solve_ivp(lambda s, p: fun_dpds(s, p, mul, mug, Lp, Lr, CA, beta, DH, AREA, EPS, G, Cl, Cg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol),
                   [tspan[0], tspan[-1]], [Ps], t_eval=tspan, **options)
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Inverter os vetores de saída 
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
        #print(f"Step {i} - rhog: {nrhogv[i]}, rhol: {nrholv[i]}")
        njg = mug / nrhogv[i]
        njl = mul / nrholv[i]
        theta = catenary.fun_or_geo(vns[i] * Lr, Lp, beta, CA)
        thetav[i] = theta
        if theta <= 0:
            alphav[i] = voidFraction.FracaoVazio_comp(njl, njg, nrhogv[i], nrholv[i], -1.0*theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        else:
            alphav[i], Cd, Ud = voidFraction.FracaoVazio_swananda_ndim(njl, njg, nrhogv[i], nrholv[i], theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)    
        nugv[i] = njg / alphav[i]
        nulv[i] = njl / (1 - alphav[i])

    return vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav