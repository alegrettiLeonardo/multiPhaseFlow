# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 07:50:11 2024

@author: leonardoab
"""

import numpy as np
import catenary
import voidFraction
import steadyState
import closureLaw
import matplotlib.pyplot as plt

def vetor_h_ndim_simp(N, vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, mul, mug, Ps, Lp, Lr, CA, beta, DH, AREA, EPS, G, nCl, nCg, rho_l0, P_l0, MUL, MUG, sigma, w_p, w_u, w_rho, tol):
   theta = catenary.fun_or_geo(vns[0] * Lr, Lp, beta, CA)

   dfda, dfdrhog, dfdrhol, dfdug, dfdul = closureLaw.Dlei_fechamento_or_ndim_simp(
       alphav[0], nrholv[0], nrhogv[0], nulv[0], nugv[0], theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   dadu1, dpdu1, duldu1, dugdu1 = closureLaw.Dvar_primDvar_uj(
       1, alphav[0], nrholv[0], nrhogv[0], nulv[0], nugv[0], nCl, nCg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
   dadu2, dpdu2, duldu2, dugdu2 = closureLaw.Dvar_primDvar_uj(
       2, alphav[0], nrholv[0], nrhogv[0], nulv[0], nugv[0], nCl, nCg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   u1 = (1 - alphav[0]) * nrholv[0]
   auxa = dfdrhol * dpdu1 / (nCl ** 2) + dfdrhog * dpdu1 / (nCg ** 2) + dfda * dadu1 - dfdul * mul / (u1 ** 2)

   u2 = alphav[0] * nrhogv[0]
   auxb = dfdrhol * dpdu2 / (nCl ** 2) + dfdrhog * dpdu2 / (nCg ** 2) + dfda * dadu2 - dfdug * mug / (u2 ** 2)

   hv = np.zeros(3 * N - 3)

   F11 = (1 - alphav[0]) * nrholv[0] * nulv[0]
   F12 = (1 - alphav[1]) * nrholv[1] * nulv[1]
   F21 = alphav[0] * nrhogv[0] * nugv[0]
   F22 = alphav[1] * nrhogv[1] * nugv[1]
   hv[0] = auxa * (F11 - F12) + auxb * (F21 - F22)

   for i in range(1, N - 2):
       F11 = (1 - alphav[i]) * nrholv[i] * nulv[i]
       F12 = (1 - alphav[i + 1]) * nrholv[i + 1] * nulv[i + 1]
       hv[i] = F11 - F12

   F11 = (1 - alphav[N - 2]) * nrholv[N - 2] * nulv[N - 2]
   F12 = (1 - alphav[N - 1]) * nrholv[N - 1] * nulv[N - 1]
   F21 = alphav[N - 2] * nrhogv[N - 2] * nugv[N - 2]
   F22 = alphav[N - 1] * nrhogv[N - 1] * nugv[N - 1]
   hv[N - 2] = (F11 - F12) / nrholv[N - 1] + (F22 - F21) / nrhogv[N - 1]

   F21 = alphav[0] * nrhogv[0] * nugv[0]
   F22 = alphav[1] * nrhogv[1] * nugv[1]
   hv[N - 1] = F21 - F22

   for i in range(1, N - 1):
       F21 = alphav[i] * nrhogv[i] * nugv[i]
       F22 = alphav[i + 1] * nrhogv[i + 1] * nugv[i + 1]
       hv[N - 1 + i] = F21 - F22

   PIP = w_p / ((w_u ** 2) * w_rho)
   PIG = G * Lr / (w_u ** 2)
   PIF = Lr / DH
   dmu = MUG / MUL

   for i in range(N - 1):
       F31 = (1 - alphav[i]) * nrholv[i] * (nulv[i] ** 2) + alphav[i] * nrhogv[i] * (nugv[i] ** 2) + PIP * vnp[i]
       F32 = (1 - alphav[i + 1]) * nrholv[i + 1] * (nulv[i + 1] ** 2) + alphav[i + 1] * nrhogv[i + 1] * (nugv[i + 1] ** 2) + PIP * vnp[i + 1]

       theta = catenary.fun_or_geo(vns[i] * Lr, Lp, beta, CA)
       nrhom = nrholv[i] * (1 - alphav[i]) + nrhogv[i] * alphav[i]
       auxa = -PIG * np.sin(theta) * nrhom

       jt = nulv[i] * (1 - alphav[i]) + nugv[i] * alphav[i]
       Rem = (w_rho * w_u * DH / MUL) * nrhom * abs(jt) / ((1 - alphav[i]) + dmu * alphav[i])
       fm = voidFraction.ffan(EPS / DH, Rem)
       auxb = -PIF * nrhom * fm * jt * abs(jt) / 2.0
       auxc = auxa + auxb

       theta = catenary.fun_or_geo(vns[i + 1] * Lr, Lp, beta, CA)
       nrhom = nrholv[i + 1] * (1 - alphav[i + 1]) + nrhogv[i + 1] * alphav[i + 1]
       auxa = -PIG * np.sin(theta) * nrhom

       jt = nulv[i + 1] * (1 - alphav[i + 1]) + nugv[i + 1] * alphav[i + 1]
       Rem = (w_rho * w_u * DH / MUL) * nrhom * abs(jt) / ((1 - alphav[i + 1]) + dmu * alphav[i + 1])
       fm = voidFraction.ffan(EPS / DH, Rem)
       auxb = -PIF * nrhom * fm * jt * abs(jt) / 2.0
       auxd = auxa + auxb

       hv[2 * N - 2 + i] = F31 - F32 + (vns[i + 1] - vns[i]) * (auxd + auxc) / 2.0

   return hv


# Constantes
N = 101                          # número de pontos da malha
X = 6.435                       # comprimento do riser
Z = 9.886                       # altura do tubo
Lp = 10.0                       # comprimento do oleoduto
beta = 5.0                      # ângulo inicial em graus
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
time = 0.0009                   # tempo total de simulação
CFL = 0.9                       # número de Courant-Friedrichs-Lewy
tol = 1e-15
tola = tol * 100
AREA = np.pi * (D**2)/4.0
sigma = 7.28e-2
G = 9.81
jl = 6.0#e-2
jg = 1.0#e-2


# Escalas de pressão e densidade
omega_P = P_l0
omega_rho = rho_l0
omega_c = 1 / np.sqrt(omega_rho / omega_P)
# Escala de velocidade
omega_u = max(jl, jg)

# Vazão de massa de líquido e gás que entra no oleoduto
Pb = Ps + rho_l0 * G * Z
rhog = Pb / (Cg**2)
rhol = rho_l0 + (Pb - P_l0) / (Cl**2)
mul = rhol * jl 
mug = rhog * jg 

# Adimensionalização
nCg = Cg / omega_c
nCl = Cl / omega_c
nP_l0 = P_l0 / omega_P
nrho_l0 = rho_l0 / omega_rho
njl = jl / omega_u
njg = jg / omega_u
nPs = Ps / omega_P
nmul = mul / (rho_l0 * omega_u)
nmug = mug / (rho_l0 * omega_u)

# Parâmetro da catenária
CA = catenary.catenary_constant(X, Z, tol)
Lr = CA * np.sinh(X / CA)

# Estado Estacionário
vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, thetav = steadyState.EstadoEstacionario_ndim_simp(
   N, nmul, nmug, nPs, Lp, Lr, CA, np.radians(beta), D, AREA, eps, G, nCl, nCg, nrho_l0, nP_l0, mu_l, mu_g, sigma, omega_P, omega_u, omega_rho, tol
)

# Avalie vetor h no estado estacionário
hv = vetor_h_ndim_simp(
   N, vns, vnp, nrhogv, nrholv, alphav, nugv, nulv, nmul, nmug, nPs, Lp, Lr, CA, beta, D, AREA, eps, G, nCl, nCg, nrho_l0, nP_l0, mu_l, mu_g, sigma, omega_P, omega_u, omega_rho, tol
)

# Determine o erro com que o estado estacionário foi avaliado
errolinf = 0
errol2 = 0
for i in range(N):
   errol2 += hv[i]**2
   errolinf = max(errolinf, abs(hv[i]))

errol2 = np.sqrt(errol2)

print(f'nCg = {nCg}, nCl = {nCl}, nP_l0 = {nP_l0}, nrho_l0 = {nrho_l0}, njl = {njl}, njg = {njg}, nPs = {nPs}, nmul = {nmul}, nmug = {nmug}')

print(f'errol2 = {errol2}, errolinf = {errolinf}')

# Plotagens
plt.figure(1)
plt.title("Pressure")
plt.plot(vns, vnp, '+k')
plt.grid()

# plt.figure(2)
# plt.plot(vns, nrhogv, '+b')

# plt.figure(3)
# plt.plot(vns, nrholv, '+g')

plt.figure(4)
plt.title("Alpha")
plt.plot(vns, alphav, '*k')
plt.grid()

# plt.figure(5)
# plt.plot(vns, nulv, '*b')

# plt.figure(6)
# plt.plot(vns, nugv, '*g')

# plt.close('all')
