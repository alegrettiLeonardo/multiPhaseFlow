# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:28:36 2024

@author: leonardoab
"""

import numpy as np
import voidFraction
import closureLaw
import math

# def Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
#     """
#     Derivadas da relação de deslizamento adimensional em relação a alpha, rhog, rhol, ul e ug.

#     Parameters:
#         alpha: fração de vazio em um ponto do tubo;
#         rhol: densidade adimensional do líquido;
#         rhog: densidade adimensional do gás;
#         ul: velocidade adimensional do líquido em um ponto do tubo;
#         ug: velocidade adimensional do gás em um ponto do tubo;
#         theta: ângulo de inclinação do tubo;
#         DH: diâmetro hidráulico do tubo;
#         AREA: área seccional do tubo;
#         EPS: rugosidade do tubo;
#         G: aceleração da gravidade;
#         MUL: viscosidade dinâmica do líquido;
#         MUG: viscosidade dinâmica do gás;
#         sigma: não especificado no código fornecido;
#         w_u: escala de velocidade;
#         w_rho: escala de densidade;
#         tol: tolerância numérica.

#     Returns:
#         dfda, dfdrhog, dfdrhol, dfdug, dfdul: Derivadas em relação a alpha, rhog, rhol, ug e ul, respectivamente.
#     """
#     # Parâmetro de distribuição e velocidade de deslizamento
#     Cd, Ud = CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

#     # Derivadas do parâmetro de distribuição e da velocidade de deslizamento
#     dCdda, dCddrhog, dCddrhol, dCddug, dCddul, dUdda, dUddrhog, dUddrhol = DCdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

#     jt = (1 - alpha) * ul + alpha * ug

#     # Derivada em relação a alpha
#     dfda = -ug + (Cd * jt + Ud / w_u) + alpha * (dCdda * jt + dUdda / w_u + Cd * (-ul + ug))

#     # Derivada em relação a rhog
#     dfdrhog = alpha * (dCddrhog * jt + dUddrhog / w_u)

#     # Derivada em relação a rhol
#     dfdrhol = alpha * (dCddrhol * jt + dUddrhol / w_u)

#     # Derivada em relação a ul
#     dfdul = alpha * (dCddul * jt + (1 - alpha) * Cd)

#     # Derivada em relação a ug
#     dfdug = -alpha + alpha * (dCddug * jt + alpha * Cd)

#     return dfda, dfdrhog, dfdrhol, dfdug, dfdul

# Adicione as funções CdUd_swananda e DCdUd_swananda aqui ou importe-as, já que não foram fornecidas

def DFduj(alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, w_P, tol):
    """
    Avalia numericamente o jacobiano do vetor F em relação às variáveis u_j.

    Parameters:
        alpha: fração de vazio em um ponto do tubo;
        rhol: densidade adimensional do líquido;
        rhog: densidade adimensional do gás;
        ul: velocidade adimensional do líquido em um ponto do tubo;
        ug: velocidade adimensional do gás em um ponto do tubo;
        Cl: velocidade adimensional do som no líquido;
        Cg: velocidade adimensional do som no gás;
        theta: ângulo de inclinação do tubo;
        DH: diâmetro hidráulico do tubo;
        AREA: área seccional do tubo;
        EPS: rugosidade do tubo;
        G: aceleração da gravidade;
        MUL: viscosidade dinâmica do líquido;
        MUG: viscosidade dinâmica do gás;
        sigma: não especificado no código fornecido;
        w_u: escala de velocidade;
        w_rho: escala de densidade;
        w_P: escala de pressão;
        tol: tolerância numérica.

    Returns:
        MJF: Jacobiano do vetor F em relação às variáveis u_j (3x3 matriz).
    """
    # Derivadas da lei de fechamento
    dfda, dfdrhog, dfdrhol, dfdug, dfdul = Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    PIP = w_P / (w_rho * (w_u**2))

    MJF = np.zeros((3, 3))

    # dF1du1
    t1 = Cg**2
    t2 = rhog * t1
    t3 = Cl**2
    t7 = alpha * (-rhol * t3 + t2) - t2
    t9 = dfdul
    t12 = dfda
    t14 = dfdrhol
    t16 = dfdrhog
    t22 = -1 + alpha
    t27 = dfdug
    MJF[0, 0] = 1 / t7 / (rhog * alpha * t9 + t27 * t22 * rhol) * alpha * (t9 * t7 * ul - t22 * (-t12 * alpha * t3 + (t14 * t1 + t16 * t3) * rhog) * rhol) * rhog

    # dF1du2
    t1 = Cg**2
    t2 = rhog * t1
    t3 = Cl**2
    t7 = alpha * (-rhol * t3 + t2) - t2
    t9 = dfdug
    t11 = -1 + alpha
    t13 = dfda
    t15 = dfdrhol
    t17 = dfdrhog
    t28 = dfdul
    MJF[0, 1] = 1 / t7 / (rhog * alpha * t28 + t9 * t11 * rhol) * rhol * t11 * (-t9 * ug * t7 + alpha * rhog * (t13 * t11 * t1 - (t15 * t1 + t17 * t3) * rhol))

    # dF1du3
    t1 = 1 - alpha
    t3 = dfdug
    t7 = dfdul
    MJF[0, 2] = -1 / (rhog * alpha * t7 - t3 * rhol * t1) * t3 * rhol * t1

    # dF2du1
    t1 = Cg**2
    t2 = rhog * t1
    t3 = Cl**2
    t7 = alpha * (-rhol * t3 + t2) - t2
    t9 = dfdul
    t12 = dfda
    t14 = dfdrhol
    t16 = dfdrhog
    t22 = -1 + alpha
    t27 = dfdug
    MJF[1, 0] = -1 / t7 / (rhog * alpha * t9 + t27 * t22 * rhol) * alpha * (t9 * t7 * ul - t22 * (-t12 * alpha * t3 + (t14 * t1 + t16 * t3) * rhog) * rhol) * rhog

    # dF2du2
    t1 = Cg**2
    t2 = rhog * t1
    t3 = Cl**2
    t7 = alpha * (-rhol * t3 + t2) - t2
    t9 = dfdug
    t11 = -1 + alpha
    t13 = dfda
    t15 = dfdrhol
    t17 = dfdrhog
    t28 = dfdul
    MJF[1, 1] = -1 / t7 / (rhog * alpha * t28 + t9 * t11 * rhol) * rhol * t11 * (-t9 * ug * t7 + alpha * rhog * (t13 * t11 * t1 - (t15 * t1 + t17 * t3) * rhol))

    # dF2du3
    t2 = dfdul
    t5 = dfdug
    MJF[1, 2] = 1 / (t5 * (-1 + alpha) * rhol + rhog * alpha * t2) * t2 * alpha * rhog

    # dF3du1
    t1 = alpha * rhog
    t2 = Cg**2
    t3 = rhog * t2
    t4 = Cl**2
    t6 = -rhol * t4 + t3
    t8 = (alpha * t6) - t3
    t11 = dfdul
    t12 = -1 + alpha
    t13 = t12 * rhol
    t14 = dfdug
    t19 = 1 / (rhog * alpha * t11 + t14 * t13)
    t21 = 1 / t8
    t22 = ul * t21
    t27 = dfda
    t29 = dfdrhol
    t31 = dfdrhog
    t35 = -t27 * alpha * t4 + (t29 * t2 + t31 * t4) * rhog
    t55 = ul**2
    t73 = rhog**2
    MJF[2, 0] = ug * (2 * t21 * t19 * alpha * t12 * t35 * rhol * rhog - 2 * t22 * t19 * t11 * t8 * t1) + t55 * t21 * t19 * (-t13 * t14 * t8 - t1 * t11 * (-alpha * t6 + t3)) - 2 * t22 * t19 * t12 * t35 * rhol * t1 + t21 * t19 * (-t14 * rhol * t12 * t2 * t4 * rhog * PIP - t73 * alpha * PIP * t11 * t4 * t2)

    # dF3du2
    t1 = Cg**2
    t2 = rhog * t1
    t3 = Cl**2
    t7 = alpha * (-rhol * t3 + t2) - t2
    t9 = dfdul
    t10 = t9 * alpha
    t13 = -1 + alpha
    t15 = dfdug
    t19 = t13 * rhol
    t23 = 1 / (rhog * t10 + t15 * t19)
    t25 = 1 / t7
    t26 = ug**2
    t36 = t13**2
    t37 = dfda
    t40 = rhol**2
    t41 = t13 * t40
    t42 = dfdrhol
    t44 = dfdrhog
    t49 = 2 * t37 * t36 * rhol * t1 - 2 * (t42 * t1 + t44 * t3) * t41
    t62 = t3 * t1
    MJF[2, 1] = t26 * t25 * t23 * (t15 * rhol * t13 * t7 - t10 * t7 * rhog) + ug * (-2 * ul * t25 * t23 * t15 * t7 * t19 - t25 * t23 * alpha * t49 * rhog) + ul * t25 * t23 * alpha * t49 * rhog + t25 * t23 * (-PIP * alpha * rhog * rhol * t9 * t62 - t15 * t62 * PIP * t41)

    # dF3du3
    t2 = (-1 + alpha) * rhol
    t3 = dfdug
    t7 = dfdul
    MJF[2, 2] = 1 / (rhog * alpha * t7 + t3 * t2) * (2 * t7 * ug * alpha * rhog + 2 * ul * t3 * t2)

    return MJF

def Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    Derivadas da lei de fechamento do sistema oleoduto-riser em relação às variáveis primitivas.
    Relação de equilíbrio local para o oleoduto (theta <= 0) e relação de deslizamento para o riser (theta > 0).

    Parameters:
        alpha: fração de vazio em um ponto do tubo;
        rhol: densidade adimensional do líquido;
        rhog: densidade adimensional do gás;
        ul: velocidade adimensional do líquido em um ponto do tubo;
        ug: velocidade adimensional do gás em um ponto do tubo;
        theta: ângulo de inclinação do tubo;
        DH: diâmetro hidráulico do tubo;
        AREA: área seccional do tubo;
        EPS: rugosidade do tubo do tubo;
        G: aceleração da gravidade;
        MUL: viscosidade dinâmica do líquido;
        MUG: viscosidade dinâmica do gás;
        sigma: não especificado no código fornecido;
        w_u: escala de velocidade;
        w_rho: escala de densidade;
        tol: tolerância numérica.

    Returns:
        dfda: Derivada de F em relação a alpha;
        dfdrhog: Derivada de F em relação a rhog;
        dfdrhol: Derivada de F em relação a rhol;
        dfdug: Derivada de F em relação a ug;
        dfdul: Derivada de F em relação a ul.
    """
    # Para o oleoduto
    if theta <= 0:
        dfda, dfdrhol, dfdrhog, dfdul, dfdug = closureLaw.DRelEquiLocal_pipe_comp(alpha, rhol, rhog, ul, ug, -theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
    elif theta > 0:
        # Para o riser
        dfda, dfdrhog, dfdrhol, dfdug, dfdul = closureLaw.Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    return dfda, dfdrhog, dfdrhol, dfdug, dfdul

def Dvar_primDvar_uj(n, alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
   """
   Avalia as derivadas das variáveis primitivas em relação a uma das variáveis conservativas u_j, j=1, 2, 3.

   Parâmetros:
   n: Número da variável primitiva (1, 2 ou 3)
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   alpha: Fração de vazio em um ponto do pipe
   Cl: Velocidade adimensional do som no líquido
   Cg: Velocidade adimensional do som no gás
   theta: Ângulo de inclinação do pipe
   AREA: Área seccional da tubulação do pipeline e riser
   DH: Diâmetro hidráulico da tubulação
   EPS: Rugosidade do tubo do pipeline
   G: Aceleração da gravidade
   MUL: Viscosidade dinâmica do líquido
   MUG: Viscosidade dinâmica do gás
   w_u: Escala de velocidade
   w_rho: Escala de densidade
   tol: Tolerância numérica

   Retorna:
   daduj: Derivada de alpha em relação a u_j
   dpduj: Derivada da pressão em relação a u_j
   dulduj: Derivada de ul em relação a u_j
   dugduj: Derivada de ug em relação a u_j
   """

   # Derivadas da lei de fechamento em relação às variáveis primitivas
   dfda, dfdrhog, dfdrhol, dfdug, dfdul = Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   if n == 1:
       t1 = Cl ** 2
       t5 = Cg ** 2
       daduj = 1 / (t5 * (-1 + alpha) * rhog - t1 * rhol * alpha) * t1 * alpha

       dpduj = -1 / (t5 * (-1 + alpha) * rhog - t1 * rhol * alpha) * t5 * t1 * rhog

       t2 = rhog * t5
       t3 = Cl ** 2
       t7 = alpha * (-rhol * t3 + t2) - t2
       t9 = dfdug
       t12 = dfda
       t14 = dfdrhol
       t16 = dfdrhog
       t27 = dfdul
       dulduj = 1 / t7 / (t9 * (-1 + alpha) * rhol + rhog * alpha * t27) * (t9 * ul * t7 + rhog * alpha * (-t12 * alpha * t3 + (t2 * t14 + t3 * t16) * rhog))

       t21 = -1 + alpha
       t28 = dfdug
       dugduj = 1 / (rhog * alpha * dfdul + t28 * t21 * rhol) / t7 * (-dfdul * ul * t7 + rhol * t21 * (-t12 * alpha * t3 + (t14 * t5 + t16 * t3) * rhog))

   elif n == 2:
       t1 = -1 + alpha
       t2 = Cg ** 2
       t7 = Cl ** 2
       daduj = 1 / (-t7 * rhol * alpha + t2 * t1 * rhog) * t2 * t1

       dpduj = -1 / (t2 * (-1 + alpha) * rhog - t7 * rhol * alpha) * rhol * t2 * t7

       t2 = rhog * t2
       t3 = Cl ** 2
       t7 = alpha * (-rhol * t3 + t2) - t2
       t9 = dfdug
       t11 = -1 + alpha
       t13 = dfda
       t15 = dfdrhol
       t17 = dfdrhog
       t29 = dfdul
       dulduj = 1 / (rhog * alpha * t29 + t9 * t11 * rhol) / t7 * (t9 * ug * t7 - alpha * (t13 * t11 * t2 - (t15 * t2 + t17 * t3) * rhol) * rhog)

       t8 = -t7 * rhol * alpha + t3 * t1 * rhog
       t28 = dfdug
       dugduj = 1 / (rhog * alpha * dfdul + t28 * t1 * rhol) / t8 * (-dfdul * ug * t8 - rhol * t1 * (t13 * t1 * t3 - (t15 * t3 + t17 * t7) * rhol))

   elif n == 3:
       daduj = 0
       dpduj = 0
       dulduj = 1 / (-rhog * alpha * dfdul + rhol * (1 - alpha) * dfdug) * dfdug
       dugduj = -1 / (-rhog * alpha * dfdul + rhol * (1 - alpha) * dfdug) * dfdul

   return daduj, dpduj, dulduj, dugduj