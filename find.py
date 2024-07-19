# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 07:26:46 2023

@author: leonardoab
"""

import math
import numpy as np
import scipy.optimize as optimize
import closureLaw

def pressure_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0):
    # Calculate constants
    auxa = (Cg / Cl)**2
    auxb = -P_l0 / (Cl**2) + rho_l0
    auxc = -u1 - u2 * auxa + auxb

    # Discriminant of quadratic equation
    delta = auxc**2 + 4.0 * u2 * auxa * auxb

    # Calculate two possible solutions
    P1 = (-auxc - math.sqrt(delta)) / 2
    P1 = P1 * (Cl**2)

    P2 = (-auxc + math.sqrt(delta)) / 2
    P2 = P2 * (Cl**2)

    # Choose the positive solution
    if P1 > 0:
        P = P1
    elif P2 > 0:
        P = P2
    else:
        # No positive solution found
        P = None

    return P


def alpha_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0, tol):
    # Calculate alpha based on conservative variables u1 and u2
    alpha = 0.5
    if abs(u2) < tol:
        alpha = 0
    else:
        # Constants
        auxa = (Cg / Cl)**2
        auxb = P_l0 / (Cl**2) - rho_l0

        # Intermediate calculation
        auxc = u1 + u2 * auxa + auxb

        # Discriminant of quadratic equation
        delta = auxc**2 - 4.0 * u2 * auxa * auxb

        # Calculate two possible solutions
        alpha1 = (auxc - math.sqrt(delta)) / (2 * auxb)
        alpha2 = (auxc + math.sqrt(delta)) / (2 * auxb)

        # Choose the valid solution within the range [0, 1]
        if 0 <= alpha1 <= 1:
            alpha = alpha1
        elif 0 <= alpha2 <= 1:
            alpha = alpha2
    return alpha

def find_ul_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol, tola):
    """
    Function to obtain ul and ug from the local equilibrium relationship.

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param BETA: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance
    :param tola: Tolerance to differentiate from zero

    :return: ul - Dimensionless liquid velocity at a point in the pipe
             ug - Dimensionless gas velocity at a point in the pipe
    """
    if abs(u2) < tola:
        ug = 0
        ul = u3 / u1
    elif abs(u1) < tola:
        ul = 0
        ug = u3 / u2
    elif u2 <= u1 and abs(u2) >= tola and abs(u1) >= tola:
        ug = find_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        ul = u3 / u1 - (u2 / u1) * ug
    elif u2 > u1 and abs(u1) >= tola and abs(u2) >= tola:
        ul = find_ul_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        ug = u3 / u2 - (u1 / u2) * ul

    return ul, ug

def find_one_ug_from_uj_dfr(ugmin, ugmax, u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit):
    """
    Find gas velocity (ug) from the drift relation using the Swananda correlation.

    :param ugmin: Minimum gas velocity
    :param ugmax: Maximum gas velocity
    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Adimensional liquid density
    :param rhog: Adimensional gas density
    :param ALPHA: Void fraction in a pipe
    :param theta: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipe
    :param EPS: Pipe roughness
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param sigma: Surface tension between liquid and gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance
    :param maxit: Maximum number of iterations

    :return: Calculated gas velocity (ug)
    """
    # Check if there is a root in the interval [ugmin,ugmax]
    ul = u3/u1 - (u2/u1) * ugmin
    fv1 = closureLaw.drift_flux_swananda(ul, ugmin, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    ul = u3/u1 - (u2/u1) * ugmax
    fv2 = closureLaw.drift_flux_swananda(ul, ugmax, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    if fv1 * fv2 > 0:
        print("No root in the interval [ugmin, ugmax]")
        return 0
    elif fv1 * fv2 == 0:
        return ugmin if fv1 == 0 else ugmax
    elif fv1 * fv2 < 0:
        # Bisection
        if fv1 < 0:
            ug = ugmin
            du = ugmax - ugmin
        else:
            ug = ugmax
            du = ugmin - ugmax

        ul = u3/u1 - (u2/u1) * ug
        fv = closureLaw.drift_flux_swananda(ul, ug, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
        count = 0

        while abs(fv) > tol and abs(du) > tol / 10000 and count < maxit:
            count += 1
            du /= 2.0
            ugmid = ug + du
            ul = u3/u1 - (u2/u1) * ugmid
            fv = closureLaw.drift_flux_swananda(ul, ugmid, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

            if fv <= 0:
                ug = ugmid

        if count < maxit and abs(fv) < tol:
            return ug
        else:
            return 0
        
        
def find_all_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit):
   """
   Função que acha todos os valores de ug que satisfazem a relação de deriva
   com os parâmetros Cd e Ud dados pela relação de Swananda.

   Parâmetros:
   u1, u2, u3: Variáveis conservativas
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   ALPHA: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe
   D: Diâmetro da tubulação
   AREA: Área seccional da tubulação do pipeline e riser
   EPS: Rugosidade do tubo do pipeline
   G: Aceleração da gravidade
   MUL: Viscosidade dinâmica do líquido
   MUG: Viscosidade dinâmica do gás
   sigma: Tensão superficial líquido-gás
   w_u: Escala de velocidade
   w_rho: Escala de densidade
   tol: Tolerância numérica
   maxit: Número máximo de iterações

   Retorna:
   ugv: Lista de valores de ug que satisfazem a relação de deriva
   count: Contagem de valores encontrados
   """
   # Avaliar a velocidade máxima do gás
   ugmax = u3 / u2

   # Avaliar o número de intervalos para [0, ugmax]
   N = round(np.log10(ugmax))
   aux = 10 ** N

   if aux > ugmax:
       N = 10 * aux
   else:
       N = 10 ** (N + 2)

   count = 0
   counti = 0
   dug = ugmax / (N - 1)
   ugminv = []
   ugmaxv = []
   ugv = []

   for i in range(2, N + 1):
       if i == 2:
           ug1 = dug * (i - 2)
           ul1 = u3 / u1 - (u2 / u1) * ug1
           fv1 = closureLaw.drift_flux_swananda(ul1, ug1, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
           ug2 = dug * (i - 1)
           ul2 = u3 / u1 - (u2 / u1) * ug2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
       else:
           ug2 = dug * (i - 1)
           ul2 = u3 / u1 - (u2 / u1) * ug2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

       if fv1 * fv2 < 0:
           counti += 1
           ugminv.append(ug1)
           ugmaxv.append(ug2)
       elif fv1 * fv2 == 0:
           count += 1
           if abs(fv1) == 0:
               ugv.append(ug1)
           elif abs(fv2) == 0:
               ugv.append(ug2)

       ug1 = ug2
       fv1 = fv2

   if counti == 0:
       print('Nenhum intervalo contendo raiz da relação de deriva foi encontrado')
       if count > 0:
           print('Raízes da relação de deriva foram encontradas')
           print(count)
   else:
       for i in range(counti):
           ug = find_one_ug_from_uj_dfr(ugminv[i], ugmaxv[i], u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
           if ug > 0:
               count += 1
               ugv.append(ug)

   if count == 0:
       ugv.append(0)

   return ugv, count

# A função auxiliar drift_flux_swananda e find_one_ug_from_uj_dfr precisam ser implementadas no seu código Python.

def find_all_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit):
   """
   Função que acha todos os valores de ug que satisfazem a relação de deriva 
   com os parâmetros Cd e Ud dados pela relação de swananda.

   Parâmetros:
   uj: Variáveis conservativas
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   ALPHA: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe
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
   maxit: Número máximo de iterações

   Retorna:
   ugv: Lista de valores de ug que satisfazem a relação de deriva
   count: Contagem de valores encontrados
   """
   # Avaliar a velocidade máxima do gás
   ugmax = u3 / u2

   # Avaliar o número de intervalos para [0, ugmax]
   N = round(np.log10(ugmax))
   aux = 10 ** N

   if aux > ugmax:
       N = 10 * aux
   else:
       N = 10 ** (N + 2)

   count = 0
   counti = 0
   dug = ugmax / (N - 1)
   ugminv = []
   ugmaxv = []
   ugv = []

   for i in range(2, N + 1):
       if i == 2:
           ug1 = dug * (i - 2)
           ul1 = u3 / u1 - (u2 / u1) * ug1
           fv1 = closureLaw.drift_flux_swananda(ul1, ug1, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
           ug2 = dug * (i - 1)
           ul2 = u3 / u1 - (u2 / u1) * ug2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
       else:
           ug2 = dug * (i - 1)
           ul2 = u3 / u1 - (u2 / u1) * ug2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

       if fv1 * fv2 < 0:
           counti += 1
           ugminv.append(ug1)
           ugmaxv.append(ug2)
       elif fv1 * fv2 == 0:
           count += 1
           if abs(fv1) == 0:
               ugv.append(ug1)
           elif abs(fv2) == 0:
               ugv.append(ug2)

       ug1 = ug2
       fv1 = fv2

   if counti == 0:
       print('Nenhum intervalo contendo raiz da relação de deriva foi encontrado')
       if count > 0:
           print('Raízes da relação de deriva foram encontradas')
           print(count)
   else:
       for i in range(counti):
           ug = find_one_ug_from_uj_dfr(ugminv[i], ugmaxv[i], u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
           if ug > 0:
               count += 1
               ugv.append(ug)

   if count == 0:
       ugv.append(0)

   return ugv, count

def find_one_ul_from_uj_dfr(ulmin, ulmax, u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit):
   """
   Obtém ul a partir da relação de deriva utilizando a correlação de swananda.

   Parâmetros:
   ulmin: Mínima velocidade do líquido
   ulmax: Máxima velocidade do líquido
   u1, u2, u3: Variáveis conservativas
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   ALPHA: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe
   D: Diâmetro da tubulação
   AREA: Área seccional da tubulação do pipeline e riser
   EPS: Rugosidade do tubo do pipeline
   G: Aceleração da gravidade
   MUL: Viscosidade dinâmica do líquido
   MUG: Viscosidade dinâmica do gás
   sigma: Tensão superficial líquido-gás
   w_u: Escala de velocidade
   w_rho: Escala de densidade
   tol: Tolerância numérica
   maxit: Número máximo de iterações

   Retorna:
   value: Valor de ul que satisfaz a relação de deriva
   """
   # Verificar se há raiz no intervalo [ulmin, ulmax]
   ug = u3 / u2 - (u1 / u2) * ulmin
   fv1 = closureLaw.drift_flux_swananda(ulmin, ug, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   ug = u3 / u2 - (u1 / u2) * ulmax
   fv2 = closureLaw.drift_flux_swananda(ulmax, ug, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   if fv1 * fv2 > 0:
       print('Não há raiz no intervalo [ulmin, ulmax]')
       return 0
   elif fv1 * fv2 == 0:
       if fv1 == 0:
           return ulmin
       else:
           return ulmax
   elif fv1 * fv2 < 0:
       # Bisecção
       if fv1 < 0:
           ul = ulmin
           du = ulmax - ulmin
       else:
           ul = ulmax
           du = ulmin - ulmax

       ug = u3 / u2 - (u1 / u2) * ul
       fv = closureLaw.drift_flux_swananda(ul, ug, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

       count = 0

       while abs(fv) > tol and count < maxit:
           count += 1
           du /= 2.0
           ulmid = ul + du
           ug = u3 / u2 - (u1 / u2) * ulmid
           fv = closureLaw.drift_flux_swananda(ulmid, ug, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

           if fv <= 0:
               ul = ulmid

       if count < maxit and abs(fv) < tol:
           return ul
       else:
           return 0

# A função auxiliar drift_flux_swananda precisa ser implementada no seu código Python.


def find_all_ul_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit):
   """
   Função que acha todos os valores de ul que satisfazem a relação de deriva 
   com os parâmetros Cd e Ud dados pela relação de swananda.

   Parâmetros:
   uj: Variáveis conservativas
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   ALPHA: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe
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
   maxit: Número máximo de iterações

   Retorna:
   ulv: Lista de valores de ul que satisfazem a relação de deriva
   count: Contagem de valores encontrados
   """
   # Avaliar a velocidade máxima do líquido
   ulmax = u3 / u1

   # Avaliar o número de intervalos para [0, ulmax]
   N = round(np.log10(ulmax))
   aux = 10 ** N
   aux = max(aux, 1000)

   if aux > ulmax:
       N = 10 * aux
   else:
       N = 10 ** (N + 2)

   count = 0
   counti = 0
   dul = ulmax / (N - 1)
   ulminv = []
   ulmaxv = []
   ulv = []

   for i in range(2, N + 1):
       if i == 2:
           ul1 = dul * (i - 2)
           ug1 = u3 / u2 - (u1 / u2) * ul1
           fv1 = closureLaw.drift_flux_swananda(ul1, ug1, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
           ul2 = dul * (i - 1)
           ug2 = u3 / u2 - (u1 / u2) * ul2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
       else:
           ul2 = dul * (i - 1)
           ug2 = u3 / u2 - (u1 / u2) * ul2
           fv2 = closureLaw.drift_flux_swananda(ul2, ug2, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

       if fv1 * fv2 < 0:
           counti += 1
           ulminv.append(ul1)
           ulmaxv.append(ul2)
       elif fv1 * fv2 == 0:
           count += 1
           if abs(fv1) == 0:
               ulv.append(ul1)
           elif abs(fv2) == 0:
               ulv.append(ul2)

       ul1 = ul2
       fv1 = fv2

   if counti == 0:
       print('Nenhum intervalo contendo raiz da relação de deriva foi encontrado')
       if count > 0:
           print('Raízes da relação de deriva foram encontradas')
           print(count)
   else:
       for i in range(counti):
           ul = find_one_ul_from_uj_dfr(ulminv[i], ulmaxv[i], u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
           if ul > 0:
               count += 1
               ulv.append(ul)

   if count == 0:
       ulv.append(0)

   return ulv, count


def find_ul_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):
    """
    Function to obtain ul and ug from the local equilibrium relationship.

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param theta: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param sigma: Surface tension between liquid and gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance
    :param tola: Tolerance to differentiate from zero
    :param maxit: Maximum number of iterations

    :return: ul - Dimensionless liquid velocity at a point in the pipe
             ug - Dimensionless gas velocity at a point in the pipe
    """
    if abs(u2) < tola:
        ug = 0
        ul = u3 / u1
    elif abs(u1) < tola:
        ul = 0
        ug = u3 / u2
    elif u2 <= u1 and abs(u2) >= tola and abs(u1) >= tola:
        Fr = w_u / (D * G) ** 0.5
        Fr /= ((rhol - rhog) * np.cos(theta) / rhog) ** 0.5
        if Fr > 0.1:
            ug = find_all_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
            ul = u3 / u1 - (u2 / u1) * ug
        else:
            ugv, count = find_all_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
            if count > 0:
                ug = ugv[0]
                ul = u3 / u1 - (u2 / u1) * ug
            else:
                ulv, count = find_all_ul_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
                if count > 0:
                    ul = ulv[0]
                    ug = u3 / u2 - (u1 / u2) * ul
                else:
                    print('Inversion not possible for u1, u2, and u3 =', u1, u2, u3)
    elif u2 > u1 and abs(u1) >= tola and abs(u2) >= tola:
        Fr = w_u / (D * G) ** 0.5
        Fr /= ((rhol - rhog) * np.cos(theta) / rhog) ** 0.5
        if Fr > 0.1:
            ul = find_all_ul_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
            ug = u3 / u2 - (u1 / u2) * ul
        else:
            ulv, count = find_all_ul_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
            if count > 0:
                ul = ulv[0]
                ug = u3 / u2 - (u1 / u2) * ul
            else:
                ugv, count = find_all_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, maxit)
                if count > 0:
                    ug = ugv[0]
                    ul = u3 / u1 - (u2 / u1) * ug
                else:
                    print('Inversion not possible for u1, u2, and u3 =', u1, u2, u3)

    return ul, ug

def find_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol):
    """
    Function to obtain ug from the local equilibrium relationship. Note that u1 > u2.

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param BETA: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance

    :return: ug - Dimensionless gas velocity at a point in the pipe
    """
    ug1 = 0
    ul = u3 / u1 - (u2 / u1) * ug1
    fv1 = closureLaw.RelEquilLocalPipe_comp(ul, ug1, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    
    ug2 = u3 / u2
    ul = u3 / u1 - (u2 / u1) * ug2
    fv2 = closureLaw.RelEquilLocalPipe_comp(ul, ug2, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    fv = fv2

    while fv1 * fv2 > 0 and abs(fv) > tol:
        ug = ug1 - (ug2 - ug1) * fv1 / (fv2 - fv1)
        ul = u3 / u1 - (u2 / u1) * ug
        fv = closureLaw.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        fv1 = fv2
        fv2 = fv
        ug1 = ug2
        ug2 = ug

    if fv1 < 0:
        ug = ug1
        du = ug2 - ug1
    else:
        ug = ug2
        du = ug1 - ug2

    ul = u3 / u1 - (u2 / u1) * ug
    fv = closureLaw.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

    while abs(fv) > tol and abs(du) > tol / 100:
        du = du / 2.0
        ugmid = ug + du
        ul = u3 / u1 - (u2 / u1) * ugmid
        fv = closureLaw.RelEquilLocalPipe_comp(ul, ugmid, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

        if fv <= 0:
            ug = ugmid

    return ug

def find_ul_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol):
    """
    Function to obtain ul from the local equilibrium relationship. Note that u2 > u1.

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param BETA: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance

    :return: ul - Dimensionless liquid velocity at a point in the pipe
    """
    ul1 = 0
    ug = u3 / u2 - (u1 / u2) * ul1
    fv1 = closureLaw.RelEquilLocalPipe_comp(ul1, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

    ul2 = u3 / u1
    ug = u3 / u2 - (u1 / u2) * ul2
    fv2 = closureLaw.RelEquilLocalPipe_comp(ul2, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    fv = fv2

    while fv1 * fv2 > 0 and abs(fv) > tol:
        ul = ul1 - (ul2 - ul1) * fv1 / (fv2 - fv1)
        ug = u3 / u2 - (u1 / u2) * ul
        fv = closureLaw.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        fv1 = fv2
        fv2 = fv
        ul1 = ul2
        ul2 = ul

    if fv1 < 0:
        ul = ul1
        du = ul2 - ul1
    else:
        ul = ul2
        du = ul1 - ul2

    ug = u3 / u2 - (u1 / u2) * ul
    fv = closureLaw.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

    while abs(fv) > tol and abs(du) > tol / 100:
        du = du / 2.0
        ulmid = ul + du
        ug = u3 / u2 - (u1 / u2) * ulmid
        fv = closureLaw.RelEquilLocalPipe_comp(ulmid, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

        if fv <= 0:
            ul = ulmid

    return ul


def find_ul_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol, tola):
    """
    Function to obtain ul and ug from the local equilibrium relationship.

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param BETA: Angle of inclination of the pipe
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance
    :param tola: Tolerance to differentiate from zero

    :return: ul - Dimensionless liquid velocity at a point in the pipe
             ug - Dimensionless gas velocity at a point in the pipe
    """
    if abs(u2) < tola:
        ug = 0
        ul = u3 / u1
    elif abs(u1) < tola:
        ul = 0
        ug = u3 / u2
    elif u2 <= u1 and abs(u2) >= tola and abs(u1) >= tola:
        ug = find_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        ul = u3 / u1 - (u2 / u1) * ug
    elif u2 > u1 and abs(u1) >= tola and abs(u2) >= tola:
        ul = find_ul_from_uj(u1, u2, u3, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
        ug = u3 / u2 - (u1 / u2) * ul

    return ul, ug


def find_ulug_from_uj_genflw_simp(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):
    """
    This routine obtains the liquid and gas velocities as a function of conservative variables uj.
    For theta > 0 - non-stratified pattern
    For theta <= 0 - stratified pattern

    :param u1: Conservative variable 1
    :param u2: Conservative variable 2
    :param u3: Conservative variable 3
    :param rhol: Dimensionless density of liquid
    :param rhog: Dimensionless density of gas
    :param ALPHA: Void fraction at a point in the pipe
    :param theta: Angle of inclination of the pipe (positive upward and negative downward)
    :param D: Diameter of the pipe
    :param AREA: Cross-sectional area of the pipeline and riser
    :param EPS: Roughness of the pipeline tube
    :param G: Acceleration due to gravity
    :param MUL: Dynamic viscosity of the liquid
    :param MUG: Dynamic viscosity of the gas
    :param sigma: Surface tension between liquid and gas
    :param w_u: Velocity scale
    :param w_rho: Density scale
    :param tol: Numerical tolerance
    :param tola: Tolerance to decide if uj is considered zero
    :param maxit: Maximum number of iterations

    :return: ul - Dimensionless liquid velocity at a point in the pipe
             ug - Dimensionless gas velocity at a point in the pipe
             index - 0 for stratified, 1 for non-stratified
    """
    if theta > 0:
        index = 1
        ul, ug = find_ul_ug_from_uj_dfr(u1, u2, u3, rhol, rhog, ALPHA, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit)
    else:
        index = 0
        ul, ug = find_ul_ug_from_uj(u1, u2, u3, rhol, rhog, ALPHA, -theta, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol, tola)

    return ul, ug, index


def find_primvar_from_cvar_simp(u1, u2, u3, theta, Cg, Cl, rho_l0, P_l0, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit):
    # Function to find primitive variables from conservative variables

    # Obtain alpha from u1 and u2
    alpha = alpha_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0, tol)

    # Obtain pressure from u1 and u2
    P = pressure_fun_uj(u1, u2, Cg, Cl, rho_l0, P_l0)

    # Densities of gas and liquid
    rhog = P / (Cg**2)
    rhol = rho_l0 + (P - P_l0) / (Cl**2)

    # Velocities of gas and liquid
    ul, ug, index = find_ulug_from_uj_genflw_simp(u1, u2, u3, rhol, rhog, alpha, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol, tola, maxit)

    return alpha, rhol, rhog, P, ul, ug, index