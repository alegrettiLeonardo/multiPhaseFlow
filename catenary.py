# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:00:22 2023

@author: leonardoab
"""

import math
import numpy as np

def fun_or_geo(s, Lp, beta, AC):
   """
   Função que devolve o ângulo de inclinação local da tubulação em um sistema
   oleoduto-riser com riser em catenária e oleoduto reto com ângulo de inclinação
   beta.

   Parâmetros:
   s : float
       Posição ao longo da tubulação.
   Lp : float
       Comprimento do oleoduto.
   beta : float
       Ângulo de inclinação do oleoduto.
   AC : float
       Constante da catenária.

   Retorna:
   theta : float
       Ângulo de inclinação local da tubulação.
   """
   if s < Lp:
       theta = -beta
   else:
       xa = np.arcsinh((s - Lp) / AC)
       theta = np.arctan(np.sinh(xa))
   
   return theta


def Dfungeo(s,VPARAM):
    #
    # this function generates the riser inclinarion angle at position s.
    # it needs a function that specifies the riser geometry.
    #
    # s: posição ao longo do riser
    # VPARAM: vetor de parametros da geometria do riser
    #
    A = VPARAM[0]
    LR = VPARAM[1]
    #
    # para geometria em catenária temos:
    #
    aux = LR*s/A
    dthetads = (1.0/(1.0 + aux**2))*(LR/A)
    
    return dthetads


def fungeo(s,VPARAM):
    #
    # this function generates the riser inclinarion angle at position s.
    # it needs a function that specifies the riser geometry.
    #
    # s: posição ao longo do riser
    # VPARAM: vetor de parametros da geometria do riser
    #
    A = VPARAM[0]
    LR = VPARAM[1]
    #
    # para geometria em catenária temos:
    #
    theta = math.atan2(LR*s,A)
    
    return theta

def catenaryf(x, z, a):
    auxa = x / a
    auxa = np.cosh(auxa)
    auxb = z / a
    value = auxb - auxa + 1.0
    return value

def catenary_constant(x, z, tol):
    # Adicionar verificação para x ser zero
    if x == 0:
        A = 0
    
    A_min = -math.sqrt(x**2 + z**2)
    fv_min = catenaryf(x, z, A_min)
    
    # Verificação adicional para evitar divisão por zero
    if z == 0:
        A_max = x  # Se z é zero, arcsinh(0) é 0, então A_max é apenas x
    else:
        A_max = x / np.arcsinh(z / x)
    
    fv_max = catenaryf(x, z, A_max)

    while abs(A_max - A_min) > tol:
        A = (A_max + A_min) / 2.0
        fv = catenaryf(x, z, A)

        if fv > 0:
            A_max = A
            fv_max = fv
        elif fv < 0:
            A_min = A
            fv_min = fv
        else:
            A_max = A
            A_min = A

    A = (A_max + A_min) / 2.0
    return A

def compute_catenary(s, z, Lp, theta_0):
    tol = 1e-15
    # Adicionar verificação para s ser zero
    if s == 0:
        return -theta_0 * (math.pi / 180)  # Convertendo -theta_0 graus para radianos
    
    A = catenary_constant(s, z, tol)
    LR = A * math.sinh(s / A)
    VPARAM = np.array([A, LR])
    
    if s < Lp:
        return -theta_0 * (math.pi / 180)  # Convertendo -theta_0 graus para radianos
    elif s == Lp:
        return 0  # 0 graus é 0 radianos
    else:
        theta_graus = Dfungeo(s, VPARAM)
        theta_radianos = theta_graus * (math.pi / 180)
        return theta_radianos 