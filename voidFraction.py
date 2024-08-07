# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:31:17 2023

@author: leonardoab
"""

import math
import closureLaw

def ffan(epsD, Re):
   if Re < 2000:
       f = 16 / Re
   elif Re > 2300:
       ft = math.log10((epsD ** 1.1098) / 2.8257 + 5.8506 / Re ** 0.8981)
       f = (epsD / 3.7065 - 5.0452 * ft / Re) ** -2
       if Re > 4000:
           f = (epsD / 3.7065 - 5.0452 * ft / Re) ** -2
   else:
       fl = 16 / 2000
       ft = math.log10((epsD ** 1.1098) / 2.8257 + 5.8506 / 2300 ** 0.8981)
       ft = epsD / 3.7065 - 5.0452 * ft / 2300
       f = (Re - 2000) * ft + (2300 - Re) * fl
       f = f / 300
   return f

def Dffan(Rem, EDIA):
   if Rem < 2000:
       D1ffan = -16 / (Rem ** 2)
       D2ffan = 0
   elif Rem > 2300:
       a = EDIA / 3.7065
       b = 5.0452
       c = (EDIA ** 1.1098) / 2.8257
       d = 5.8506
       e = 0.8981
       
       aux1 = a - b * math.log(c + d * (Rem ** -e)) / (Rem * math.log(10))
       aux2 = math.log(aux1) ** 3
       dfm = 1 / (aux1 * aux2)
       dfm = -(1 / 8) * math.log(10) ** 2 * dfm
       dfm = dfm * (b / math.log(10)) * ((d * e * (Rem ** (-2 - e)) / (c + d * (Rem ** -e))) + math.log(c + d * (Rem ** -e)) / (Rem ** 2))
       D1ffan = dfm
       
       a = 3.7065
       b = 5.0452
       c = 1.1098
       d = 2.8257
       e = 5.8506
       f = 0.8981
       g = math.log(10)
       auxa = (EDIA ** c) / d + e / (Rem ** f)
       auxb = b * c * (EDIA ** (c - 1.0))
       auxb = 1.0 / a - auxb / (Rem * d * g * auxa)
       auxc = EDIA / a - (b / (g * Rem)) * math.log(auxa)
       auxd = ((math.log(auxc)) ** 3) * auxc
       D2ffan = -(auxb * (g ** 2)) / (8.0 * auxd)
   else:
       fl = 16 / 2000
       a = EDIA / 3.7065
       b = 5.0452
       c = (EDIA ** 1.1098) / 2.8257
       d = 5.8506
       e = 0.8981
       
       aux1 = a - b * math.log(c + d * (2300 ** -e)) / (2300 * math.log(10))
       aux1 = (-4 * math.log(aux1) / math.log(10)) ** 2
       ft = 1 / aux1
       dfm = (ft - fl) / 300
       D1ffan = dfm
       
       x = 2300
       a = 3.7065
       b = 5.0452
       c = 1.1098
       d = 2.8257
       e = 5.8506
       f = 0.8981
       g = math.log(10)
       auxa = (EDIA ** c) / d + e / (x ** f)
       auxb = b * c * (EDIA ** (c - 1.0))
       auxb = 1.0 / a - auxb / (x * d * g * auxa)
       auxc = EDIA / a - (b / x) * math.log10(auxa)
       auxd = ((math.log(auxc)) ** 3) * auxc
       auxd = -(auxb * (g ** 2)) / (8.0 * auxd)
       D2ffan = auxd * (Rem - 2000) / 300
   
   return D1ffan, D2ffan

def fvgamma(alpha, gamma):
   """
   Esta função avalia alpha-1+gamma-(1/2*Pi)*sin(2*Pi*gamma)
   """
   fv = 2 * math.pi * gamma
   fv = math.sin(fv) / (2 * math.pi)
   fv = alpha - 1.0 + gamma - fv
   return fv

def alpha2gamma(alpha, tol):
   """
   Dado alpha, a rotina fornece o gamma correspondente, solução da equação:
   
   alpha = 1 - gamma + (1 / (2 * Pi)) * sin(2 * Pi * gamma)
   
   alpha - fração de vazio na tubulação
   gamma - fração de perímetro molhado
   """
   if abs(alpha) <= tol:
       return 1.0
   elif abs(1 - alpha) <= tol:
       return 0.0
   else:
       gamamax = 1.0
       gamamin = 0.0

       while abs(gamamax - gamamin) > tol:
           gama = (gamamax + gamamin) / 2.0
           fv = fvgamma(alpha, gama)

           if fv > 0:
               gamamax = gama
           elif fv < 0:
               gamamin = gama
           else:
               gamamax = gama
               gamamin = gama

       return (gamamax + gamamin) / 2.0

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def FracaoVazio_comp(jl, jg, rhog, rhol, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol):
   """
   Determina a fração de vazio no tubo utilizando a relação de equilíbrio local
   para escoamento estratificado. Assume-se escoamento estratificado.

   Args:
       jl (float): Velocidade superficial adimensional do líquido em um ponto do pipe.
       jg (float): Velocidade superficial adimensional do gás em um ponto do pipe.
       rhol (float): Densidade adimensional do líquido.
       rhog (float): Densidade adimensional do gás.
       BETA (float): Ângulo de inclinação do pipe (positivo para escoamento descendente).
       AREA (float): Área seccional da tubulação do pipeline e riser.
       D (float): Diâmetro da tubulação.
       EPS (float): Rugosidade do tubo do pipeline.
       G (float): Aceleração da gravidade.
       MUL (float): Viscosidade dinâmica do líquido.
       MUG (float): Viscosidade dinâmica do gás.
       w_u (float): Escala de velocidade.
       w_rho (float): Escala de densidade.
       tol (float): Tolerância numérica.

   Returns:
       float: Fração de vazio em um ponto do pipe.
   """

   alpha_max = 1.0 - math.sqrt(tol)
   alpha_min = math.sqrt(tol)
   def calculate_ul_ug(alpha):
       """Calcula ul e ug com base em alpha."""
       ul = jl / (1 - alpha)
       ug = jg / alpha
       return ul, ug

   while abs(alpha_max - alpha_min) > tol:
       alpha = (alpha_max + alpha_min) / 2.0
       ul, ug = calculate_ul_ug(alpha)
       fv = closureLaw.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, alpha, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)

       if fv < 0:
           alpha_max = alpha
       elif fv > 0:
           alpha_min = alpha
       else:
           alpha_max = alpha
           alpha_min = alpha
   return (alpha_max + alpha_min) / 2.0

def FracaoVazio_swananda_ndim(jl, jg, rhog, rhol, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
   """
   Determina a fração de vazio no tubo utilizando a relação de equilíbrio local
   para escoamento estratificado. Assume-se escoamento estratificado.

   Parâmetros:
   jl: velocidade superficial adimensional do líquido em um ponto do pipe
   jg: velocidade superficial adimensional do gás em um ponto do pipe
   rhol: densidade adimensional do líquido
   rhog: densidade adimensional do gás
   theta: ângulo de inclinação do pipe (positivo para escoamento ascendente)
   D: diâmetro da tubulação
   AREA: área seccional da tubulação do pipeline e riser
   EPS: rugosidade do tubo do pipeline
   G: aceleração da gravidade
   MUL: viscosidade dinâmica do líquido
   MUG: viscosidade dinâmica do gás
   sigma: tensão superficial líquido-gás
   w_u: escala de velocidade
   w_rho: escala de densidade
   tol: tolerância numérica

   Retorna:
   alpha: fração de vazio em um ponto do pipe
   Cd: coeficiente de arrasto
   Ud: velocidade de arrasto
   """
   alpha_max = 1.0 - math.sqrt(tol)
   alpha_min = math.sqrt(tol)

   while abs(alpha_max - alpha_min) > tol:
       alpha = (alpha_max + alpha_min) / 2.0
       ul = jl / (1 - alpha)
       ug = jg / alpha
       fv = closureLaw.drift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

       if fv > 0:
           alpha_max = alpha
       elif fv < 0:
           alpha_min = alpha
       else:
           alpha_max = alpha
           alpha_min = alpha

   alpha = (alpha_max + alpha_min) / 2.0
   Cd, Ud = closureLaw.CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   return alpha, Cd, Ud