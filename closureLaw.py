import numpy as np
import math 
import voidFraction

import numpy as np

def cdud_swanand_v3k(a, jl, jg, Rog, Rol, Mul, DH, theta, EPS, q, m, Sigma, g, precision):
   """
   Calcula a velocidade de drift e o parâmetro de distribuição com base na
   correlação de Swanand (2014).

   Parâmetros:
   a: Fração de vazio
   jl: Velocidade superficial do líquido (m/s)
   jg: Velocidade superficial do gás (m/s)
   Rog: Densidade do gás (kg/m³)
   Rol: Densidade do líquido (kg/m³)
   Mul: Viscosidade do líquido (kg.s/m)
   DH: Diâmetro hidráulico do tubo (m)
   theta: Ângulo de inclinação do tubo (rad)
   EPS: Rugosidade do tubo (m)
   q: Vazão volumétrica de líquido (m³/s)
   m: Vazão mássica de gás (kg/s)
   Sigma: Tensão superficial entre os fluidos (Pa)
   g: Aceleração da gravidade (m/s²)
   precision: Tolerância numérica

   Retorna:
   Cd: Parâmetro de distribuição
   Ud: Velocidade de drift (m/s)
   """
   # Cálculo dos adimensionais envolvidos
   Fr = np.sqrt(Rog / (Rol - Rog)) * jg / np.sqrt(g * DH * np.cos(theta))
   jt = jl + jg

   # Número de Reynolds
   Re = jt * Rol * DH / Mul
   BETA = jg / (jg + jl)
   x = m / (m + Rol * q)

   # Número de Laplace
   La = np.sqrt(Sigma / (g * (Rol - Rog))) / DH

   # Cálculo do fator de atrito de fanning - Colebrook (1939)
   f = 1
   df = 1
   while df > precision:
       b = -4 * np.log10(((EPS / DH) / 3.7) + (1.256 / (Re * np.sqrt(f))))
       f_new = 1 / (b ** 2)
       df = abs(f_new - f)
       f = f_new

   # Parâmetros constantes da simulação
   C1 = 0.2  # Constante igual a 0.2 para cilindros

   if Mul / 0.001 > 10:
       C2 = (0.434 / np.log10(Mul / 0.001)) ** 0.15
   else:
       C2 = 1

   if La < 0.025:
       C3 = (La / 0.025) ** 0.9
   else:
       C3 = 1

   if (Fr <= 0.1) and (theta < 0) and (theta >= -50 * np.pi / 180):
       C4 = -1
   else:
       C4 = 1

   if (Fr <= 0.1) and (theta < 0) and (theta > -50 * np.pi / 180):
       Co1 = 0
   else:
       Co1 = (C1 - C1 * np.sqrt(Rog / Rol)) * (((2.6 - BETA) ** 0.15) - np.sqrt(f)) * ((1 - x) ** 1.5)

   Co2 = (2 - (Rog / Rol) ** 2) / (1 + (Re / 1000) ** 2)
   Co3 = np.sqrt((1 + (Rog / Rol) ** 2 * np.cos(theta)) / (1 + np.cos(theta)))
   Co4 = 1 + (1000 / Re) ** 2

   Ud01 = 0.35 * np.sin(theta) + 0.45 * np.cos(theta)
   Ud02 = np.sqrt(g * DH * (Rol - Rog) / Rol)

   Cd0 = Co2 + (((Co3 ** (1 - a)) ** (2 / 5)) + Co1) / Co4
   Ud0 = Ud01 * Ud02 * ((1 - a) ** 0.5) * C2 * C3 * C4

   Cd = Cd0
   Ud = Ud0

   return Cd, Ud

def CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    Parâmetro de distribuição C0 e da velocidade de deslizamento Ud.

    Parâmetros:
    ul: velocidade adimensional do líquido em um ponto do pipe
    ug: velocidade adimensional do gás em um ponto do pipe
    rhol: densidade adimensional do líquido
    rhog: densidade adimensional do gás
    alpha: fração de vazio em um ponto do pipe
    theta: ângulo de inclinação do pipe
    AREA: área seccional da tubulação do pipeline e riser
    DH: diâmetro hidráulico da tubulação
    EPS: rugosidade do tubo do pipeline
    G: aceleração da gravidade
    MUL: viscosidade dinâmica do líquido
    MUG: viscosidade dinâmica do gás
    sigma: tensão superficial líquido-gás
    w_u: escala de velocidade
    w_rho: escala de densidade
    tol: tolerância numérica

    Retorna:
    C0: parâmetro de distribuição
    Ud: velocidade de deslizamento
    """
    # Velocidades superficiais
    jl = ul * (1 - alpha)
    jg = ug * alpha
    jt = jl + jg

    # Número de Froude
    Fr = np.sqrt(rhog / (rhol - rhog)) * (jg * w_u) / np.sqrt(G * DH * np.cos(theta))

    # Número de Reynolds Retp
    Retp = jt * rhol * w_rho * w_u * DH / MUL

    beta = jg / jt

    # Título
    x = (jg * rhog) / (jg * rhog + jl * rhol)

    # Número de Laplace
    La = np.sqrt(sigma / (G * w_rho * (rhol - rhog))) / DH

    # Cálculo do fator de atrito de Fanning - Colebrook (1939)
    df = 1
    ftp = 1
    while df > tol:
        b = -4 * np.log10((EPS / DH) / 3.7 + 1.256 / (Retp * np.sqrt(ftp)))
        f_new = 1 / b**2
        df = abs(f_new - ftp)
        ftp = f_new

    # Avaliação dos parâmetros C1, C2, C3 e C4 da velocidade de deslizamento
    C1 = 0.2  # Constante igual a 0.2 para cilindros

    # C2
    if MUL / 0.001 > 10:
        C2 = (0.434 / np.log10(MUL / 0.001))**0.15
    else:
        C2 = 1

    # C3
    if La < 0.025:
        C3 = (La / 0.025)**0.9
    else:
        C3 = 1

    # C4
    if (Fr <= 0.1) and (theta < 0) and (theta >= -50 * np.pi / 180):
        C4 = -1
    else:
        C4 = 1

    # Coeficiente C01 para o parâmetro de distribuição C0
    if (Fr <= 0.1) and (theta < 0) and (theta > -50 * np.pi / 180):
        C01 = 0
    else:
        C01 = (C1 - C1 * np.sqrt(rhog / rhol)) * ((2.6 - beta)**0.15 - np.sqrt(ftp)) * ((1 - x)**1.5)

    # Parâmetro de distribuição
    aux = 1 + (Retp / 1000)**2
    C0a = 2 - (rhog / rhol)**2
    C0a = C0a / aux

    aux = 1 + (1000 / Retp)**2
    C0b = (1 + (rhog / rhol)**2 * np.cos(theta)) / (1 + np.cos(theta))
    C0b = np.sqrt(C0b)
    C0b = C0b**(1 - alpha)
    C0b = C0b**(2 / 5)
    C0b = (C0b + C01) / aux

    Cd = C0a + C0b

    # Velocidade de deslizamento
    Ud = (1 - alpha)**0.5
    Ud = Ud * np.sqrt((rhol - rhog) * DH * G / rhol)
    Ud = Ud * (0.35 * np.sin(theta) + 0.45 * np.cos(theta)) * C2 * C3 * C4

    return Cd, Ud

# def CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
#     """
#     Parâmetro de distribuição C0 e da velocidade de deslizamento Ud.

#     Parâmetros:
#     ul: velocidade adimensional do líquido em um ponto do pipe
#     ug: velocidade adimensional do gás em um ponto do pipe
#     rhol: densidade adimensional do líquido
#     rhog: densidade adimensional do gás
#     alpha: fração de vazio em um ponto do pipe
#     theta: ângulo de inclinação do pipe
#     AREA: área seccional da tubulação do pipeline e riser
#     DH: diâmetro hidráulico da tubulação
#     EPS: rugosidade do tubo do pipeline
#     G: aceleração da gravidade
#     MUL: viscosidade dinâmica do líquido
#     MUG: viscosidade dinâmica do gás
#     sigma: tensão superficial líquido-gás
#     w_u: escala de velocidade
#     w_rho: escala de densidade
#     tol: tolerância numérica

#     Retorna:
#     C0: parâmetro de distribuição
#     Ud: velocidade de deslizamento
#     """

#     # Velocidades superficiais
#     jl = ul * (1 - alpha)
#     jg = ug * alpha
#     jt = jl + jg

#     # Número de Froude
#     Fr = np.sqrt(rhog / (rhol - rhog)) * (jg * w_u) / np.sqrt(G * DH * np.cos(theta))

#     # Número de Reynolds Retp
#     Retp = jt * rhol * w_rho * w_u * DH / MUL

#     beta = jg / jt

#     # Título
#     x = (jg * rhog) / (jg * rhog + jl * rhol)

#     # Número de Laplace
#     La = np.sqrt(sigma / (G * w_rho * (rhol - rhog))) / DH

#     # Cálculo do fator de atrito de Fanning - Colebrook (1939)
#     def colebrook(EPS, DH, Retp, tol):
#         df = 1
#         ftp = 1
#         while df > tol:
#             b = -4 * np.log10((EPS / DH) / 3.7 + 1.256 / (Retp * np.sqrt(ftp)))
#             f_new = 1 / b**2
#             df = abs(f_new - ftp)
#             ftp = f_new
#         return ftp

#     ftp = colebrook(EPS, DH, Retp, tol)

#     # Avaliação dos parâmetros C1, C2, C3 e C4 da velocidade de deslizamento
#     C1 = 0.2  # constante igual a 0.2 para tubo redondo

#     # C2
#     C2 = (0.434 / np.log10(MUL / 0.001))**0.15 if MUL / 0.001 > 10 else 1

#     # C3
#     C3 = (La / 0.025)**0.9 if La < 0.025 else 1

#     # C4
#     C4 = -1 if (Fr <= 0.1) and (theta < 0) and (theta >= -50 * np.pi / 180) else 1

#     # Coeficiente C01 para o parâmetro de distribuição C0
#     C01 = 0 if (Fr <= 0.1) and (theta < 0) and (theta > -50 * np.pi / 180) else (C1 - C1 * np.sqrt(rhog / rhol)) * ((2.6 - beta)**0.15 - np.sqrt(ftp)) * ((1 - x)**1.5)

#     # Parâmetro de distribuição
#     C0a = (2 - (rhog / rhol)**2) / (1 + (Retp / 1000)**2)
#     C0b = np.sqrt((1 + (rhog / rhol)**2 * np.cos(theta)) / (1 + np.cos(theta)))**((1 - alpha) * (2 / 5))
#     C0b = (C0b + C01) / (1 + (1000 / Retp)**2)
#     Cd = C0a + C0b

#     # Velocidade de deslizamento
#     Ud = np.sqrt((1 - alpha) * (rhol - rhog) * DH * G / rhol) * (0.35 * np.sin(theta) + 0.45 * np.cos(theta)) * C2 * C3 * C4

#     return Cd, Ud


def DCdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    # derivada do parametro de distribuição C0 e da velocidade de deslizamento Ud
    # em relação a alph, rho_l, rho_g, u_l e u_g
    
    # velocidades superficiais
    jl = ul * (1 - alpha) * w_u
    jg = ug * alpha * w_u
    jt = jl + jg
    
    # numero de Froude
    Fr = np.sqrt(rhog / (rhol - rhog)) * jg / np.sqrt(G * DH * np.cos(theta))
    
    # numero de Reynolds Retp
    Retp = jt * rhol * w_rho * DH / MUL
    
    beta = jg / jt
    
    # titulo
    x = (jg * rhog) / (jg * rhog + jl * rhol)
    
    # número de Laplace
    La = np.sqrt(sigma / (G * w_rho * (rhol - rhog))) / DH
    
    # cálculo do fator de atrito de fanning - Colebrook (1939)
    df = 1
    ftp = 1
    while df > tol:
        b = -4 * np.log10((EPS / DH / 3.7) + (1.256 / (Retp * np.sqrt(ftp))))
        f_new = 1 / (b ** 2)
        df = abs(f_new - ftp)
        ftp = f_new
    
    # avaliação dos parâmetro C1, C2, C3 e C4 da velocidade de deslizamento
    C1 = 0.2  # constante igual a 0.2 para tubo redondo
    
    # C2
    if MUL / 0.001 > 10:
        C2 = (0.434 / np.log10(MUL / 0.001)) ** 0.15
    else:
        C2 = 1
    
    # C3
    if La < 0.025:
        C3 = (La / 0.025) ** 0.9
    else:
        C3 = 1
    
    # C4
    if (Fr <= 0.1) and (theta < 0) and (theta >= -50 * np.pi / 180):
        C4 = -1
    else:
        C4 = 1
    
    # coeficiente C01 para o parametro de distribuição C0
    if (Fr <= 0.1) and (theta < 0) and (theta > -50 * np.pi / 180):
        C01 = 0
    else:
        C01 = (C1 - C1 * np.sqrt(rhog / rhol)) * (((2.6 - beta) ** 0.15) - np.sqrt(ftp)) * ((1 - x) ** 1.5)
    
    # derivadas das quantidades de interesse
    
    # derivadas do titulo
    # em relação a alpha
    t1 = (rhog * ug)
    t2 = (alpha * rhog)
    t7 = ug * t2 + rhol * (1 - alpha) * ul
    t10 = t7 ** 2
    dxda = 1 / t7 * t1 - (-rhol * ul + t1) / t10 * ug * t2
    
    # em relação a rhog
    t7 = rhog * alpha * ug + rhol * (1 - alpha) * ul
    t10 = alpha ** 2
    t12 = ug ** 2
    t13 = t7 ** 2
    dxdrhog = 1 / t7 * ug * alpha - 1 / t13 * t12 * t10 * rhog
    
    # em relação a rhol
    t2 = (rhog * alpha * ug)
    t3 = 1 - alpha
    t7 = (ul * t3 * rhol + t2) ** 2
    dxdrhol = -ul * t3 / t7 * t2
    
    # em relação a ul
    t2 = (rhog * alpha * ug)
    t3 = 1 - alpha
    t7 = (ul * t3 * rhol + t2) ** 2
    dxdul = -t3 * rhol / t7 * t2
    
    # em relação a ug
    t1 = (alpha * rhog)
    t6 = ug * t1 + rhol * (1 - alpha) * ul
    t9 = (rhog ** 2)
    t10 = alpha ** 2
    t12 = t6 ** 2
    dxdug = 1 / t6 * t1 - 1 / t12 * ug * t10 * t9
    
    # derivadas do Reynolds
    # em relação a alpha
    dRetpda = rhol * w_u * w_rho * (ug - ul) * DH / MUL
    
    # em relação a rhol
    dRetpdrhol = w_u * w_rho * (alpha * ug + (1 - alpha) * ul) * DH / MUL
    
    # em relação a ul
    dRetpdul = rhol * w_u * w_rho * (1 - alpha) * DH / MUL
    
    # em relação a ug
    dRetpdug = rhol * w_u * w_rho * alpha * DH / MUL
    
    # derivada do fator de atrito de fanning ftp em relação a Retp
    t4 = ftp
    t5 = np.sqrt(t4)
    dftpdRetp = -0.2009600000e11 * t5 * t4 / (0.1244640591e10 * t5 * Retp * EPS + 0.1004800000e11 * t5 * DH + 0.5784093754e10 * DH) * DH / Retp
    
    # derivadas de C01
    if (Fr <= 0.1) and (theta < 0) and (theta > -50 * np.pi / 180):
        dC01da = 0
        dC01drhog = 0
        dC01drhol = 0
        dC01dul = 0
        dC01dug = 0
    else:
        # em relação a alpha
        t3 = np.sqrt(rhog / rhol)
        t5 = -t3 * C1 + C1
        t6 = (alpha * ug)
        t9 = t6 + (1 - alpha) * ul
        t10 = 1 / t9
        t12 = 0.26 - (t10 * t6)
        t13 = t12 ** (-0.85)
        t15 = t9 ** 2
        t23 = Retp
        t26 = ftp
        t27 = np.sqrt(t26)
        t30 = dRetpda
        t35 = x
        t36 = 1 - t35
        t37 = t36 ** 1.5
        t39 = t12 ** 0.15
        t42 = t36 ** 0.5
        t43 = dxda
        dC01da = t37 * (0.15 * (-t10 * ug + (ug - ul) / t15 * t6) * t13 - t30 * dftpdRetp / t27 / 2) * t5 - 1.5 * t43 * t42 * (t39 - t27) * t5
        
        # em relação a rhog
        t1 = 1 / rhol
        t3 = np.sqrt(t1 * rhog)
        t6 = (alpha * ug)
        t13 = (0.26 - (1 / (t6 + (1 - alpha) * ul) * t6)) ** 0.15
        t14 = Retp
        t17 = ftp
        t18 = np.sqrt(t17)
        t19 = t13 - t18
        t21 = x
        t22 = 1 - t21
        t23 = t22 ** 1.5
        t30 = t22 ** 0.5
        t31 = dxdrhog
        dC01drhog = -t23 * t19 * t1 / t3 * C1 / 2 - 1.5 * t31 * t30 * t19 * (-t3 * C1 + C1)
        
        # em relação a rhol
        t3 = np.sqrt(rhog / rhol)
        t7 = rhol ** 2
        t9 = (alpha * ug)
        t16 = (0.26 - (1 / (t9 + (1 - alpha) * ul) * t9)) ** 0.15
        t17 = Retp
        t20 = ftp
        t21 = np.sqrt(t20)
        t22 = t16 - t21
        t24 = x
        t25 = 1 - t24
        t26 = t25 ** 1.5
        t31 = -t3 * C1 + C1
        t34 = t25 ** 0.5
        t35 = dxdrhol
        dC01drhol = t26 * t22 * rhog / 2 / t7 / t3 * C1 - 1.5 * t35 * t34 * t22 * t31
        
        # em relação a ul
        t1 = np.sqrt(rhog / rhol)
        t3 = -t1 * C1 + C1
        t4 = alpha - 1
        t5 = t4 * rhol
        t8 = (alpha * ug)
        t11 = t8 + ul * t5
        t12 = 1 / t11
        t13 = 0.26 - (t12 * t8)
        t14 = t13 ** 0.15
        t15 = Retp
        t18 = ftp
        t19 = np.sqrt(t18)
        t20 = t14 - t19
        t22 = x
        t23 = 1 - t22
        t24 = t23 ** 1.5
        t27 = t23 ** 0.5
        t28 = dxdul
        dC01dul = 0.15 * t24 * t3 * (-t12 * t5 + t4 * w_u * w_rho * DH / MUL / t11 ** 2 * t8) * t13 ** (-0.85) - 1.5 * t28 * t27 * t20 * t3
        
        # em relação a ug
        t3 = np.sqrt(rhog / rhol)
        t5 = -t3 * C1 + C1
        t6 = (alpha * ug)
        t7 = alpha ** 2
        t11 = (t6 + (1 - alpha) * ul) ** 2
        t13 = 0.26 - (1 / np.sqrt(t6 + (1 - alpha) * ul) * t6)
        t14 = t13 ** (-0.85)
        t15 = Retp
        t18 = ftp
        t19 = np.sqrt(t18)
        t20 = dRetpdug
        t25 = x
        t26 = 1 - t25
        t27 = t26 ** 1.5
        t30 = t26 ** 0.5
        t31 = dxdug
        dC01dug = t27 * (0.15 * (-1 / np.sqrt(t6 + (1 - alpha) * ul) * alpha + t6 * w_u * w_rho * DH * rhol / MUL / t11 * alpha) * t14 - t20 * dftpdRetp / t19 / 2) * t5 - 1.5 * t31 * t30 * (t13 ** 0.15 - t19) * t5
    
    # avaliação da velocidade de deslizamento Ud
    Ud = (C1 - np.sqrt(rhog / rhol) * C1) * C2 * C3 * C4 * (((2.6 - beta) ** 0.15) - np.sqrt(ftp))
    
    # derivadas de Ud
    # em relação a alpha
    t4 = rhog / rhol
    t5 = np.sqrt(t4)
    t7 = C1 * t5
    t9 = C1 - t7
    t11 = 2.6 - beta
    t12 = t11 ** 0.15
    t13 = np.sqrt(ftp)
    t14 = t12 - t13
    t19 = dRetpda
    dUdda = t14 * t9 * C2 * C3 * C4 * -0.15 / t11 + t9 * C2 * C3 * C4 * (-dftpdRetp / t13 / 2 * t19)
    
    # em relação a rhog
    t1 = 1 / rhol
    t2 = np.sqrt(t1 * rhog)
    t5 = C1 * t2
    t6 = C1 - t5
    t12 = 2.6 - beta
    t13 = t12 ** 0.15
    t14 = ftp
    t17 = np.sqrt(t14)
    t18 = t13 - t17
    t20 = C2 * C3 * C4
    dUddrhog = -t6 * C2 * C3 * C4 * t18 * t1 / t2 * C1 / 2
    
    # em relação a rhol
    t4 = rhog / rhol
    t5 = np.sqrt(t4)
    t7 = C1 * t5
    t8 = C1 - t7
    t10 = C2 * C3 * C4
    t12 = 2.6 - beta
    t13 = t12 ** 0.15
    t14 = ftp
    t17 = np.sqrt(t14)
    t18 = t13 - t17
    t21 = rhol ** 2
    dUddrhol = t8 * t10 * t18 * rhog / t21 / t5 * C1 / 2
    
    # em relação a ul
    t4 = rhog / rhol
    t5 = np.sqrt(t4)
    t7 = C1 * t5
    t8 = C1 - t7
    t9 = 2.6 - beta
    t10 = t9 ** 0.15
    t11 = ftp
    t14 = np.sqrt(t11)
    t15 = t10 - t14
    t17 = C2 * C3 * C4
    t19 = dRetpdul
    dUddul = t15 * t8 * t17 * -0.15 / t9 + t8 * t17 * (-dftpdRetp / t14 / 2 * t19)
    
    # em relação a ug
    t4 = rhog / rhol
    t5 = np.sqrt(t4)
    t7 = C1 * t5
    t8 = C1 - t7
    t9 = 2.6 - beta
    t10 = t9 ** 0.15
    t11 = ftp
    t14 = np.sqrt(t11)
    t15 = t10 - t14
    t17 = C2 * C3 * C4
    t19 = dRetpdug
    dUdg = t15 * t8 * t17 * -0.15 / t9 + t8 * t17 * (-dftpdRetp / t14 / 2 * t19)
    
    return dC01da, dC01drhog, dC01drhol, dC01dul, dC01dug, dUdda, dUddrhog, dUddrhol, dUddul, dUdg


def drift_flux_swananda_ndim(ul, ug, rhol, rhog, alpha, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    # Avalia a relação de deslizamento com parâmetros Cd e Ud dados pela correlação de Swanada

    # Vazão volumétrica dimensional de líquido
    q = ul * (1 - alpha) * AREA * w_u  # m³/s

    # Vazão dimensional de massa de gás
    m = ug * alpha * AREA * rhog * w_u * w_rho  # kg/s

    # Densidades de líquido e gás
    ROL = rhol * w_rho
    ROG = rhog * w_rho

    # Velocidades superficiais dimensionais
    jl = ul * (1 - alpha) * w_u
    jg = ug * alpha * w_u

    Cd, Ud = CdUd_swananda(alpha, jl, jg, ROG, ROL, MUL, D, theta, EPS, q, m, sigma, G, w_u, w_rho, tol)

    # Relação de deriva
    njl = ul * (1 - alpha)
    njg = ug * alpha
    nUd = Ud / w_u

    auxa = -njg + alpha * (Cd * (njl + njg) + nUd)
    auxb = ul * (1 - alpha) * rhol + ug * alpha * rhog

    if abs(auxa) >= abs(auxb):
        value = auxa
    else:
        value = auxb

    return value

def drift_flux_swananda(ul, ug, rhol, rhog, alpha, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
   """
   Avalia a relação de deslizamento com parâmetros Cd e Ud dados pela correlação de swananda.

   Parâmetros:
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   alpha: Fração de vazio em um ponto do pipe
   theta: Ângulo de inclinação do pipe (positivo upward e negativo downward)
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

   Retorna:
   value: Valor da relação de deriva
   """
   # Vazão volumétrica dimensional de líquido
   q = ul * (1 - alpha) * AREA * w_u  # m³/s

   # Vazão dimensional de massa de gás
   m = ug * alpha * AREA * rhog * w_u * w_rho  # kg/s

   # Densidades do líquido e gás
   ROL = rhol * w_rho
   ROG = rhog * w_rho

   # Velocidades superficiais dimensionais
   jl = ul * (1 - alpha) * w_u
   jg = ug * alpha * w_u

   # Obter Cd e Ud usando a função cdud_swanand_v3k
   Cd, Ud = cdud_swanand_v3k(alpha, jl, jg, ROG, ROL, MUL, D, theta, EPS, q, m, sigma, G, tol)

   # Relação de deriva
   njl = ul * (1 - alpha)
   njg = ug * alpha
   nUd = Ud / w_u

   value = -njg + alpha * (Cd * (njl + njg) + nUd)
   # value = alpha - njg / (Cd * (njl + njg) + nUd)

   return value

def RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol):
   """
   Calcula o valor da relação de equilíbrio local adimensional.

   Parâmetros:
   ul: Velocidade adimensional do líquido em um ponto do pipe
   ug: Velocidade adimensional do gás em um ponto do pipe
   rhol: Densidade adimensional do líquido
   rhog: Densidade adimensional do gás
   ALPHA: Fração de vazio em um ponto do pipe
   BETA: Ângulo de inclinação do pipe
   AREA: Área seccional da tubulação do pipeline e riser
   D: Diâmetro da tubulação
   EPS: Rugosidade do tubo do pipeline
   G: Aceleração da gravidade
   MUL: Viscosidade dinâmica do líquido
   MUG: Viscosidade dinâmica do gás
   w_u: Escala de velocidade
   w_rho: Escala de densidade
   tol: Tolerância numérica

   Retorna:
   fun: Valor da relação de equilíbrio local adimensional
   """
   PID = (D * G) / (4 * (w_u ** 2))
   gamma = voidFraction.alpha2gamma(ALPHA, tol)
   gammai = np.sin(np.pi * gamma) / np.pi

   # Número de Reynolds para o líquido
   Rel = ((w_rho * w_u * D) / MUL) * rhol * abs((1 - ALPHA) * ul) / gamma

   # Número de Reynolds para o gás
   Reg = ((w_u * w_rho * D) / MUG) * rhog * abs(ALPHA * ug) / (1.0 - gamma + gammai)

   # Perímetros molhados
   Sl = np.pi * D * gamma
   Sg = np.pi * D * (1.0 - gamma)
   Si = np.pi * D * gammai

   # Diâmetros hidráulicos
   Dl = 4.0 * (1.0 - ALPHA) * AREA / Sl
   Dg = 4.0 * ALPHA * AREA / (Sg + Si)

   # Fatores de atrito
   if Rel >= tol:
       fl = voidFraction.ffan(EPS / Dl, Rel)
   else:
       fl = 0

   if Reg >= tol:
       fg = voidFraction.ffan(EPS / Dg, Reg)
   else:
       fg = 0

   fi = 0.0142  # mesmo que utilizado em fungamma.m

   # Cálculo de ui
   if Rel < 2000:
       ui = 1.8 * ul
   elif Rel > 2200:
       ui = ul
   else:
       ui = ul * (1.8 * (2200 - Rel) + (Rel - 2000)) / 200

   # Cálculo da relação de equilíbrio local
   auxa = fg * rhog * ug * abs(ug) * (1 - gamma) / ALPHA
   auxb = rhol * fl * ul * abs(ul) * gamma / (1.0 - ALPHA)
   auxc = fi * rhog * (ug - ui) * abs(ug - ui) * (gammai / (ALPHA * (1.0 - ALPHA)))
   auxd = PID * (rhol - rhog) * np.sin(BETA)

   fun = auxa / 2 - auxb / 2 + auxc / 2 + auxd

   return fun

def Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    # derivada da relação de deslizamento adimensional em relação a alpha, rhog, rhol, ul e ug.

    # parametro de distribuição e velocidade de deslizamento
    Cd, Ud = CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    # derivada do parametro de distribuição e da velocidade de deslizamento
    dC01da, dC01drhog, dC01drhol, dC01dul, dC01dug, dUdda, dUddrhog, dUddrhol, dUddul, dUdg = DCdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
    
    dCdda = dC01da
    dCddrhog = dC01drhog
    dCddrhol = dC01drhol
    dCddul = dC01dul
    dCddug = dC01dug
    
    jt = (1 - alpha) * ul + alpha * ug

    # derivada em relação a alpha
    dfda = -ug + (Cd * jt + Ud / w_u) + alpha * (dCdda * jt + dUdda / w_u + Cd * (-ul + ug))

    # derivada em relação a rhog
    dfdrhog = alpha * (dCddrhog * jt + dUddrhog / w_u)

    # derivada em relação a rhol
    dfdrhol = alpha * (dCddrhol * jt + dUddrhol / w_u)

    # derivada em relação a ul
    dfdul = alpha * (dCddul * jt + (1 - alpha) * Cd)

    # derivada em relação a ug
    dfdug = -alpha + alpha * (dCddug * jt + alpha * Cd)

    return dfda, dfdrhog, dfdrhol, dfdug, dfdul

# def RelEquilLocalPipe_comp(ul, ug, rhol, rhog, ALPHA, BETA, D, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol):
#    """
#    Calcula o valor da relação de equilíbrio local adimensional.

#    Parâmetros:
#    ul: Velocidade adimensional do líquido em um ponto do pipe
#    ug: Velocidade adimensional do gás em um ponto do pipe
#    rhol: Densidade adimensional do líquido
#    rhog: Densidade adimensional do gás
#    ALPHA: Fração de vazio em um ponto do pipe
#    BETA: Ângulo de inclinação do pipe
#    AREA: Área seccional da tubulação do pipeline e riser
#    D: Diâmetro da tubulação
#    EPS: Rugosidade do tubo do pipeline
#    G: Aceleração da gravidade
#    MUL: Viscosidade dinâmica do líquido
#    MUG: Viscosidade dinâmica do gás
#    w_u: Escala de velocidade
#    w_rho: Escala de densidade
#    tol: Tolerância numérica

#    Retorna:
#    fun: Valor da relação de equilíbrio local adimensional
#    """
#    PID = (D * G) / (4 * (w_u ** 2))
#    gamma = voidFraction.alpha2gamma(ALPHA, tol)
#    gammai = math.sin(math.pi * gamma) / math.pi

#    # Número de Reynolds para o líquido
#    Rel = ((w_rho * w_u * D) / MUL) * rhol * abs((1 - ALPHA) * ul) / gamma

#    # Número de Reynolds para o gás
#    Reg = ((w_u * w_rho * D) / MUG) * rhog * abs(ALPHA * ug) / (1.0 - gamma + gammai)

#    # Perímetros molhados
#    Sl = math.pi * D * gamma
#    Sg = math.pi * D * (1.0 - gamma)
#    Si = math.pi * D * gammai

#    # Diâmetros hidráulicos
#    Dl = 4.0 * (1.0 - ALPHA) * AREA / Sl
#    Dg = 4.0 * ALPHA * AREA / (Sg + Si)

#    # Fatores de atrito
#    if Rel >= tol:
#        fl = voidFraction.ffan(EPS / Dl, Rel)
#    else:
#        fl = 0

#    if Reg >= tol:
#        fg = voidFraction.ffan(EPS / Dg, Reg)
#    else:
#        fg = 0

#    fi = 0.0142  # mesmo que utilizado em fungamma.m

#    # Cálculo de ui
#    if Rel < 2000:
#        ui = 1.8 * ul
#    elif Rel > 2200:
#        ui = ul
#    else:
#        ui = ul * (1.8 * (2200 - Rel) + (Rel - 2000)) / 200

#    # Cálculo da relação de equilíbrio local
#    auxa = fg * rhog * ug * abs(ug) * (1 - gamma) / ALPHA
#    auxb = rhol * fl * ul * abs(ul) * gamma / (1.0 - ALPHA)
#    auxc = fi * rhog * (ug - ui) * abs(ug - ui) * (gammai / (ALPHA * (1.0 - ALPHA)))
#    auxd = PID * (rhol - rhog) * math.sin(BETA)

#    fun = auxa / 2 - auxb / 2 + auxc / 2 + auxd

#    return fun


def DRelEquiLocal_pipe_comp(alpha,rhol,rhog,ul,ug,theta,D,AREA,EPS,G,MUL,MUG,sigma,w_u,w_rho,tol):
    #
    # derivada da relação de equilibrio local em relação a alph, rho_l, rho_g,
    # u_l e u_g
    #
    # ul: velocidade adimensional do líquido em um ponto do pipe;
    # ug: velocidade adimensional do gás em um ponto do pipe;
    # rhol: densidade adimensional do líquido;
    # rhog: densidade adimensional do gás;
    # alpha: fração de vazio em um ponto do pipe;
    # theta: angulo de inclinação do pipe;
    # AREA: area seccional da tubulação do pipeline e riser
    # D: diametro da tubulação
    # EPS: rugosidade do tubo do pipeline
    # G: aceleração da gravidade
    # MUL: viscosidade dinâmica do líquido
    # MUG: viscosidade dinâmica do gas
    # w_u: escala de velocidade;
    # w_rho: escala de densidade;
    # tol: numerical tolerance.
    #
    PID = (D*G)/(4*(w_u**2));
    
    gamma = voidFraction.alpha2gamma(alpha,tol);
    
    gammai = np.sin(np.pi*gamma)/np.pi;
    
    # # Reynolds para o líquido
    Rel = ((w_rho*w_u*D)/MUL)*rhol*abs((1-alpha)*ul)/gamma;
    
    # # Reynolds para o gás
    Reg = ((w_u*w_rho*D)/MUG)*rhog*abs(alpha*ug)/(1.0-gamma+gammai);
    
    # perimetros molhados
    
    Sl = np.pi*D*gamma;
    Sg = np.pi*D*(1.0-gamma);
    Si = np.pi*D*gammai;
    
    # Diâmetros hidraulicos
    
    Dl = 4.0*(1.0-alpha)*AREA/Sl;
    Dg = 4.0*alpha*AREA/(Sg+Si);
    
    # fatores de atrito
    
    fl = voidFraction.ffan(EPS/Dl,Rel);
    fg = voidFraction.ffan(EPS/Dg,Reg);
    fi = 0.0142;           # mesmo que utilizado em fungamma.m
    
    # calculo de ui
    if Rel < 2000:
      ui = 1.8*ul;
    elif Rel > 2200:
      ui = ul;
    else:
      ui = ul * (1.8*(2200-Rel) + (Rel-2000) )/200;
    
    # derivada das quantidades de interesse
    
    dgammada = 1.0/(np.cos(2*gamma*np.pi)-1.0);
    
    dgammaida = np.cos(gamma*np.pi)*dgammada;
    
    dDlda = -D/gamma-((1-alpha)*D*dgammada)/(gamma**2);
    
    dDgda = D/(1 - gamma + gammai) - ((alpha*D)/((1 - gamma + gammai)**2))*(-dgammada + dgammaida);
    
    # derivadas do fator de atrito para o líquido
    dfldrel,dfldepsl = voidFraction.Dffan(Rel,EPS/Dl);
    
    # derivadas do fator de atrito para o gas
    dfgdreg,dfgdepsg = voidFraction.Dffan(Reg,EPS/Dg);
    
    # derivadas do numero de Reynolds do liquido
    #
    # em relação a alpha
    t3 = (w_rho * w_u * D * rhol);
    t5 = (alpha - 1) * ul;
    #t6 = abs(1, t5);
    if t5 >= 0:
      t6 = 1.0;
    else:
      t6 = -1.0;

    t8 = 1 / MUL;
    t9 = gamma;
    t14 = abs(t5);
    t16 = (t9**2);
    t18 = dgammada;
    drelda = 1 / t9 * t8 * t6 * ul * t3 - t18 / t16 * t8 * t14 * t3;
    
    # em relação a rhol
    t5 = abs((alpha - 1) * ul);
    t8 = gamma;
    dreldrhol = 1 / t8 / MUL * t5 * w_rho * w_u * D;
    
    # em relação a ul
    t4 = alpha - 1;
    #t6 = abs(1, ul * t4);
    if ul*t4 >= 0:
      t6 = 1.0;
    else:
      t6 = -1.0;
    t9 = gamma;
    dreldul = 1 / t9 / MUL * t6 * t4 * w_rho * w_u * D * rhol;
    
    # derivadas do numero de Reynolds do gas 
    #
    # em relação a alpha
    t3 = (w_rho * w_u * D * rhog);
    t4 = (alpha * ug);
    #t5 = abs(1, t4);
    if t4 >= 0:
      t5 = 1.0;
    else:
      t5 = -1.0;
    t7 = 1 / MUG;
    t8 = gamma;
    t9 = gammai;
    t10 = 1 - t8 + t9;
    t15 = abs(t4);
    t17 = t10 ** 2;
    t19 = dgammada;
    t20 = dgammaida;
    dregda = 1 / t10 * t7 * t5 * ug * t3 - (-t19 + t20) / t17 * t7 * t15 * t3;
    
    # em relação a rhog
    t4 = abs(alpha * ug);
    t7 = gamma;
    t8 = gammai;
    dregdrhog = 1 / (1 - t7 + t8) / MUG * t4 * w_rho * w_u * D;
    
    # em relação a ug
    #t5 = abs(1, alpha * ug);
    if alpha*ug >= 0:
      t5 = 1.0;
    else:
      t5 = -1.0;
    t8 = gamma;
    t9 = gammai;
    dregdug = 1 / (1 - t8 + t9) / MUG * t5 * alpha * w_rho * w_u * D * rhog;
    
    # derivada de ui e, relação a ul
    if Rel < 2000:
      duidul = 1.8;
    elif Rel > 2200:
      duidul = 1.0;
    else:
      duidul = (1.8*(2200-Rel) + (Rel-2000) )/200;
    
    # derivada da relação de equilíbrio local em relação a alpha
    
    t1 = dregda;
    t4 = Dg;
    t5 = (t4 ** 2);
    t7 = dDgda;
    t13 = abs(ug);
    t14 = gamma;
    t16 = (1 - t14) * t13;
    t17 = 1 / alpha;
    t21 = Reg;
    t24 = fg;
    t26 = ug * t24 * rhog;
    t27 = dgammada;
    t32 = alpha ** 2;
    t33 = 1 / t32;
    t37 = drelda;
    t40 = Dl;
    t41 = (t40 ** 2);
    t43 = dDlda;
    t49 = abs(ul);
    t50 = t14 * t49;
    t51 = 1 - alpha;
    t52 = 2 * t51;
    t53 = 1 / t52;
    t56 = Rel;
    t59 = fl;
    t61 = ul * t59 * rhol;
    t65 = t52 ** 2;
    t71 = ui;
    t72 = ug - t71;
    t73 = t72 * fi * rhog;
    t74 = abs(t72);
    t75 = dgammaida;
    t77 = 1 / t51;
    t82 = gammai;
    t83 = t82 * t74;
    t88 = t51 ** 2;
    dfda = (t17 * t16 * ug * (t1 * dfgdreg - t7 / t5 * dfgdepsg * EPS) * rhog) / 0.2e1 - (t17 * t27 * t13 * t26) / 0.2e1 - (t33 * t16 * t26) / 0.2e1 - (t53 * t50 * ul * (t37 * dfldrel - t43 / t41 * dfldepsl * EPS) * rhol) - (t53 * t27 * t49 * t61) - (2 / t65 * t50 * t61) + (t77 * t17 * t75 * t74 * t73) / 0.2e1 - (t77 * t33 * t83 * t73) / 0.2e1 + (1 / t88 * t17 * t83 * t73) / 0.2e1;
    
    # derivada da relação de equilíbrio local em relação a densidade do gás rhog
    t1 = Reg;
    t2 = Dg;
    t5 = fg;
    t7 = abs(ug);
    t8 = gamma;
    t9 = 1 - t8;
    t11 = 1 / alpha;
    t16 = dregdrhog;
    t23 = ui;
    t24 = ug - t23;
    t26 = abs(t24);
    t28 = gammai;
    t35 = np.sin(theta);
    dfdrhog = (t11 * t9 * t7 * ug * t5) / 0.2e1 + (t11 * t9 * t7 * ug * t16 * dfgdreg * rhog) / 0.2e1 + (1 / (1 - alpha) * t11 * t28 * t26 * t24 * fi) / 0.2e1 - t35 * PID;
    
    # derivada da relação de equilíbrio local em relação a densidade do líquido rhol
    t1 = Rel;
    t2 = Dl;
    t5 = fl;
    t7 = abs(ul);
    t8 = gamma;
    t12 = 1 / (2 - 2 * alpha);
    t16 = dreldrhol;
    t22 = np.sin(theta);
    dfdrhol = -(t12 * t8 * t7 * ul * t16 * dfldrel * rhol) - (t12 * t8 * t7 * ul * t5) + t22 * PID;
    
    # derivada da relação de equilíbrio local em relação a ul
    t2 = dreldul;
    t4 = abs(ul);
    t6 = gamma;
    t7 = 1 - alpha;
    t9 = 0.1e1 / t7 / 0.2e1;
    t13 = Rel;
    t14 = Dl;
    t17 = fl;
    t18 = t17 * rhol;
    #t23 = abs(1, ul);
    if ul >= 0:
      t23 = 1;
    elif ul < 0:
      t23 = -1;
    t27 = (fi * rhog);
    t28 = duidul;
    t30 = ui;
    t31 = (ug - t30);
    t32 = abs(t31);
    t33 = gammai;
    t37 = 1 / t7 / alpha;
    #t43 = abs(1, t31);
    if t31 >= 0:
      t43 = 1;
    elif t31 < 0:
      t43 = -1;
    
    dfdul = -t9 * t6 * t4 * ul * t2 * dfldrel * rhol - t9 * t6 * t4 * t18 - t9 * t6 * t23 * ul * t18 - (t37 * t33 * t32 * t28 * t27) / 0.2e1 - (t37 * t33 * t43 * t28 * t31 * t27) / 0.2e1;
    
    # derivada da relação de equilíbrio local em relação a ug
    t2 = dregdug;
    t4 = abs(ug);
    t6 = gamma;
    t7 = 1 - t6;
    t8 = 1 / alpha;
    t12 = Reg;
    t13 = Dg;
    t16 = fg;
    t17 = (t16 * rhog);
    #t22 = abs(1, ug);
    if ug >= 0:
      t22 = 1.0;
    elif ug < 0:
      t22 = -1.0;
    t26 = (fi * rhog);
    t27 = ui;
    t28 = ug - t27;
    t29 = abs(t28);
    t31 = gammai;
    t34 = 1 / (1 - alpha);
    #t38 = abs(1, t28);
    if t28 >= 0:
      t38 = 1.0;
    elif t28 < 0:
      t38 = -1.0;

    dfdug = (t8 * t7 * t4 * ug * t2 * dfgdreg * rhog) / 0.2e1 + (t34 * t8 * t31 * t38 * t28 * t26) / 0.2e1 + (t8 * t7 * t22 * ug * t17) / 0.2e1 + (t34 * t8 * t31 * t29 * t26) / 0.2e1 + (t8 * t7 * t4 * t17) / 0.2e1;

    return dfda,dfdrhol,dfdrhog,dfdul,dfdug

def Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    # derivadas da lei de fechamento do sistema oleoduto-riser em relação às variáveis primitivas.
    # Relação de equilíbrio local para o oleoduto (theta <= 0) e relação de deslizamento para o riser (theta > 0)
    dfda = 0.0
    dfdrhol = 0.0
    dfdrhog = 0.0
    dfdul = 0.0
    dfdug = 0.0
    # para o oleoduto
    if theta <= 0:
        dfda, dfdrhol, dfdrhog, dfdul, dfdug = DRelEquiLocal_pipe_comp(alpha, rhol, rhog, ul, ug, -theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
    # para o riser
    elif theta > 0:
        dfda, dfdrhog, dfdrhol, dfdug, dfdul = Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    return dfda, dfdrhog, dfdrhol, dfdug, dfdul

def Dvar_primDvar_uj(n, alpha, rhol, rhog, ul, ug, Cl, Cg, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    dfda, dfdrhog, dfdrhol, dfdug, dfdul = Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    if n == 1:
        t1 = Cl ** 2
        t5 = Cg ** 2
        daduj = 1 / (t5 * (-1 + alpha) * rhog - t1 * rhol * alpha) * t1 * alpha

        t1 = Cl ** 2
        t3 = Cg ** 2
        dpduj = -1 / (t3 * (-1 + alpha) * rhog - t1 * rhol * alpha) * t3 * t1 * rhog

        t1 = Cg ** 2
        t2 = rhog * t1
        t3 = Cl ** 2
        t7 = (alpha * (-rhol * t3 + t2) - t2)
        t9 = dfdug
        t12 = dfda
        t14 = dfdrhol
        t16 = dfdrhog
        t27 = dfdul
        dulduj = 1 / t7 / (t9 * (-1 + alpha) * rhol + rhog * alpha * t27) * (t9 * ul * t7 + rhog * alpha * (-t12 * alpha * t3 + (t1 * t14 + t3 * t16) * rhog))

        t1 = Cg ** 2
        t2 = rhog * t1
        t3 = Cl ** 2
        t7 = (alpha * (-rhol * t3 + t2) - t2)
        t9 = dfdul
        t12 = dfda
        t14 = dfdrhol
        t16 = dfdrhog
        t21 = -1 + alpha
        t28 = dfdug
        dugduj = 1 / (rhog * alpha * t9 + t28 * t21 * rhol) / t7 * (-t9 * ul * t7 + rhol * t21 * (-t12 * alpha * t3 + (t14 * t1 + t16 * t3) * rhog))

    elif n == 2:
        t1 = -1 + alpha
        t2 = Cg ** 2
        t7 = Cl ** 2
        daduj = 1 / (-t7 * rhol * alpha + t2 * t1 * rhog) * t2 * t1

        t1 = Cl ** 2
        t2 = Cg ** 2
        dpduj = -1 / (t2 * (-1 + alpha) * rhog - t1 * rhol * alpha) * rhol * t2 * t1

        t1 = Cg ** 2
        t2 = rhog * t1
        t3 = Cl ** 2
        t7 = (alpha * (-rhol * t3 + t2) - t2)
        t9 = dfdug
        t11 = -1 + alpha
        t13 = dfda
        t15 = dfdrhol
        t17 = dfdrhog
        t29 = dfdul
        dulduj = 1 / (rhog * alpha * t29 + t9 * t11 * rhol) / t7 * (t9 * ug * t7 - alpha * (t13 * t11 * t1 - (t15 * t1 + t17 * t3) * rhol) * rhog)

        t1 = -1 + alpha
        t3 = Cg ** 2
        t6 = Cl ** 2
        t8 = -t6 * rhol * alpha + t3 * t1 * rhog
        t10 = dfdul
        t13 = dfda
        t15 = dfdrhol
        t17 = dfdrhog
        t28 = dfdug
        dugduj = 1 / (rhog * alpha * t10 + t28 * t1 * rhol) / t8 * (-t10 * ug * t8 - rhol * t1 * (t13 * t1 * t3 - (t15 * t3 + t17 * t6) * rhol))

    elif n == 3:
        daduj = 0
        dpduj = 0
        t1 = dfdug
        t2 = dfdul
        dulduj = 1 / (-rhog * alpha * t2 + rhol * (1 - alpha) * t1) * t1

        t1 = dfdul
        t4 = dfdug
        dugduj = -1 / (-rhog * alpha * t1 + rhol * (1 - alpha) * t4) * t1

    return daduj, dpduj, dulduj, dugduj


def lei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    # Lei de fechamento para sistema-oleoduto riser. Relação de equilíbrio local
    # para o oleoduto (theta <=0 ) e relação de deslizamento para o riser (theta > 0)

    # ul: velocidade adimensional do líquido em um ponto do pipe;
    # ug: velocidade adimensional do gás em um ponto do pipe;
    # rhol: densidade adimensional do líquido;
    # rhog: densidade adimensional do gás;
    # alpha: fração de vazio em um ponto do pipe;
    # theta: angulo de inclinação do pipe;
    # AREA: area seccional da tubulação do pipeline e riser
    # DH: diametro hidraulico da tubulação
    # EPS: rugosidade do tubo do pipeline
    # G: aceleração da gravidade
    # MUL: viscosidade dinâmica do líquido
    # MUG: viscosidade dinâmica do gas
    # w_u: escala de velocidade;
    # w_rho: escala de densidade;
    # tol: numerical tolerance.

    # para o oleoduto
    if theta <= 0:
        value = voidFraction.RelEquilLocalPipe_comp(ul, ug, rhol, rhog, alpha, -theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    # para o riser
    elif theta > 0:
        value = drift_flux_swananda_ndim(ul, ug, rhol, rhog, alpha, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    return value