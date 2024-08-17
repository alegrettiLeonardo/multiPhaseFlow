import numpy as np
import math 
import voidFraction


def CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
   # Velocidades superficiais
   jl = ul * (1 - alpha)
   jg = ug * alpha
   jt = jl + jg

   # Número de Froude
   Fr = math.sqrt(rhog / (rhol - rhog)) * (jg * w_u) / math.sqrt(G * DH * math.cos(theta))

   # Número de Reynolds Retp
   Retp = jt * rhol * w_rho * w_u * DH / MUL

   beta = jg / jt

   # Título
   x = (jg * rhog) / (jg * rhog + jl * rhol)

   # Número de Laplace
   La = math.sqrt(sigma / (G * w_rho * (rhol - rhog))) / DH

   # Cálculo do fator de atrito de Fanning - Colebrook (1939)
   df = 1
   ftp = 1
   while df > tol:
       b = -4 * math.log10(((EPS / DH) / 3.7) + (1.256 / (Retp * math.sqrt(ftp))))
       f_new = 1 / (b ** 2)
       df = abs(f_new - ftp)
       ftp = f_new

   # Avaliação dos parâmetros C1, C2, C3 e C4 da velocidade de deslizamento
   C1 = 0.2  # Constante igual a 0.2 para tubo redondo

   # C2
   if MUL / 0.001 > 10:
       C2 = (0.434 / (math.log10(MUL / 0.001))) ** 0.15
   else:
       C2 = 1

   # C3
   if La < 0.025:
       C3 = (La / 0.025) ** 0.9
   else:
       C3 = 1

   # C4
   if (Fr <= 0.1) and (theta < 0) and (theta >= -50 * math.pi / 180):
       C4 = -1
   else:
       C4 = 1

   # Coeficiente C01 para o parâmetro de distribuição C0
   if (Fr <= 0.1) and (theta < 0) and (theta > -50 * math.pi / 180):
       C01 = 0
   else:
       C01 = (C1 - C1 * math.sqrt(rhog / rhol)) * (((2.6 - beta) ** 0.15) - math.sqrt(ftp)) * ((1 - x) ** 1.5)

   # Parâmetro de distribuição
   aux = 1 + (Retp / 1000) ** 2
   C0a = 2 - ((rhog / rhol) ** 2)
   C0a = C0a / aux

   aux = 1 + (1000 / Retp) ** 2
   C0b = (1 + ((rhog / rhol) ** 2) * math.cos(theta)) / (1 + math.cos(theta))
   C0b = math.sqrt(C0b)
   C0b = C0b ** (1 - alpha)
   C0b = C0b ** (2 / 5)
   C0b = (C0b + C01) / aux

   Cd = C0a + C0b

   # Velocidade de deslizamento
   Ud = (1 - alpha) ** 0.5
   Ud = Ud * math.sqrt((rhol - rhog) * DH * G / rhol)
   Ud = Ud * (0.35 * math.sin(theta) + 0.45 * math.cos(theta)) * C2 * C3 * C4

   return Cd, Ud


def DCdUd_swananda(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    derivada do parametro de distribuição C0 e da velocidade de deslizamento Ud
    em relação a alph, rho_l, rho_g, u_l e u_g
    
    ul: velocidade adimensional do líquido em um ponto do pipe;
    ug: velocidade adimensional do gás em um ponto do pipe;
    rhol: densidade adimensional do líquido;
    rhog: densidade adimensional do gás;
    alpha: fração de vazio em um ponto do pipe;
    theta: angulo de inclinação do pipe;
    AREA: area seccional da tubulação do pipeline e riser
    DH: diametro hidraulico da tubulação
    EPS: rugosidade do tubo do pipeline
    G: aceleração da gravidade
    MUL: viscosidade dinâmica do líquido
    MUG: viscosidade dinâmica do gas
    w_u: escala de velocidade;
    w_rho: escala de densidade;
    tol: numerical tolerance.
    
    velocidades superficiais
    """
    jl = ul*(1-alpha)*w_u;
    jg = ug*alpha*w_u;
    jt = jl+jg;
    
    # numero de Froude
    Fr = math.sqrt(rhog/(rhol-rhog))*jg/math.sqrt(G*DH*math.cos(theta));
    
    # numero de Reynolds Retp
    Retp = jt*rhol*w_rho*DH/MUL;
    
    beta = jg/jt;
    
    # titulo
    x = (jg*rhog)/(jg*rhog+jl*rhol);
    
    # número de Laplace
    La = math.sqrt(sigma/(G*w_rho*(rhol-rhog)))/DH;
    
    # cálculo do fator de atrito de fanning - Colebrook (1939)
    df = 1;
    ftp = 1;
    while df > tol:
        
        b=-4*np.log10(((EPS/DH)/3.7)+(1.256/(Retp*np.sqrt(ftp))));
        f_new = 1 / (b**2);
        df = abs(f_new-ftp);
        ftp = f_new;
    
    # avaliação dos parâmetro C1, C2, C3 e C4 da velocidade de deslizamento
    C1 = 0.2  # constante igual a 0.2 para tubo redondo  
    
    # C2
    if MUL/0.001 > 10:
        C2=(0.434/(math.log10(MUL/0.001)))**0.15
    else:
        C2=1;
    
    # C3
    if La < 0.025:
        C3=(La/0.025)**0.9
    else:
        C3=1
    
    # C4
    if (Fr <= 0.1) and (theta < 0) and (theta >= -50*math.pi/180):
        C4=-1
    else:    
        C4=1
    
    # coeficient C01 para o parametro de distribuição C0
    if (Fr <= 0.1) and (theta < 0) and (theta > -50*math.pi/180):
        C01 = 0
    else:   
        C01 = (C1-C1*math.sqrt(rhog/rhol))*(((2.6-beta)**0.15)-math.sqrt(ftp))*((1-x)**1.5)
    
    # derivadas das quantidades de interesse
    #
    # derivadas do titulo
    #
    # em relação a alpha
    t1 = (rhog * ug);
    t2 = (alpha * rhog);
    t7 = ug * t2 + rhol * (1 - alpha) * ul;
    t10 = t7 ** 2;
    dxda = 1 / t7 * t1 - (-rhol * ul + t1) / t10 * ug * t2;
    
    # em relação a rhog
    t7 = rhog * alpha * ug + rhol * (1 - alpha) * ul;
    t10 = alpha ** 2;
    t12 = ug ** 2;
    t13 = t7 ** 2;
    dxdrhog = 1 / t7 * ug * alpha - 1 / t13 * t12 * t10 * rhog;
    #dxdrhog = dxdrhog/w_rho;
    
    # em relação a rhol
    t2 = (rhog * alpha * ug);
    t3 = 1 - alpha;
    t7 = (ul * t3 * rhol + t2) ** 2;
    dxdrhol = -ul * t3 / t7 * t2;
    #dxdrhol = dxdrhol/w_rho;
    
    # em relação a ul
    t2 = (rhog * alpha * ug);
    t3 = 1 - alpha;
    t7 = (ul * t3 * rhol + t2) ** 2;
    dxdul = -t3 * rhol / t7 * t2;
    #dxdul = dxdul/w_u;
    
    # em relação a ug
    t1 = (alpha * rhog);
    t6 = ug * t1 + rhol * (1 - alpha) * ul;
    t9 = (rhog ** 2);
    t10 = alpha ** 2;
    t12 = t6 ** 2;
    dxdug = 1 / t6 * t1 - 1 / t12 * ug * t10 * t9;
    
    # derivadas do Reynolds 
    # em relação a alpha
    dRetpda = rhol * w_u * w_rho * (ug - ul) * DH / MUL;
    
    # em relação a rhol
    dRetpdrhol = w_u * w_rho * (alpha * ug + (1 - alpha) * ul) * DH / MUL;
    
    # em relação a ul
    dRetpdul = rhol * w_u * w_rho * (1 - alpha) * DH / MUL;
    
    # em relação a ug
    dRetpdug = rhol * w_u * w_rho * alpha * DH / MUL;
    
    # derivada do fator de atrito de fanning ftp em relação a Retp
    t4 = ftp;
    t5 = np.sqrt(t4);
    dftpdRetp = -0.2009600000e11 * t5 * t4 / (0.1244640591e10 * t5 * Retp * EPS + 0.1004800000e11 * t5 * DH + 0.5784093754e10 * DH) * DH / Retp;
    
    # derivadas de C01
    if (Fr <= 0.1) and (theta < 0) and (theta > -50*math.pi/180):
        dC01da = 0;
        dC01drhog = 0;
        dC01drhol = 0;
        dC01dul = 0;
        dC01dug = 0;
    else:   
        # em relação a alpha
        t3 = math.sqrt(rhog / rhol);
        t5 = -t3 * C1 + C1;
        t6 = (alpha * ug);
        t9 = t6 + (1 - alpha) * ul;
        t10 = 1 / t9;
        t12 = 0.26e1 - (t10 * t6);
        t13 = t12 ** (-0.85e0);
        t15 = t9 ** 2;
        t23 = Retp;
        t26 = ftp;
        t27 = math.sqrt(t26);
        t30 = dRetpda;
        t35 = x;
        t36 = 1 - t35;
        t37 = t36 ** 0.15e1;
        t39 = t12 ** 0.15e0;
        t42 = t36 ** 0.5e0;
        t43 = dxda;
        dC01da = t37 * (0.15e0 * (-t10 * ug + (ug - ul) / t15 * t6) * t13 - t30 * dftpdRetp / t27 / 0.2e1) * t5 - 0.15e1 * t43 * t42 * (t39 - t27) * t5;
        
        # em relação a rhog
        t1 = 0.1e1 / rhol;
        t3 = math.sqrt(t1 * rhog);
        t6 = (alpha * ug);
        t13 = (0.26e1 - (1 / (t6 + (1 - alpha) * ul) * t6)) ** 0.15e0;
        t14 = Retp;
        t17 = ftp;
        t18 = math.sqrt(t17);
        t19 = t13 - t18;
        t21 = x;
        t22 = 1 - t21;
        t23 = t22 ** 0.15e1;
        t30 = t22 ** 0.5e0;
        t31 = dxdrhog;
        dC01drhog = -t23 * t19 * t1 / t3 * C1 / 0.2e1 - 0.15e1 * t31 * t30 * t19 * (-t3 * C1+C1);
        
        # em relação a rhol  
        t3 = math.sqrt(rhog / rhol);
        t7 = rhol ** 2;
        t9 = (alpha * ug);
        t16 = (0.26e1 - (1 / (t9 + (1 - alpha) * ul) * t9)) ** 0.15e0;
        t17 = Retp;
        t20 = ftp;
        t21 = math.sqrt(t20);
        t22 = t16 - t21;
        t24 = x;
        t25 = 1 - t24;
        t26 = t25 ** 0.15e1;
        t31 = -t3 * C1 + C1;
        t34 = dRetpdrhol;
        t40 = t25 ** 0.5e0;
        t41 = dxdrhol;
        dC01drhol = t26 * t22 / t7 * rhog / t3 * C1 / 0.2e1 - t26 * t34 * dftpdRetp / t21 * t31 / 0.2e1 - 0.15e1 * t41 * t40 * t22 * t31;
        
        # em relação a ul
        t3 = math.sqrt(rhog / rhol);
        t5 = -t3 * C1 + C1;
        t6 = (alpha * ug);
        t7 = 1 - alpha;
        t9 = ul * t7 + t6;
        t12 = 0.26e1 - (1 / t9 * t6);
        t13 = t12 ** (-0.85e0);
        t15 = t9 ** 2;
        t21 = Retp;
        t24 = ftp;
        t25 = math.sqrt(t24);
        t28 = dRetpdul;
        t33 = x;
        t34 = 1 - t33;
        t35 = t34 ** 0.15e1;
        t37 = t12 ** 0.15e0;
        t40 = t34 ** 0.5e0;
        t41 = dxdul;
        dC01dul = t35 * (0.15e0 * t7 / t15 * ug * alpha * t13 - t28 * dftpdRetp / t25 / 0.2e1) * t5 - 0.15e1 * t41 * t40 * (t37 - t25) * t5;
        
        # em relação a ug
        t3 = math.sqrt(rhog / rhol);
        t5 = -t3 * C1 + C1;
        t6 = (alpha * ug);
        t9 = t6 + (1 - alpha) * ul;
        t10 = 1 / t9;
        t12 = 0.26e1 - (t10 * t6);
        t13 = t12 ** (-0.85e0);
        t15 = alpha ** 2;
        t17 = t9 ** 2;
        t23 = Retp;
        t26 = ftp;
        t27 = math.sqrt(t26);
        t30 = dRetpdug;
        t35 = x;
        t36 = 1 - t35;
        t37 = t36 ** 0.15e1;
        t39 = t12 ** 0.15e0;
        t42 = t36 ** 0.5e0;
        t43 = dxdug;
        dC01dug = t37 * (0.15e0 * (-t10 * alpha + 1 / t17 * ug * t15) * t13 - t30 * dftpdRetp / t27 / 0.2e1) * t5 - 0.15e1 * t43 * t42 * (t39 - t27) * t5;
        
    # derivadas de C0
    # em relação a alpha
    t1 = rhog ** 2;
    t2 = rhol ** 2;
    t4 = 0.1e1 / t2 * t1;
    t6 = Retp;
    t7 = t6 ** 2;
    t10 = (0.1e1 + t7 / 0.1000000e7) ** 2;
    t13 = dRetpda;
    t17 = math.cos(theta);
    t22 = 0.1e1 / (0.1e1 + t17) * (t17 * t4 + 0.1e1);
    t23 = math.sqrt(t22);
    t25 = t23 ** (1 - alpha);
    t26 = t25 ** (0.1e1 / 0.5e1);
    t27 = t26 ** 2;
    t28 = math.log(t22);
    t31 = dC01da;
    t35 = 0.1e1 + 0.1000000e7 / t7;
    t38 = C01;
    t40 = t35 ** 2;
    dC0da = -t13 * t6 / t10 * (0.2e1 - t4) / 0.500000e6 + 0.1e1 / t35 * (-t28 * t27 / 0.5e1 + t31) + 0.2000000e7 * t13 / t7 / t6 / t40 * (t27 + t38);
    
    # em relação a rhog
    t1 = rhol ** 2;
    t2 = 0.1e1 / t1;
    t3 = t2 * rhog;
    t4 = Retp;
    t5 = t4 ** 2;
    t11 = rhog ** 2;
    t13 = math.cos(theta);
    t15 = t13 * t2 * t11 + 0.1e1;
    t19 = math.sqrt(0.1e1 / (0.1e1 + t13) * t15);
    t20 = 1 - alpha;
    t21 = t19 ** t20;
    t22 = t21 ** (0.1e1 / 0.5e1);
    t23 = t22 ** 2;
    t30 = dC01drhog;
    dC0drhog = -0.2e1 / (0.1e1 + t5 / 0.1000000e7) * t3 + 0.1e1 / (0.1e1 + 0.1000000e7 / t5) * (0.2e1 / 0.5e1 * t13 * t3 / t15 * t20 * t23 + t30);
    
    # em relação a rhol
    t1 = rhog ** 2;
    t2 = rhol ** 2;
    t5 = 0.1e1 / t2 / rhol * t1;
    t6 = Retp;
    t7 = t6 ** 2;
    t9 = 0.1e1 + t7 / 0.1e7;
    t14 = 0.1e1 / t2 * t1;
    t16 = t9 ** 2;
    t19 = dRetpdrhol;
    t23 = math.cos(theta);
    t25 = t23 * t14 + 0.1e1;
    t29 = math.sqrt(0.1e1 / (0.1e1 + t23) * t25);
    t30 = 1 - alpha;
    t31 = t29 ** t30;
    t32 = t31 ** (0.1e1 / 0.5e1);
    t33 = t32 ** 2;
    t40 = dC01drhol;
    t44 = 0.1e1 + 0.1000000e7 / t7;
    t47 = C01;
    t49 = t44 ** 2;
    dC0drhol = 0.2e1 / t9 * t5 - t19 * t6 / t16 * (0.2e1 - t14) / 0.5e6 + 0.1e1 / t44 * (-0.2e1 / 0.5e1 * t23 * t5 / t25 * t30 * t33 + t40) + 0.2e7 * t19 / t7 / t6 / t49 * (t33 + t47);
    
    # em relação a ul
    t1 = rhog ** 2;
    t2 = rhol ** 2;
    t4 = 0.1e1 / t2 * t1;
    t6 = Retp;
    t7 = t6 ** 2;
    t10 = (0.1e1 + t7 / 0.1e7) ** 2;
    t13 = dRetpdul;
    t17 = dC01dul;
    t20 = 0.1e1 + 0.1e7 / t7;
    t23 = math.cos(theta);
    t29 = math.sqrt(0.1e1 / (0.1e1 + t23) * (t23 * t4 + 0.1e1));
    t31 = t29 ** (1 - alpha);
    t32 = t31 ** (0.1e1 / 0.5e1);
    t33 = t32 ** 2;
    t34 = C01;
    t36 = t20 ** 2;
    dC0dul = -t13 * t6 / t10 * (0.2e1 - t4) / 0.5e6 + 0.1e1 / t20 * t17 + 0.2e7 * t13 / t7 / t6 / t36 * (t33 + t34);
    
    # em relação a ug
    t1 = rhog ** 2;
    t2 = rhol ** 2;
    t4 = 0.1e1 / t2 * t1;
    t6 = Retp;
    t7 = t6 ** 2;
    t10 = (0.1e1 + t7 / 0.1e7) ** 2;
    t13 = dRetpdug;
    t17 = dC01dug;
    t20 = 0.1e1 + 0.1e7 / t7;
    t23 = math.cos(theta);
    t29 = math.sqrt(0.1e1 / (0.1e1 + t23) * (t23 * t4 + 0.1e1));
    t31 = t29 ** (1 - alpha);
    t32 = t31 ** (0.1e1 / 0.5e1);
    t33 = t32 ** 2;
    t34 = C01;
    t36 = t20 ** 2;
    dC0dug = -t13 * t6 / t10 * (0.2e1 - t4) / 0.5e6 + 0.1e1 / t20 * t17 + 0.2e7 * t13 / t7 / t6 / t36 * (t33 + t34);
    
    # derivadas de C3
    if La < 0.025:
        # em relação a rhog
        t2 = sigma / w_rho;
        t3 = rhol - rhog;
        t5 = 0.1e1 / G;
        t8 = math.sqrt(t5 / t3 * t2);
        t9 = 0.1e1 / DH;
        t11 = (t9 * t8) ** (-0.1e0);
        t15 = t3 ** 2;
        dC3drhog = 0.1244705206e2 * t5 / t15 * t2 * t9 / t8 * t11;
        
        # em relação a rhol
        t2 = sigma / w_rho;
        t3 = rhol - rhog;
        t5 = 0.1e1 / G;
        t8 = math.sqrt(t5 / t3 * t2);
        t9 = 0.1e1 / DH;
        t11 = (t9 * t8) ** (-0.1e0);
        t15 = t3 ** 2;
        dC3drhol = -0.1244705206e2 * t5 / t15 * t2 * t9 / t8 * t11;
    else:
      dC3drhog = 0;
      dC3drhol = 0;

    # derivadas de Ud 
    #
    # em relação a alpha
    #
    t1 = math.sin(theta);
    t3 = math.cos(theta);
    t11 = np.sqrt((rhol - rhog) * DH * G / rhol);
    t14 = (1 - alpha) ** (-0.5e0)
    t16 = C3
    dUdda = -0.5e0 * C4 * t16 * C2 * t14 * t11 * (0.35e0 * t1 + 0.45e0 * t3);
    
    # em relação a rhog
    t1 = math.sin(theta)
    t3 = math.cos(theta)
    t5 = 0.35e0 * t1 + 0.45e0 * t3;
    t8 = 0.1e1 / rhol;
    t11 = math.sqrt(t8 * G * (rhol - rhog) * DH);
    t15 = (1 - alpha) ** 0.5e0;
    t18 = C3;
    t27 = dC3drhog;
    dUddrhog = -t8 * G * DH * C4 * t18 * C2 * t15 / t11 * t5 / 0.2e1 + C4 * t27 * C2 * t15 * t11 * t5;
    
    # em relação a rhol
    t1 = math.sin(theta);
    t3 = math.cos(theta);
    t5 = 0.35e0 * t1 + 0.45e0 * t3;
    t7 = (rhol - rhog) * DH;
    t8 = 0.1e1 / rhol;
    t11 = math.sqrt(t8 * G * t7);
    t15 = (1 - alpha) ** 0.5e0;
    t17 = C3;
    t21 = rhol ** 2;
    t32 = dC3drhol;
    dUddrhol = (t8 * G * DH - 0.1e1 / t21 * G * t7) * C4 * t17 * C2 * t15 / t11 * t5 / 0.2e1 + C4 * t32 * C2 * t15 * t11 * t5;
    
    # em relação a ul
    dUddul = 0;
    
    # em relação a ug
    dUddug = 0;
    return dC0da,dC0drhog,dC0drhol,dC0dug,dC0dul,dUdda,dUddrhog,dUddrhol,dUddul,dUddug

def drift_flux_swananda_ndim(ul, ug, rhol, rhog, alpha, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
   """
   Avalia a relação de deslizamento com parâmetros Cd e Ud dados pela correlação
   de swanada na forma adimensional.

   ul: velocidade adimensional do líquido em um ponto do pipe;
   ug: velocidade adimensional do gás em um ponto do pipe;
   rhol: densidade adimensional do líquido;
   rhog: densidade adimensional do gás;
   alpha: fração de vazio em um ponto do pipe;
   theta: ângulo de inclinação do pipe (positivo upward e negativo downward);
   AREA: área seccional da tubulação do pipeline e riser;
   D: diâmetro da tubulação;
   EPS: rugosidade do tubo do pipeline;
   G: aceleração da gravidade;
   MUL: viscosidade dinâmica do líquido;
   MUG: viscosidade dinâmica do gás;
   sigma: tensão superficial líquido-gás;
   w_u: escala de velocidade;
   w_rho: escala de densidade;
   tol: tolerância numérica.
   """
   Cd, Ud = CdUd_swananda(alpha, rhol, rhog, ul, ug, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

   # Relação de deriva
   njl = ul * (1 - alpha)
   njg = ug * alpha
   nUd = Ud / w_u

   value = -njg + alpha * (Cd * (njl + njg) + nUd)
   return value


def Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    derivada da relação de deslizamento adimensional em relação a alpha, rhog, rhol, ul e ug.
    """
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

def DRelEquiLocal_pipe_comp(alpha, rhol, rhog, ul, ug, theta, D, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    derivada da relação de equilibrio local em relação a alph, rho_l, rho_g,
    u_l e u_g
    
    ul: velocidade adimensional do líquido em um ponto do pipe;
    ug: velocidade adimensional do gás em um ponto do pipe;
    rhol: densidade adimensional do líquido;
    rhog: densidade adimensional do gás;
    alpha: fração de vazio em um ponto do pipe;
    theta: angulo de inclinação do pipe;
    AREA: area seccional da tubulação do pipeline e riser
    D: diametro da tubulação
    EPS: rugosidade do tubo do pipeline
    G: aceleração da gravidade
    MUL: viscosidade dinâmica do líquido
    MUG: viscosidade dinâmica do gas
    w_u: escala de velocidade;
    w_rho: escala de densidade;
    tol: numerical tolerance.
    
    """
    PID = (D*G)/(4*(w_u**2));
    gamma = voidFraction.alpha2gamma(alpha,tol);
    gammai = np.sin(np.pi*gamma)/np.pi;
    
    # Reynolds para o líquido
    Rel = ((w_rho*w_u*D)/MUL)*rhol*abs((1-alpha)*ul)/gamma;
    
    # Reynolds para o gás
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
        
    #
    # derivada das quantidades de interesse
    #
    dgammada = 1.0/(np.cos(2*gamma*np.pi)-1.0);
    dgammaida = np.cos(gamma*np.pi)*dgammada;
    
    dDlda = -D/gamma-((1-alpha)*D*dgammada)/(gamma**2);
    dDgda = D/(1-gamma+gammai)-((alpha*D)/((1-gamma+gammai)**2))*(-dgammada+dgammaida)
    
    # derivadas do fator de atrito para o líquido
    dfldrel,dfldepsl = voidFraction.Dffan(Rel,EPS/Dl)
    
    # derivadas do fator de atrito para o gas
    dfgdreg,dfgdepsg = voidFraction.Dffan(Reg,EPS/Dg)
    
    # derivadas do numero de Reynolds do liquido 
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
    t16 = (t9 ** 2);
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
    #
    # em relação a rhog
    #
    t4 = abs(alpha * ug);
    t7 = gamma;
    t8 = gammai;
    dregdrhog = 1 / (1 - t7 + t8) / MUG * t4 * w_rho * w_u * D;
    #
    # em relação a ug
    #
    #t5 = abs(1, alpha * ug);
    if alpha*ug >= 0:
        t5 = 1.0
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
    """
    Lei de fechamento para sistema-oleoduto riser. Relação de equilíbrio local
    para o oleoduto (theta <=0 ) e relação de deslizamento para o riser (theta > 0)

    ul: velocidade adimensional do líquido em um ponto do pipe;
    ug: velocidade adimensional do gás em um ponto do pipe;
    rhol: densidade adimensional do líquido;
    rhog: densidade adimensional do gás;
    alpha: fração de vazio em um ponto do pipe;
    theta: angulo de inclinação do pipe;
    AREA: area seccional da tubulação do pipeline e riser
    DH: diametro hidraulico da tubulação
    EPS: rugosidade do tubo do pipeline
    G: aceleração da gravidade
    MUL: viscosidade dinâmica do líquido
    MUG: viscosidade dinâmica do gas
    w_u: escala de velocidade;
    w_rho: escala de densidade;
    tol: numerical tolerance.
    """
    # para o oleoduto
    if theta <= 0:
        value = RelEquilLocalPipe_comp(ul, ug, rhol, rhog, alpha, theta, DH, AREA, EPS, G, MUL, MUG, w_u, w_rho, tol)
    # para o riser
    elif theta > 0:
        value = drift_flux_swananda_ndim(ul, ug, rhol, rhog, alpha, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)

    return value

def Dlei_fechamento_or_ndim_simp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol):
    """
    derivadas da lei de fechamento do sistema oleoduto-riser em relação às variáveis primitivas.
    Relação de equilíbrio local para o oleoduto (theta <= 0) e relação de deslizamento para o riser (theta > 0)
    """
    # para o oleoduto
    if theta <= 0:
        dfda, dfdrhol, dfdrhog, dfdul, dfdug = DRelEquiLocal_pipe_comp(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
    # para o riser
    elif theta > 0:
        dfda, dfdrhog, dfdrhol, dfdug, dfdul = Ddrift_flux_swananda_ndim(alpha, rhol, rhog, ul, ug, theta, DH, AREA, EPS, G, MUL, MUG, sigma, w_u, w_rho, tol)
    
    return dfda, dfdrhog, dfdrhol, dfdug, dfdul