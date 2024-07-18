# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:00:22 2023

@author: leonardoab
"""

import math
import numpy as np

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
    
    A_min = -np.sqrt(x**2 + z**2)
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


# This script tests the code to obtain the catenary constant. Use data from
# Wordsworth e al 1998.
#
# x - horizontal coordinate of the top of the catenary
# z - vertical coordinate of the top of the catenary
# tol - tolerance

# x = 1.5 # meters
# z = 1.5 # meters
# tol = 0.0001 #input('tolerance to be used = ');

# # perform test

# A = catenary_constant(x,z,tol)
# value = catenaryf(x,z,A)
# LR = A*math.sinh(x/A)


# n = 1000 #input(' number of points = ');
# dx = x/(n-1)
# ds = LR/(n-1)

# # for i=1:1:n
# #    xv(i) = (i-1)*dx;
# #    zv(i) = A*(cosh(xv(i)/A)-1.0);
# # end
# # %
# # plot(xv,zv);
# # pause
# # %
# VPARAM = np.array([A,LR])

# sv = np.zeros([n])
# vtheta = np.zeros([n])
# vdtheta = np.zeros([n])
# vdthetaa = np.zeros([n])
# # %
# for i in np.arange(0,n):
#   sv[i] = (i-1)*ds
#   vtheta[i] = fungeo(sv[i],VPARAM)
#   vdtheta[i] = Dfungeo(sv[i],VPARAM)
#   vdthetaa[i] = (LR/A)*(math.cos(vtheta[i])**2)

# plot(sv,vtheta);
# plot(sv,vdtheta,'-',sv,vdthetaa,'*');