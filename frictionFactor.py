# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:20:08 2024

@author: leonardoab
"""

import math

def FatorAtrito(REYNOLDS, EDIA):
    """
    Avalia o fator de atrito em função do número de Reynolds e da rugosidade
    do tubo para a fase líquida ou para a fase gasosa.

    Parameters:
        REYNOLDS: número de Reynolds
        EDIA: razão entre a rugosidade e o diâmetro hidráulico da fase líquida
              ou da fase gasosa.

    Returns:
        FATRITO: fator de atrito
    """

    if REYNOLDS == 0:
        FATRITO = 0
        return FATRITO
    elif REYNOLDS > 2300:
        FATRITO = 5.8506 / (REYNOLDS**0.8981)
        FATRITO = FATRITO + (EDIA**1.1098) / 2.8257
        FATRITO = (EDIA/3.7065) - (5.0452/REYNOLDS) * math.log10(FATRITO)
        FATRITO = -4.0 * math.log10(FATRITO)
        FATRITO = FATRITO**(-2)
        return FATRITO
    elif REYNOLDS < 2000:
        FATRITO = 16 / REYNOLDS
    else:
        # Interpolação do fator de atrito para 2000 < REYNOLDS < 2300
        Re = 2300
        FATRITO = 5.8506 / (Re**0.8981)
        FATRITO = FATRITO + (EDIA**1.1098) / 2.8257
        FATRITO = (EDIA/3.7065) - (5.0452/Re) * math.log10(FATRITO)
        FATRITO = -4.0 * math.log10(FATRITO)
        F2300 = FATRITO**(-2)

        F2000 = 8.0 * 10**(-3)
        FATRITO = F2000 + ((F2300 - F2000) / 300) * (REYNOLDS - 2000)

    return FATRITO