import numpy as np
from src.atomicorbit.orbital_maths.generell_functions.normalizsation_factor import normalization_factor
# Konstanten
a0 = 0.529177210903e-10  # Bohr-Radius in Metern
def R_nl(n, l, r, Z=1):
    """
    Allgemeine radiale Wellenfunktion für Wasserstoff-ähnliche Orbitale.
    Parameter:
    n : int
        Hauptquantenzahl (1 <= n <= 4)
    l : int
        Nebenquantenzahl (Drehimpulsquantenzahl, 0 <= l < n)
    r : float oder numpy.ndarray
        Radialer Abstand vom Kern
    Z : float
        Effektive Kernladung
    Rückgabe:
    R : float oder numpy.ndarray
        Wert der radialen Wellenfunktion
    """
    if n < 1 or n > 4:
        raise ValueError("n muss zwischen 1 und 4 liegen.")
    if l < 0 or l >= n:
        raise ValueError("l muss zwischen 0 und n-1 liegen.")
    rho = (2 * Z * r) / (n * a0)
    N = normalization_factor(n, l, Z)
    # Laguerre-Polynome für verschiedene n und l
    if n == 1:
        L = 1
    elif n == 2:
        if l == 0:
            L = 1 - rho/2
        else:  # l == 1
            L = 1
    elif n == 3:
        if l == 0:
            L = 1 - 2 * rho/3 + 2 * rho * 2/27
        elif l == 1:
            L = 1 - rho/6
        else:  # l == 2
            L = 1
    elif n == 4:
        if l == 0:
            L = 1 - 3*rho/4 + rho*2/8 - rho* 3/192
        elif l == 1:
            L = 1 - rho/4 + rho* 2/80
        elif l == 2:
            L = 1 - rho/12
        else:  # l == 3
            L = 1
    else:
        raise NotImplementedError("Diese Funktion ist nur für n=3 implementiert.")
    return N * (rho**l) * np.exp(-rho/2) * L