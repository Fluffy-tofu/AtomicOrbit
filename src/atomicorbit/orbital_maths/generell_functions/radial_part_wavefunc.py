import numpy as np
from atomicorbit.orbital_maths.generell_functions.normalizsation_factor import normalization_factor

# Constants
a0 = 0.529177210903e-10  # Bohr radius in meters


def generalized_laguerre(n, l, x):
    """
    Calculate the generalized Laguerre polynomial L^(2l+1)_(n-l-1)(x)
    """
    if n == l + 1:
        return 1

    if l == 0:
        Lm1 = 0  # L_{-1}
        L0 = 1  # L_0
        L1 = 1 - x  # L_1

        if n - l - 1 <= 1:
            return L1 if n - l - 1 == 1 else L0

        for i in range(2, n - l):
            L2 = ((2 * i - 1 - x) * L1 - (i - 1) * L0) / i
            L0 = L1
            L1 = L2
        return L1
    else:
        alpha = 2 * l + 1
        p0 = 1
        if n == l + 1:
            return p0
        p1 = -x / (alpha + 1) + 1
        if n == l + 2:
            return p1

        for i in range(2, n - l):
            p2 = ((2 * i - 1 + alpha - x) * p1 - (i + alpha - 1) * p0) / i
            p0 = p1
            p1 = p2
        return p1


def R_nl(n, l, r, Z=1):
    """
    General radial wave function for hydrogen-like orbitals.
    Parameters:
    n : int
        Principal quantum number (n >= 1)
    l : int
        Angular momentum quantum number (0 <= l < n)
    r : float or numpy.ndarray
        Radial distance from nucleus
    Z : float
        Effective nuclear charge
    Returns:
    R : float or numpy.ndarray
        Value of the radial wave function
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("l must be between 0 and n-1")

    rho = (2 * Z * r) / (n * a0)
    N = normalization_factor(n, l, Z)

    # Calculate the associated Laguerre polynomial
    L = generalized_laguerre(n, l, rho)

    return N * (rho ** l) * np.exp(-rho / 2) * L