import numpy as np
from scipy.special import lpmv, factorial


def Y_lm(l, m, theta, phi):
    """
    Allgemeine Kugelflächenfunktion für beliebige l und m.
    Parameter:
    l : int
        Drehimpulsquantenzahl (l >= 0)
    m : int
        Magnetische Quantenzahl (-l ≤ m ≤ l)
    theta : float oder numpy.ndarray
        Polarwinkel in Radiant (0 <= theta <= pi)
    phi : float oder numpy.ndarray
        Azimuthalwinkel in Radiant (0 <= phi < 2pi)
    Rückgabe:
    Y : complex oder numpy.ndarray
        Wert der Kugelflächenfunktion
    """
    if l < 0 or abs(m) > l:
        raise ValueError("Ungültige Werte für l oder m")
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    P_lm = lpmv(abs(m), l, np.cos(theta))
    if m < 0:
        return np.sqrt(2) * norm * P_lm * np.sin(abs(m) * phi)
    elif m > 0:
        return np.sqrt(2) * norm * P_lm * np.cos(m * phi)
    else:  # m == 0
        return norm * P_lm
