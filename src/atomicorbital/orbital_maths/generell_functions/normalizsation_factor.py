import numpy as np
import math
# Konstanten
a0 = 0.529177210903e-10  # Bohr-Radius in Metern
def normalization_factor(n, l, Z=1):
    """Berechnet den Normierungsfaktor für Wasserstoff-ähnliche Orbitale."""
    numerator = (2 * Z / (n * a0))**3 * math.factorial(n-l-1)
    denominator = 2*n * (math.factorial(n+l))
    return np.sqrt(numerator / denominator)
