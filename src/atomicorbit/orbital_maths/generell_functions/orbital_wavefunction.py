try:
    from atomicorbit.orbital_maths.generell_functions.radial_part_wavefunc import R_nl
    from atomicorbit.orbital_maths.generell_functions.spherical_harmonic_func import Y_lm
except:
    from src.atomicorbit.orbital_maths.generell_functions.radial_part_wavefunc import R_nl
    from src.atomicorbit.orbital_maths.generell_functions.spherical_harmonic_func import Y_lm

def orbital_wavefunction(n, l, m, r, theta, phi):
    """Wahrscheinlichtkeitsdichte berechnen"""
    R = R_nl(n=n, l=l, r=r)
    Y = Y_lm(m=m, l=l, phi=phi, theta=theta)
    return (R * Y) ** 2