import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, laguerre
from src.atomicorbit.visualization_vispy.atom_orbital import SlaterDeterminant


def R_nl_explicit(n, l, r, Z=1):
    """
    Explicit calculation of the radial wavefunction
    """
    # Normalization
    norm = np.sqrt((2 * Z / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))

    # Bohr radius (in atomic units)
    a0 = 1.0

    # Scaled radius
    rho = (2 * Z * r) / (n * a0)

    # Associated Laguerre polynomial
    L = laguerre(n - l - 1, 2 * l + 1)(rho)

    return norm * np.exp(-rho / 2) * (rho ** l) * L


def Y_lm_explicit(l, m, theta, phi):
    """
    Explicit calculation of spherical harmonics
    """
    # Normalization
    norm = np.sqrt((2 * l + 1) * factorial(l - abs(m)) / (4 * np.pi * factorial(l + abs(m))))

    # Associated Legendre polynomial
    P = np.polynomial.legendre.Legendre.basis(l)

    # Angular part
    angular = np.exp(1j * m * phi) * P(np.cos(theta))

    return norm * angular


def calculate_density_comparison(n, l, m, positions, with_slater=True):
    """
    Calculate electron density with and without Slater determinant
    """
    r = np.sqrt(np.sum(positions ** 2, axis=1))
    theta = np.arccos(positions[:, 2] / (r + 1e-10))
    phi = np.arctan2(positions[:, 1], positions[:, 0])

    # Calculate basic wavefunction
    R = R_nl_explicit(n, l, r)
    Y = Y_lm_explicit(l, m, theta, phi)
    basic_wf = R * Y
    basic_density = np.abs(basic_wf) ** 2

    if not with_slater:
        return basic_density

    # Calculate with Slater determinant
    slater = SlaterDeterminant()
    slater.add_orbital(n, l, m, 0.5)  # Add the orbital we're analyzing

    # Calculate Slater determinant for each position
    slater_densities = np.zeros(len(positions))
    batch_size = 100

    for i in range(0, len(positions), batch_size):
        batch_end = min(i + batch_size, len(positions))
        batch_positions = positions[i:batch_end]

        try:
            slater_det = np.abs(slater.calculate_determinant(batch_positions)) ** 2
            slater_densities[i:batch_end] = slater_det
        except Exception as e:
            print(f"Warning in Slater calculation: {str(e)}")
            slater_densities[i:batch_end] = basic_density[i:batch_end]

    return slater_densities


def plot_density_comparison(n, l, m):
    """
    Create comparison plots of electron density
    """
    # Create grid of points
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))

    # Calculate densities
    basic_density = calculate_density_comparison(n, l, m, positions, with_slater=False)
    slater_density = calculate_density_comparison(n, l, m, positions, with_slater=True)

    # Reshape for plotting
    basic_density = basic_density.reshape(X.shape)
    slater_density = slater_density.reshape(X.shape)

    # Calculate difference
    difference = slater_density - basic_density

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot basic density
    im1 = ax1.imshow(basic_density, extent=[-10, 10, -10, 10], cmap='viridis')
    ax1.set_title('Basic Density')
    plt.colorbar(im1, ax=ax1)

    # Plot Slater density
    im2 = ax2.imshow(slater_density, extent=[-10, 10, -10, 10], cmap='viridis')
    ax2.set_title('With Slater Determinant')
    plt.colorbar(im2, ax=ax2)

    # Plot difference
    im3 = ax3.imshow(difference, extent=[-10, 10, -10, 10],
                     cmap='RdBu', norm=plt.Normalize(vmin=-np.max(np.abs(difference)),
                                                     vmax=np.max(np.abs(difference))))
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nStatistics for n={n}, l={l}, m={m}:")
    print(f"Maximum basic density: {np.max(basic_density):.6f}")
    print(f"Maximum Slater density: {np.max(slater_density):.6f}")
    print(f"Maximum absolute difference: {np.max(np.abs(difference)):.6f}")
    print(f"Average absolute difference: {np.mean(np.abs(difference)):.6f}")


# Example usage
if __name__ == "__main__":
    # Plot for 2p orbital (n=2, l=1, m=0)
    plot_density_comparison(2, 1, 0)

    # Plot for 3d orbital (n=3, l=2, m=0)
    plot_density_comparison(3, 2, 0)