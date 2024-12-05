import numpy as np
from scipy.special import sph_harm, genlaguerre
from scipy.integrate import quad
from vispy import app, scene
import math
import logging
from src.atomicorbit.visualization_vispy.atom_orbital import prepare_orbital_data, visualize_orbital


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_quantum_numbers(electron_count):
    """
    Generate quantum numbers following the Aufbau principle.
    """
    n_list, l_list, m_list = [], [], []
    n = 1
    electrons_placed = 0

    while electrons_placed < electron_count:
        # For each principal quantum number n
        for l in range(n):  # l goes from 0 to n-1
            for m in range(-l, l + 1):  # m goes from -l to +l
                # Each orbital can hold 2 electrons (spin up/down)
                for _ in range(2):
                    if electrons_placed < electron_count:
                        n_list.append(n)
                        l_list.append(l)
                        m_list.append(m)
                        electrons_placed += 1
        n += 1

    return n_list, l_list, m_list


def radial_wavefunction(n, l, r):
    """
    Calculate the radial part of the wavefunction.
    """
    # Bohr radius (in atomic units)
    a0 = 1.0

    # Calculate rho
    rho = 2 * r / (n * a0)

    # Normalization factor
    norm = np.sqrt((2.0 / (n * a0)) ** 3 *
                   math.factorial(n - l - 1) /
                   (2 * n * math.factorial(n + l)))

    # Associated Laguerre polynomial
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)

    return norm * np.exp(-rho / 2) * (rho ** l) * L


def angular_wavefunction(l, m, theta, phi):
    """
    Calculate the angular part of the wavefunction using spherical harmonics.
    """
    return sph_harm(m, l, phi, theta)


def single_electron_wavefunction(n, l, m, r, theta, phi):
    """
    Calculate the complete wavefunction for a single electron.
    """
    R = radial_wavefunction(n, l, r)
    Y = angular_wavefunction(l, m, theta, phi)
    return R * Y


def calculate_multi_electron_density(electron_count, grid_points=50):
    """
    Calculate electron density for multi-electron system.
    Returns coordinates and probability density in format suitable for visualization.
    """
    logger.info(f"Calculating {electron_count}-electron system...")

    # Get quantum numbers for all electrons
    n_list, l_list, m_list = get_quantum_numbers(electron_count)

    # Create spatial grid
    r_max = max(n_list) * 2.0  # Adjust radius based on highest n
    r = np.linspace(0, r_max, grid_points)
    theta = np.linspace(0, np.pi, grid_points)
    phi = np.linspace(0, 2 * np.pi, grid_points)

    # Create meshgrid
    r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')

    # Calculate Cartesian coordinates
    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)

    # Initialize total electron density
    total_density = np.zeros_like(r_grid, dtype=np.complex128)

    # Calculate wavefunctions for each electron and sum their densities
    for i, (n, l, m) in enumerate(zip(n_list, l_list, m_list)):
        logger.info(f"Processing electron {i + 1}/{electron_count} (n={n}, l={l}, m={m})")
        wavefunction = single_electron_wavefunction(n, l, m, r_grid, theta_grid, phi_grid)
        total_density += np.abs(wavefunction) ** 2

    # Normalize the total density
    total_density = np.real(total_density)  # Take real part as density should be real
    total_density = total_density / np.max(total_density)

    logger.info("Calculation completed successfully")
    return x, y, z, total_density


if __name__ == "__main__":
    try:
        # Calculate electron density for atoms up to Calcium (Z=20)
        electron_count = 50
        grid_points = 50

        # Calculate the electron density
        x, y, z, density = calculate_multi_electron_density(electron_count, grid_points)
        print("calcu finsihed")
        # Prepare data for visualization using your functions
        points, colors = prepare_orbital_data(x, y, z, density)
        print("prep date")

        # Visualize using your function
        canvas, view = visualize_orbital(points, colors)
        print("visualize")
        canvas.show()
        app.run()
        print("ru?")

    except Exception as e:
        logger.error(f"Error in calculation: {str(e)}")