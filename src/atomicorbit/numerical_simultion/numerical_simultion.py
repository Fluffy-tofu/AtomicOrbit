import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Physical constants (in atomic units)
hbar = 1
m_e = 1
e = 1
k_e = 1

# Magnetic field strength (in atomic units)
B = 0.0  # Can be adjusted to see different magnetic field strengths

# Problem parameters - adjust based on B field
n = 75  # increased number of points for better resolution
a = max(20, 40 / np.sqrt(1 + B))  # adaptive box size based on B field
d = a / n  # step size

# Create 2D grid with higher resolution for smoother plotting
plot_resolution = 500  # Increased resolution for plotting
x = np.linspace(-a / 2, a / 2, n)
y = np.linspace(-a / 2, a / 2, n)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

# Create high-resolution grid for plotting
x_plot = np.linspace(-a / 2, a / 2, plot_resolution)
y_plot = np.linspace(-a / 2, a / 2, plot_resolution)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)


def V_total(x, y, B):
    """
    Total potential including:
    - Coulomb potential
    - Paramagnetic term (∝ B)
    - Diamagnetic term (∝ B²)
    """
    r = np.sqrt(x ** 2 + y ** 2)

    # Coulomb potential with soft core to avoid singularity
    V_coulomb = -1 / np.sqrt(r ** 2 + 0.1)

    # Magnetic terms
    # Lz term (paramagnetic) - scaled with B
    V_para = 0.5 * B * (x * y - y * x)

    # Diamagnetic term - scaled with B
    V_dia = (B ** 2 / 8) * (x ** 2 + y ** 2)

    return V_coulomb + V_para + V_dia


def create_hamiltonian(n, d, B):
    N = (n - 2) ** 2  # Size of matrix (interior points only)
    H = np.zeros((N, N))

    print("Building Hamiltonian matrix...")
    total_iterations = (n - 2) ** 2

    with tqdm(total=total_iterations) as pbar:
        for i in range(n - 2):
            for j in range(n - 2):
                # Current point index
                idx = i * (n - 2) + j

                # Position of current point
                xi = x[i + 1]
                yi = y[j + 1]

                # Diagonal term (kinetic + potential)
                H[idx, idx] = 4 + d ** 2 * V_total(xi, yi, B)

                # Off-diagonal terms (kinetic energy)
                if j < n - 3:  # Right neighbor
                    H[idx, idx + 1] = -1
                if j > 0:  # Left neighbor
                    H[idx, idx - 1] = -1
                if i < n - 3:  # Down neighbor
                    H[idx, idx + (n - 2)] = -1
                if i > 0:  # Up neighbor
                    H[idx, idx - (n - 2)] = -1

                pbar.update(1)

    print(H)
    return H


def reshape_wavefunction(psi, n):
    return psi.reshape((n - 2, n - 2))


def prepare_wavefunction(psi, smooth_sigma=1.0, interpolation_factor=None):
    # Normalize
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    # Get probability density
    prob_density = np.abs(psi) ** 2

    # Apply Gaussian smoothing
    prob_density = gaussian_filter(prob_density, sigma=smooth_sigma)

    if interpolation_factor is not None:
        from scipy.interpolate import RectBivariateSpline

        # Create coordinate arrays for the original grid
        x_old = np.linspace(-a / 2, a / 2, prob_density.shape[0])
        y_old = np.linspace(-a / 2, a / 2, prob_density.shape[1])

        # Create interpolation function
        interp_spline = RectBivariateSpline(x_old, y_old, prob_density)

        # Create new coordinate arrays
        x_new = np.linspace(-a / 2, a / 2, plot_resolution)
        y_new = np.linspace(-a / 2, a / 2, plot_resolution)

        # Interpolate the data
        prob_density = interp_spline(x_new, y_new)

    return prob_density


def plot_state(eigenvalue, eigenvector, state_num, B, a, n, d, separate=True):
    """
    Plot a single state with optional separate figure creation
    """
    if separate:
        fig = plt.figure(figsize=(12, 10))

    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Prepare wavefunction data with smoothing and interpolation
    psi = reshape_wavefunction(eigenvector, n)
    prob_density = prepare_wavefunction(psi, smooth_sigma=1.5, interpolation_factor=plot_resolution)

    # Create the 3D surface plot with enhanced smoothness
    surf = ax.plot_surface(X_plot, Y_plot,
                           prob_density,
                           cmap=plt.cm.viridis,
                           linewidth=0,
                           antialiased=False,
                           rcount=500,
                           ccount=500)

    # Customize the plot
    ax.set_title(f'State {state_num}, E = {eigenvalue / (2 * d * d):.3f} a.u.')
    ax.set_xlabel('x (atomic units)')
    ax.set_ylabel('y (atomic units)')
    ax.set_zlabel('Probability Density')

    # Add color bar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Set viewing angle for better visualization
    ax.view_init(elev=30, azim=45)

    if separate:
        plt.suptitle(f'Hydrogen Atom Wavefunction in Magnetic Field (B = {B} a.u.)\n' +
                     f'Grid: {n}x{n} points, Box size: {a:.1f} a.u.',
                     fontsize=14)
        plt.tight_layout()

    return fig, ax


# Create Hamiltonian and find eigenstates
H = create_hamiltonian(n, d, B)
print("\nSolving eigenvalue problem...")
eigenvalues, eigenvectors = np.linalg.eigh(H)
print("Eigenvalue problem solved!")

# Plot states
print("\nGenerating plots...")
num_states = 10

# Option 1: Create separate figures (default)
for i in range(num_states):
    fig, ax = plot_state(eigenvalues[i], eigenvectors[:, i], i + 1, B, a, n, d, separate=True)
    plt.show()

# Option 2: Create combined figure (commented out by default)
"""
fig = plt.figure(figsize=(20, 15))
for i in range(num_states):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    plot_state(eigenvalues[i], eigenvectors[:, i], i + 1, B, a, n, d, separate=False)
plt.suptitle(f'Hydrogen Atom Wavefunctions in Magnetic Field (B = {B} a.u.)\n' +
             f'Grid: {n}x{n} points, Box size: {a:.1f} a.u.',
             fontsize=14)
plt.tight_layout()
plt.show()
"""

# Print energy levels
print("\nEnergy levels (in atomic units):")
for i in range(num_states):
    print(f"State {i + 1}: E = {eigenvalues[i] / (2 * d * d):.6f}")