import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Physikalische Konstanten (in atomaren Einheiten)
hbar = 1
m_e = 1
e = 1
k_e = 1

# Magnetfeldstärke (in atomaren Einheiten)
B = 2.0

n = 75  # Erhöhte Anzahl von Punkten für bessere Auflösung
a = max(20, 40 / np.sqrt(1 + B))  # anpassende Boxgröße basierend auf B-Feld
d = a / n  # Schrittweite

# Erstelle 2D-Gitter mit höherer Auflösung für glattere Darstellung
plot_resolution = 500
x = np.linspace(-a / 2, a / 2, n)
y = np.linspace(-a / 2, a / 2, n)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

x_plot = np.linspace(-a / 2, a / 2, plot_resolution)
y_plot = np.linspace(-a / 2, a / 2, plot_resolution)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)


def V_total(x, y, B):
    """
    Gesamtpotential einschließlich:
    - Coulomb-Potential
    - Paramagnetischer Term (∝ B)
    - Diamagnetischer Term (∝ B²)
    """
    r = np.sqrt(x ** 2 + y ** 2)

    V_coulomb = -1 / np.sqrt(r ** 2 + 0.1)

    # Magnetische Terme
    # Lz-Term (paramagnetisch) - skaliert mit B
    V_para = 0.5 * B * (x * y - y * x)

    # Diamagnetischer Term - skaliert mit B
    V_dia = (B ** 2 / 8) * (x ** 2 + y ** 2)

    return V_coulomb + V_para + V_dia


def create_hamiltonian(n, d, B):
    N = (n - 2) ** 2  # Größe der Matrix (nur innere Punkte)
    H = np.zeros((N, N))

    print("Erstelle Hamilton-Matrix...")
    total_iterations = (n - 2) ** 2

    with tqdm(total=total_iterations) as pbar:
        for i in range(n - 2):
            for j in range(n - 2):
                # Aktueller Punktindex
                idx = i * (n - 2) + j

                # Position des aktuellen Punktes
                xi = x[i + 1]
                yi = y[j + 1]

                # Diagonalterm (kinetisch + potentiell)
                H[idx, idx] = 4 + d ** 2 * V_total(xi, yi, B)

                # Nicht-Diagonalterme (kinetische Energie)
                if j < n - 3:  # Rechter Nachbar
                    H[idx, idx + 1] = -1
                if j > 0:  # Linker Nachbar
                    H[idx, idx - 1] = -1
                if i < n - 3:  # Unterer Nachbar
                    H[idx, idx + (n - 2)] = -1
                if i > 0:  # Oberer Nachbar
                    H[idx, idx - (n - 2)] = -1

                pbar.update(1)

    return H


def reshape_wavefunction(psi, n):
    return psi.reshape((n - 2, n - 2))


def prepare_wavefunction(psi, smooth_sigma=1.0, interpolation_factor=None):
    # Normierung
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    # Berechne Wahrscheinlichkeitsdichte
    prob_density = np.abs(psi) ** 2

    prob_density = gaussian_filter(prob_density, sigma=smooth_sigma)

    if interpolation_factor is not None:
        from scipy.interpolate import RectBivariateSpline

        # Erstelle Koordinatenarrays für das ursprüngliche Gitter
        x_old = np.linspace(-a / 2, a / 2, prob_density.shape[0])
        y_old = np.linspace(-a / 2, a / 2, prob_density.shape[1])

        # Erstelle Interpolationsfunktion
        interp_spline = RectBivariateSpline(x_old, y_old, prob_density)

        # Erstelle neue Koordinatenarrays
        x_new = np.linspace(-a / 2, a / 2, plot_resolution)
        y_new = np.linspace(-a / 2, a / 2, plot_resolution)

        # Interpoliere die Daten
        prob_density = interp_spline(x_new, y_new)

    return prob_density


def plot_state(eigenvalue, eigenvector, state_num, B, a, n, d, separate=True):
    if separate:
        fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(111, projection='3d')

    # Bereite Wellenfunktionsdaten mit Glättung und Interpolation vor
    psi = reshape_wavefunction(eigenvector, n)
    prob_density = prepare_wavefunction(psi, smooth_sigma=1.5, interpolation_factor=plot_resolution)

    # Erstelle den 3D-Oberflächenplot mit verbesserter Glättung
    surf = ax.plot_surface(X_plot, Y_plot,
                           prob_density,
                           cmap=plt.cm.viridis,
                           linewidth=0,
                           antialiased=False,
                           rcount=500,
                           ccount=500)

    # Anpassen der Darstellung
    ax.set_title(f'Zustand {state_num}, E = {eigenvalue / (2 * d * d):.3f} a.u.')
    ax.set_xlabel('x (atomare Einheiten)')
    ax.set_ylabel('y (atomare Einheiten)')
    ax.set_zlabel('Wahrscheinlichkeitsdichte')

    # Farbskala hinzufügen
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Betrachtungswinkel für bessere Visualisierung einstellen
    ax.view_init(elev=30, azim=45)

    if separate:
        plt.suptitle(f'Wasserstoffatom-Wellenfunktion im Magnetfeld (B = {B} a.u.)\n' +
                     f'Gitter: {n}x{n} Punkte, Boxgröße: {a:.1f} a.u.',
                     fontsize=14)
        plt.tight_layout()

    return fig, ax



H = create_hamiltonian(n, d, B)
print("\nLöse Eigenwertproblem...")
eigenvalues, eigenvectors = np.linalg.eigh(H)
print("Eigenwertproblem gelöst!")

# Stelle Zustände dar
print("\nErstelle Plots...")
num_states = 10


for i in range(num_states):
    fig, ax = plot_state(eigenvalues[i], eigenvectors[:, i], i + 1, B, a, n, d, separate=True)
    plt.show()


"""
fig = plt.figure(figsize=(20, 15))
for i in range(num_states):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    plot_state(eigenvalues[i], eigenvectors[:, i], i + 1, B, a, n, d, separate=False)
plt.suptitle(f'Wasserstoffatom-Wellenfunktionen im Magnetfeld (B = {B} a.u.)\n' +
             f'Gitter: {n}x{n} Punkte, Boxgröße: {a:.1f} a.u.',
             fontsize=14)
plt.tight_layout()
plt.show()
"""

# Gebe Energieniveaus aus
print("\nEnergieniveaus (in atomaren Einheiten):")
for i in range(num_states):
    print(f"Zustand {i + 1}: E = {eigenvalues[i] / (2 * d * d):.6f}")