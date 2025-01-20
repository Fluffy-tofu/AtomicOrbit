import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd

def set_plot_style():
    """Konfiguriere den Plot-Stil"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.grid'] = True

def V_total(x, y, B):
    """Gesamtpotential mit Coulomb- und magnetischen Termen"""
    r = np.sqrt(x ** 2 + y ** 2)
    V_coulomb = -1 / np.sqrt(r ** 2 + 0.1)
    V_para = 0.5 * B * (x * y - y * x)
    V_dia = (B ** 2 / 8) * (x ** 2 + y ** 2)
    return V_coulomb + V_para + V_dia


def create_hamiltonian(n, B):
    """Hamilton-Matrix für gegebene Gittergröße und Magnetfeld erstellen"""
    a = max(20, 40 / np.sqrt(1 + B))
    d = a / n
    N = (n - 2) ** 2
    H = np.zeros((N, N))

    x = np.linspace(-a / 2, a / 2, n)
    y = np.linspace(-a / 2, a / 2, n)

    for i in range(n - 2):
        for j in range(n - 2):
            idx = i * (n - 2) + j
            xi = x[i + 1]
            yi = y[j + 1]

            H[idx, idx] = 4 + d ** 2 * V_total(xi, yi, B)

            if j < n - 3: H[idx, idx + 1] = -1
            if j > 0: H[idx, idx - 1] = -1
            if i < n - 3: H[idx, idx + (n - 2)] = -1
            if i > 0: H[idx, idx - (n - 2)] = -1

    return H


def benchmark_grid_scaling(grid_sizes, B=1.0, num_repeats=3):
    """Zeitmessung für verschiedene Gittergrößen"""
    times_hamiltonian = []
    times_eigensolve = []
    total_times = []

    for n in tqdm(grid_sizes, desc="Benchmarking Gittergrößen"):
        time_ham_list = []
        time_eigen_list = []
        time_total_list = []

        for _ in range(num_repeats):
            start = time.time()
            H = create_hamiltonian(n, B)
            end_ham = time.time()
            time_ham = end_ham - start

            eigenvalues, eigenvectors = np.linalg.eigh(H)
            end_eigen = time.time()
            time_eigen = end_eigen - end_ham

            time_ham_list.append(time_ham)
            time_eigen_list.append(time_eigen)
            time_total_list.append(time_ham + time_eigen)

        times_hamiltonian.append(np.mean(time_ham_list))
        times_eigensolve.append(np.mean(time_eigen_list))
        total_times.append(np.mean(time_total_list))

    return times_hamiltonian, times_eigensolve, total_times


def fit_power_law(x, y):
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    return coeffs[0]


def set_plot_style():
    """Konfiguriere den Plot-Stil"""
    plt.style.use('default')  # Verwende den Standard-Stil
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.grid'] = True


grid_sizes = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
B = 1.0  # a.u

print("Führe Benchmarks durch...")
times_ham, times_eigen, times_total = benchmark_grid_scaling(grid_sizes, B)

scaling_ham = fit_power_law(grid_sizes, times_ham)
scaling_eigen = fit_power_law(grid_sizes, times_eigen)
scaling_total = fit_power_law(grid_sizes, times_total)

print("\nSkalierungsanalyse:")
print(f"Hamilton-Matrix-Erstellung skaliert mit O(n^{scaling_ham:.2f})")
print(f"Eigenwertberechnung skaliert mit O(n^{scaling_eigen:.2f})")
print(f"Gesamtberechnung skaliert mit O(n^{scaling_total:.2f})")

# Plot
set_plot_style()

# 1. Linear
plt.figure()
plt.plot(grid_sizes, times_ham, 'o-', label='Hamilton-Matrix-Erstellung', color='#1f77b4')
plt.plot(grid_sizes, times_eigen, 's-', label='Eigenwertberechnung', color='#ff7f0e')
plt.plot(grid_sizes, times_total, '^-', label='Gesamtzeit', color='#2ca02c')
plt.xlabel('Gittergröße (n)')
plt.ylabel('Zeit (Sekunden)')
plt.title('Berechnungsskalierung (Lineare Skala)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('skalierung_linear.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Logarithmic
plt.figure()

offset = 1e-10
plt.loglog(grid_sizes,
          [max(t, offset) for t in times_ham],
          'o-',
          label='Hamilton-Matrix-Erstellung',
          color='#1f77b4')
plt.loglog(grid_sizes,
          [max(t, offset) for t in times_eigen],
          's-',
          label='Eigenwertberechnung',
          color='#ff7f0e')
plt.loglog(grid_sizes,
          [max(t, offset) for t in times_total],
          '^-',
          label='Gesamtzeit',
          color='#2ca02c')
plt.xlabel('Gittergröße (n)')
plt.ylabel('Zeit (Sekunden)')
plt.title('Berechnungsskalierung (Logarithmische Skala)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('skalierung_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Normalisiert
plt.figure()

normalized_times = {
    'Hamilton-Matrix': np.array(times_ham) / times_ham[0],
    'Eigenwertberechnung': np.array(times_eigen) / times_eigen[0],
    'Gesamt': np.array(times_total) / times_total[0]
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for (label, times), color in zip(normalized_times.items(), colors):
    plt.plot(grid_sizes, times, 'o-', label=f'{label}', color=color)

plt.xlabel('Gittergröße (n)')
plt.ylabel('Normierte Zeit (relativ zu Gittergröße n=20)')
plt.title('Normierte Skalierungsanalyse')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('skalierung_normiert.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure()
memory_usage = [(n-2)**2 * 8 / (1024*1024) for n in grid_sizes]  # Memory in MB
plt.plot(grid_sizes, memory_usage, 'o-', color='#9467bd')
plt.xlabel('Gittergröße (n)')
plt.ylabel('Geschätzter Speicherbedarf (MB)')
plt.title('Speichernutzung der Hamilton-Matrix')
plt.grid(True)
plt.tight_layout()
plt.savefig('speichernutzung.png', dpi=300, bbox_inches='tight')
plt.close()


print("\nSkalierungsanalyse:")
print("\nAbsolute Maximalzeiten:")
print(f"Hamilton-Matrix-Erstellung: {max(times_ham):.2f} Sekunden")
print(f"Eigenwertberechnung: {max(times_eigen):.2f} Sekunden")
print(f"Gesamtzeit: {max(times_total):.2f} Sekunden")

print("\nNormierte Maximalwerte (relativ zum Startwert):")
print(f"Hamilton-Matrix-Erstellung: {normalized_times['Hamilton-Matrix'][-1]:.2f}x")
print(f"Eigenwertberechnung: {normalized_times['Eigenwertberechnung'][-1]:.2f}x")
print(f"Gesamtzeit: {normalized_times['Gesamt'][-1]:.2f}x")