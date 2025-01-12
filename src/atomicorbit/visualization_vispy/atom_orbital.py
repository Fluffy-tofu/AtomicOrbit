import numpy as np
from vispy import app, scene
import scipy.special as sp
from collections import defaultdict
from scipy.special import genlaguerre
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
import math
import scipy.linalg as la
from src.atomicorbit.orbital_maths.generell_functions.spherical_harmonic_func import Y_lm
from src.atomicorbit.orbital_maths.generell_functions.radial_part_wavefunc import R_nl
import seaborn as sns
from scipy.constants import e, hbar, m_e


mu_B = 5.788e-7 # eV/Tesla
feinstrukturkonstante = 1/137


class GeneralFunctions:
    def __init__(self, visual_dict):
        self.visual_dict = visual_dict
        # Physikalische Konstanten
        self.e = e  # Elementarladung
        self.hbar = hbar  # Reduziertes Plancksches Wirkungsquantum
        self.m_e = m_e  # Elektronenmasse
        self.mu_B = mu_B  # Bohrsches Magneton
        self.cached_coordinates = None

    def calculate_orbital_points(self, n, l, m, Z, electron_count, threshold=0.1, num_points=100000, magnetic_field=0):
        """
        Berechne Orbitale-Punkte mit einem mit Schwellenwert
        Parameter:
        -----------
        n, l, m: int
            Quantenzahlen
        Z: int
            Ordnungszahl
        elektronenanzahl: int
            Anzahl der Elektronen im Orbital
        threshold: float
            Mindestwert für die Wahrscheinlichkeitsdichte (0 bis 1)
        num_points: int
            Anzahl der zu generierenden Punkte
        """

        if electron_count == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Generate coordinates with appropriate scaling
        r, theta, phi = self.generate_grid(n, Z, num_points)

        if magnetic_field != 0:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, magnetic_field)

        else:
            density = self.probability_density(n, l, m, Z, r, theta, phi)

        # Normalize to electron count
        density *= electron_count / (2.0 * np.sum(density) / num_points)

        # Add small offset and normalize
        #density = density + np.max(density) * 0.01
        max_density = np.max(density)
        if max_density > 1e-10:
            density = density / max_density
        else:
            density = np.ones_like(density) * 0.1

        # Safety checks
        density = np.nan_to_num(density, nan=0.1, posinf=1.0, neginf=0.0)
        #density = np.clip(density, 0.01, 1)

        # Apply threshold filter
        mask = (density >= threshold) & (np.random.random(num_points) < (density * 0.5 + 0.1))

        x, y, z = self.convert_cartesian(r, theta, phi)

        return x[mask], y[mask], z[mask], density[mask]

    def probability_density(self, n, l, m, Z, r, theta, phi):
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        density = np.abs(R_nl * Y_lm) ** 2
        return density

    def wave_func(self, n, l, m, Z, r, theta, phi):
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        return R_nl * Y_lm

    def generate_grid(self, n, Z, num_points):
        """Generiert oder verwendet gecachte Koordinaten"""
        if self.cached_coordinates is None:
            r = np.random.exponential(scale=n ** 2 / Z, size=num_points)
            theta = np.arccos(2 * np.random.random(num_points) - 1)
            phi = 2 * np.pi * np.random.random(num_points)
            self.cached_coordinates = (r, theta, phi)
        return self.cached_coordinates

    def clear_cache(self):
        """Löscht den Cache der Koordinaten"""
        self.cached_coordinates = None

    def convert_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def radial_function(self, n, l, r, Z):
        """Besser radiale Funktion"""
        try:
            rho = (2 * Z * r) / n
            norm = np.sqrt((2 * Z / n) ** 3 * math.factorial(n - l - 1) /
                           (2 * n * math.factorial(n + l)))
            laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)(rho)
            radial_part = norm * np.exp(-rho / 2) * (rho ** l) * laguerre_poly

            # Handhabung von numerischen Problemen
            radial_part = np.nan_to_num(radial_part, nan=0.0, posinf=0.0, neginf=0.0)
            return radial_part
        except Exception as e:
            print(f"Warning: Error in radial function for n={n}, l={l}: {str(e)}")
            return np.ones_like(r) * 0.1

    def spherical_harmonics(self, l, m, theta, phi):
        """Sichere Version der Kugelflächenfunktionen"""
        try:
            Y = Y_lm(l, m, theta, phi)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            return Y
        except Exception as e:
            print(f"Warnung: Fehler in Kugelflächenfunktion für l={l}, m={m}: {str(e)}")
            return np.zeros_like(theta)

    def add_orbital(self, n, l, m, Z, electron_count, points_plotted):
        """Visualisierung"""
        x, y, z, density = self.calculate_orbital_points(n, l, m, Z, electron_count,
                                                         threshold=self.visual_dict["prob_threshold"],
                                                         num_points=self.visual_dict["num_points"],
                                                         magnetic_field=self.visual_dict["magnetic_field"])
        points_plotted += len(x)
        if len(x) > 0:
            colors = self.get_density_color(density)

            state = self.visual_dict["state"]
            not_see_inside = self.visual_dict["not_see_inside"]
            blend = self.visual_dict["blend"]
            edges = self.visual_dict["edges"]
            point_size = self.visual_dict["point_size"]

            point_size = 10 * density + 3 if point_size == "dynamic" else point_size
            edge_color = None if edges else colors
            scatter = scene.visuals.Markers()
            scatter.set_gl_state(state, depth_test=not_see_inside, blend=blend)
            scatter.set_data(
                np.column_stack((x, y, z)),
                edge_color=edge_color,
                face_color=colors,
                size=point_size
            )
            self.view.add(scatter)
        return points_plotted

    def get_density_color(self, density):
        red = np.clip(density * 2, 0, 1)
        blue = np.clip(2 - density * 2, 0, 1)
        green = np.clip(1 - np.abs(density - 0.5) * 2, 0, 1)
        alpha = np.clip(density * 0.8 + 0.2, 0, 1)
        return np.column_stack((red, green, blue, alpha))

    def get_density_color_magnetic_field(self, density):
        red = np.clip(2 - density * 2, 0, 1)
        blue = np.clip(density * 2, 0, 1)
        green = np.clip(1 - np.abs(density - 0.5) * 2, 0, 1)
        alpha = np.clip(density * 0.8 + 0.2, 0, 1)
        return np.column_stack((red, green, blue, alpha))

    def calculate_orbitals_energy(self, n, Z):
        energy = -13.6/n**2
        sommerfeld_formular = energy * (Z ** 2 / (1 + feinstrukturkonstante ** 2/n ** 2) ** 2)
        # print(f"Z={Z} || n={n} ---> {sommerfeld_formular}")
        return sommerfeld_formular

    def calculate_orbitals_energy_magneticfield(self, n, m_l, b_field, Z):
        """Erweiterte Version mit diamagnetischem Term"""
        base_energy = self.calculate_orbitals_energy(n, Z)

        # Zeeman Term (orbital)
        zeeman_energy = self.mu_B * b_field * m_l

        # Diamagnetischer Term
        # Erwartungswert von r² für Wasserstoff
        r2_expect = (n ** 2 * (5 * n ** 2 + 1)) * (0.529e-10) ** 2  # in m²
        dia_energy = (self.e ** 2 * b_field ** 2 / (8 * self.m_e)) * r2_expect

        return base_energy + zeeman_energy + dia_energy

    def scaling_factor_magnetic_field(self, E_new, E_base):
        return np.sqrt(abs(E_new)/abs(E_base))

    def first_order_apporoximation(self, n, l, r, Z, m, theta, phi, field):
        """Erweiterte Version mit Störungskorrektur"""
        # Basis-Wellenfunktion berechnen
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        psi_0 = R_nl * Y_lm

        # Kartesische Koordinaten für diamagnetischen Term
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)

        # Störungskorrektur
        orbital_term = self.mu_B * field * m
        dia_term = (self.e ** 2 * field ** 2 / (8 * self.m_e)) * (x ** 2 + y ** 2)
        correction = orbital_term + dia_term
        print(dia_term)
        print(orbital_term)

        # Gestörte Wellenfunktion
        psi = psi_0 * (1 + correction)
        return psi

    def first_order_correction(self, n, l, m, B, r, theta, phi):
        """
        Improved first-order perturbation theory implementation
        """
        # Convert to atomic units
        a0 = 5.29177e-11  # Bohr radius
        B_atomic = B * (a0 ** 2 * self.e / self.hbar)

        # Unperturbed wavefunction
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        E_n = self.calculate_orbitals_energy(n, 1)

        # Initialize correction
        psi_1 = np.zeros_like(psi_0, dtype=complex)

        # Volume element for integration
        dV = r ** 2 * np.sin(theta)

        # Sum over intermediate states
        for n_prime in range(max(1, n - 2), n + 3):
            for l_prime in range(max(0, l - 1), min(n_prime, l + 2)):
                for m_prime in range(-l_prime, l_prime + 1):
                    if (n_prime, l_prime, m_prime) == (n, l, m):
                        continue

                    try:
                        # Intermediate state
                        psi_k = self.wave_func(n_prime, l_prime, m_prime, 1, r, theta, phi)
                        E_k = self.calculate_orbitals_energy(n_prime, 1)

                        # Matrix element with proper integration
                        H_k0 = self.calculate_matrix_element(psi_k, psi_0, B_atomic, r, theta, phi)
                        integral = np.sum(H_k0 * dV)

                        # Energy denominator
                        delta_E = E_k - E_n
                        if abs(delta_E) > 1e-10:
                            psi_1 += (integral / delta_E) * psi_k

                    except Exception as e:
                        continue

        # Normalize the total wavefunction
        psi_total = psi_0 + psi_1
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))

        if norm > 0:
            psi_total = psi_total / norm

        return psi_total

    def calculate_matrix_element(self, psi_k, psi_0, B, r, theta, phi):
        """
        Improved matrix element calculation including proper integration
        """
        # Zeeman term
        Lz_psi_0 = -1j * self.hbar * np.gradient(psi_0, phi, axis=2)
        zeeman = self.mu_B * B * np.conjugate(psi_k) * Lz_psi_0

        # Diamagnetic term
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        dia = (self.e ** 2 * B ** 2 / (8 * self.m_e)) * np.conjugate(psi_k) * (x ** 2 + y ** 2) * psi_0

        return zeeman + dia

    def wave_func_magneticfield(self, n, l, r, Z, m, theta, phi, field, approx):
        if approx:
            print("approx")
            return self.first_order_apporoximation(n, l, r, Z, m, theta, phi, field)
        else:
            print("no approx")
            psi_1 = self.first_order_correction(n, l, m, field, r, theta, phi)
            return psi_1

    def probability_density_magnetic_field(self, n, l, r, Z, m, theta, phi, field):
        """
        Improved probability density calculation for magnetic field effects
        """
        # Calculate wavefunction with perturbation
        psi = self.wave_func_magneticfield(n, l, r, Z, m, theta, phi, field,
                                           approx=self.visual_dict["Störtheorie Näherung"])

        # Calculate density with proper normalization
        density = np.abs(psi) ** 2

        # Volume element for proper normalization
        dV = r ** 2 * np.sin(theta)
        norm = np.sum(density * dV)

        if norm > 1e-10:  # Avoid division by very small numbers
            density = density / norm

        # Add small offset to prevent visualization artifacts
        density = density + np.max(density) * 0.001

        return density


class AtomicConfiguration:
    def __init__(self):
        super().__init__()
        self.orbital_filling = {
            's': 2,
            'p': 6,
            'd': 10,
            'f': 14
        }

        self.orbital_order = [
            '1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p',
            '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p'
        ]

        self.m_values = {
            's': [0],
            'p': [-1, 0, 1],
            'd': [-2, -1, 0, 1, 2],
            'f': [-3, -2, -1, 0, 1, 2, 3]
        }


    def get_configuration(self, atomic_number):
        if atomic_number < 1:
            raise ValueError("Atomzahl muss positiv sein")

        configuration = defaultdict(lambda: defaultdict(int))
        electrons_left = atomic_number

        for orbital in self.orbital_order:
            if electrons_left <= 0:
                break

            n = int(orbital[0])
            l_symbol = orbital[1]
            l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}[l_symbol]

            max_per_m = 2
            m_values = self.m_values[l_symbol]

            # Verteile Elektronen nach Hundscher Regel
            for m in m_values:
                if electrons_left <= 0:
                    break
                # Spin up
                if electrons_left > 0:
                    configuration[orbital][m] += 1
                    electrons_left -= 1

            # Dann fülle mit spin down
            for m in m_values:
                if electrons_left <= 0:
                    break
                if configuration[orbital][m] < max_per_m:
                    configuration[orbital][m] += 1
                    electrons_left -= 1

        return configuration


class ElectronDensityVisualizer(GeneralFunctions):
    def __init__(self, visual_dict):
        super().__init__(visual_dict)
        self.canvas = scene.SceneCanvas(keys='interactive', size=(2000, 1200), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 2
        self.view.camera.elevation = visual_dict["angle"]
        self.axis = scene.XYZAxis(parent=self.view.scene,)
        self.atomic_config = AtomicConfiguration()
        self.visual_dict = visual_dict

        label_distance = 1.2
        self.x_label = scene.Text("x", pos=[label_distance, 0, 0], color='red',
                                  font_size=20, parent=self.view.scene)
        self.y_label = scene.Text("y", pos=[0, label_distance, 0], color='green',
                                  font_size=20, parent=self.view.scene)

        self.z_label = scene.Text("z", pos=[0, 0, label_distance], color='blue',
                                  font_size=20, parent=self.view.scene)

    def calculate_orbital_points(self, n, l, m, Z, electron_count, threshold=0.1, num_points=100000, magnetic_field=0):
        """
        Improved orbital calculation that preserves orbital structure
        """
        if electron_count == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Generate coordinates with quantum number-dependent scaling
        r_scale = n * (1 + 0.5 * l)  # Scale radial distribution based on n and l
        r, theta, phi = self.generate_grid(n, Z, num_points)
        r = r * r_scale

        if magnetic_field != 0:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, magnetic_field)
        else:
            density = self.probability_density(n, l, m, Z, r, theta, phi)

        # Scale density by electron count while preserving orbital structure
        density *= electron_count
        density = density / np.max(density) if np.max(density) > 0 else density

        # Apply angular momentum dependent threshold
        l_dependent_threshold = threshold * (1 - 0.1 * l)  # Lower threshold for higher angular momentum
        mask = density >= l_dependent_threshold

        # Add random sampling to prevent overcrowding
        random_mask = np.random.random(len(density[mask])) < (1.0 / (1 + 0.1 * electron_count))

        x, y, z = self.convert_cartesian(r, theta, phi)
        x, y, z = x[mask], y[mask], z[mask]
        density = density[mask]

        # Apply random sampling
        x, y, z = x[random_mask], y[random_mask], z[random_mask]
        density = density[random_mask]

        return x, y, z, density

    def visualize_atom(self, atomic_number):
        points_plotted = 0
        configuration = self.atomic_config.get_configuration(atomic_number)

        # Store orbitals for layered visualization
        orbital_data = []

        for orbital, m_dict in configuration.items():
            n = int(orbital[0])
            l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
            l = l_dict[orbital[1]]

            for m, electron_count in m_dict.items():
                x, y, z, density = self.calculate_orbital_points(
                    n, l, m, atomic_number,
                    electron_count,
                    threshold=self.visual_dict["prob_threshold"],
                    num_points=self.visual_dict["num_points"]
                )

                if len(x) > 0:
                    orbital_data.append({
                        'x': x, 'y': y, 'z': z, 'density': density,
                        'n': n, 'l': l, 'm': m
                    })
                    points_plotted += len(x)

        # Visualize orbitals from innermost to outermost
        for data in sorted(orbital_data, key=lambda x: (x['n'], x['l'])):
            self.add_orbital_visualization(
                data['x'], data['y'], data['z'],
                data['density'],
                data['n'], data['l'], data['m']
            )

        if self.visual_dict["field_lines"]:
            self.add_magnetic_field_lines()

    def add_orbital_visualization(self, x, y, z, density, n, l, m):
        """
        Enhanced visualization for individual orbitals
        """
        colors = self.get_density_color(density)

        if self.visual_dict["point_size"] == "dynamic":
            # Scale point size based on quantum numbers
            base_size = self.visual_dict["point_size"]
            n_scale = 1 + 0.2 * (n - 1)  # Larger for higher n
            l_scale = 1 - 0.1 * l  # Smaller for higher l
            point_size = base_size * n_scale * l_scale
        else:
            point_size = self.visual_dict["point_size"]

        scatter = scene.visuals.Markers()
        scatter.set_gl_state(
            self.visual_dict["state"],
            depth_test=self.visual_dict["not_see_inside"],
            blend=self.visual_dict["blend"]
        )
        scatter.set_data(
            np.column_stack((x, y, z)),
            face_color=colors,
            size=point_size
        )
        self.view.add(scatter)
    def add_magnetic_field_lines(self):
        if self.visual_dict["magnetic_field"] != 0:
            # Parameter für die Feldlinien
            grid_size = 8  # Anzahl der Linien in jede Richtung (x und y)
            spacing = 0.6  # Abstand zwischen den Linien
            height = 16  # Höhe der Feldlinien

            # Berechne den Startpunkt für das Gitter, um es zu zentrieren
            start_x = -(grid_size - 1) * spacing / 2
            start_y = -(grid_size - 1) * spacing / 2

            # Erstelle Gitter von Feldlinien
            for i in range(grid_size):
                for j in range(grid_size):
                    # Berechne x und y Position für jede Linie
                    x = start_x + (i * spacing)
                    y = start_y + (j * spacing)

                    # Erstelle Linienpunkte
                    pos = np.array([
                        [x, y, -height / 2],  # Startpunkt
                        [x, y, height / 2]  # Endpunkt
                    ])

                    # Erstelle Pfeil
                    arrow = scene.visuals.Arrow(
                        pos=pos,
                        color=(0.5, 0.5, 1.0, 0.6),  # Hellblau, halbtransparent
                        width=2,
                        arrow_size=10,
                        arrow_type="stealth",
                        parent=self.view.scene
                    )
    def visualize_atom(self, atomic_number):
        points_plotted = 0
        configuration = self.atomic_config.get_configuration(atomic_number)

        #print(f"Elektronenkonfiguration für Z={atomic_number}:")
        l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

        for orbital, m_dict in configuration.items():
            n = int(orbital[0])
            l = l_dict[orbital[1]]

            total_electrons = sum(m_dict.values())
            #print(f"{orbital}: {total_electrons} Elektronen")
            #print(f"  m-Werte: {dict(m_dict)}")

            for m, electron_count in m_dict.items():
                points_plotted = self.add_orbital(n, l, m, atomic_number, electron_count, points_plotted)
                print(f"n: {self.calculate_orbitals_energy(n, electron_count)}")
                print(f"in magnetfeld(m={m} {self.calculate_orbitals_energy_magneticfield(n, m, 10, Z=electron_count)}")

        #self.add_legend(points_plotted=points_plotted)
        if self.visual_dict["field_lines"]:
            self.add_magnetic_field_lines()

        if self.visual_dict["info"]:
            self.info_window = self.add_info_window(points_plotted=points_plotted)


    def add_info_window(self, points_plotted):
        # Create new canvas for info
        width = 400
        height = 400
        info_canvas = scene.SceneCanvas(size=(width, height), show=True)
        info_canvas.title = "Orbital Visualisierung - Info"

        grid = info_canvas.central_widget.add_grid()

        info_text = (
            '=== Elektronendichte ===\n\n'
            'Farbskala:\n'
            'ROT    = Hohe Dichte\n'
            'GRÜN   = Mittlere Dichte\n'
            'BLAU   = Niedrige Dichte\n\n'
            'Statistik:\n'
            f'Punkte:       {points_plotted:,}\n'
            f'Schwellenwert: {self.visual_dict["prob_threshold"]:.2f}\n\n'
        )

        text = scene.visuals.Text(
            info_text,
            color='white',
            pos=(width/2, height/2),
            font_size=12,
            italic=True,
            bold=False,
            parent=grid
        )

        grid.bgcolor = '#000000'

        return info_canvas


class RadialProbabilityPlotter(GeneralFunctions):
    def __init__(self):
        # Dummy visual_dict for parent class
        super().__init__(visual_dict={})

    def calculate_rpd(self, n, l, Z, r):
        """
        Berechnet die radiale Wahrscheinlichkeitsverteilung
        P(r) = r^2 * |R(r)|^2
        """
        R = self.radial_function(n, l, r, Z)
        return r ** 2 * np.abs(R) ** 2

    def plot_rpd(self, configurations, Z, r_max=20, num_points=1000):
        """
        Plottet die RPD für verschiedene (n,l) Konfigurationen

        Parameters:
        -----------
        configurations : list of tuples
            Liste von (n,l) Paaren
        Z : int
            Kernladungszahl
        r_max : float
            Maximaler Radius für Plot
        num_points : int
            Anzahl der Punkte für die r-Achse
        """
        r = np.linspace(0, r_max, num_points)

        plt.figure(figsize=(12, 8))

        for n, l in configurations:
            rpd = self.calculate_rpd(n, l, Z, r)
            #rpd = rpd / np.max(rpd)
            plt.plot(r, rpd, label=f'n={n}, l={l}')

        plt.title(f'Radiale Wahrscheinlichkeitsverteilung (Z={Z})')
        plt.xlabel('r (Bohr)')
        plt.ylabel('P(r) (un-normiert)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


class MagneticRadialProbabilityPlotter(GeneralFunctions):
    def __init__(self):
        super().__init__(visual_dict={})

    def calculate_rpd_magnetic(self, n, l, m, Z, r, theta, B_field):
        """
        Berechnet die radiale Wahrscheinlichkeitsverteilung im Magnetfeld unter
        Berücksichtigung der Störungstheorie erster Ordnung.

        Parameters:
        -----------
        n, l, m : int
            Quantenzahlen
        Z : int
            Kernladungszahl
        r : array
            Radiusvektor
        B_field : float
            Magnetfeldstärke in Tesla

        Returns:
        --------
        rpd : array
            Radiale Wahrscheinlichkeitsverteilung mit Magnetfeldkorrektur
        """
        # Basis-Radialfunktion
        R_0 = self.radial_function(n, l, r, Z)

        # Störungsterm für das Magnetfeld
        # Zeeman-Term (linear in B)
        zeeman_correction = self.mu_B * B_field * m

        # Diamagnetischer Term (quadratisch in B)
        # Hier müssen wir r² in Kugelkoordinaten ausdrücken
        r_squared = r ** 2 * np.sin(theta) ** 2  # nur x-y Ebene für B || z
        diamagnetic_correction = (self.e ** 2 * B_field ** 2 / (8 * self.m_e)) * r_squared

        # Gesamte Störung
        total_correction = zeeman_correction + diamagnetic_correction

        # Korrigierte Radialfunktion (Störungstheorie 1. Ordnung)
        R_corrected = R_0 * (1 + total_correction)

        # Radiale Wahrscheinlichkeitsverteilung
        rpd = r ** 2 * np.abs(R_corrected) ** 2

        return rpd

    def plot_rpd_magnetic_comparison(self, n, l, m, Z, B_fields, r_max=10, num_points=100000):
        """
        Vergleicht die RPD für verschiedene Magnetfeldstärken mit störungstheoretischer Korrektur
        """
        # Erzeuge r-Gitter und Winkelkoordinaten
        r = np.linspace(0, r_max, num_points)
        theta = np.pi / 2  # Fixiere theta für B || z

        plt.figure(figsize=(15, 10))

        # Berechne und plotte ungestörten Fall
        rpd_0 = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, 0)
        max_rpd = np.max(rpd_0)  # Normierung auf B=0 Fall
        plt.plot(r, rpd_0 / max_rpd, label='B = 0 T', linestyle='--', color='black')

        # Berechne und plotte für verschiedene Feldstärken
        colors = plt.cm.viridis(np.linspace(0, 1, len(B_fields)))
        for B, color in zip(B_fields, colors):
            rpd = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, B)
            rpd_norm = rpd / max_rpd
            plt.plot(r, rpd_norm, label=f'B = {B} T', color=color)

            # Berechne Energieverschiebung
            E_shift = self.calculate_orbitals_energy_magneticfield(n, m, B, Z) - \
                      self.calculate_orbitals_energy(n, Z)
            print(f"Energieverschiebung für B = {B}T: {E_shift:.6f} eV")

        plt.title(f'Radiale Wahrscheinlichkeitsverteilung im Magnetfeld\n'
                  f'(n={n}, l={l}, m={m}, Z={Z})')
        plt.xlabel('r (Bohr)')
        plt.ylabel('P(r) (normiert)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.05, 1.05)

        plt.show()


class MagneticEnergyDifferencePlotter(GeneralFunctions):
    def __init__(self):
        super().__init__(visual_dict={})

    def calculate_energies(self, n, Z, b_field):
        l = n-1
        energies = {}

        for m in range(-l, l+1):
            E = self.calculate_orbitals_energy_magneticfield(n, m, b_field, Z)
            energies[m] = E

        return energies

    def plot_energy_levels(self, energies_dict, B_field=50):
        """
        Erstellt ein übersichtliches Energieniveaudiagramm für den Zeeman-Effekt

        Parameters:
        -----------
        energies_dict : dict
            Dictionary mit m_l als Schlüssel und Energiewerten in eV als Werte
        B_field : float
            Magnetfeldstärke in Tesla
        """
        # Style-Einstellungen
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "grid.alpha": 0.3
        })

        # Figure erstellen
        height = len(energies_dict.keys())
        l = max(energies_dict.keys())
        width = 8 if l > 2 else 7

        fig, ax = plt.subplots(figsize=(width, height*2))

        # Konstanten
        line_length = 0.3
        text_offset = 0.35
        energy_text_offset = 0.8



        # Energieniveaus zeichnen
        sorted_items = sorted(energies_dict.items(), key=lambda x: x[1], reverse=True)

        # Y-Achsen-Bereich berechnen
        energy_values = list(energies_dict.values())
        energy_range = max(energy_values) - min(energy_values)
        y_padding = energy_range * 0.1

        for m, E in sorted_items:
            # Hauptlinie
            ax.hlines(y=E, xmin=-line_length, xmax=line_length,
                      color='blue', linewidth=2, alpha=0.8)

            # Energiebezeichnung
            label = fr"$E_{{{l},{m}}}$"
            ax.text(text_offset, E, label,
                    verticalalignment='center', fontsize=12)

            # Energiewert - weiter nach rechts verschoben und präziser formatiert
            ax.text(energy_text_offset, E,
                    f"${E:.6f}\,\mathrm{{eV}}$",
                    verticalalignment='center', fontsize=10, color='gray')

        # Kern zeichnen
        kern_position = min(energy_values) - y_padding / 2
        ax.plot(0, kern_position, 'o', color='black',
                markersize=12, label="Kern")
        ax.text(text_offset, kern_position, "Kern",
                verticalalignment='center', fontsize=12)

        # Magnetfeld-Annotation - nach rechts verschoben
        if B_field > 0:
            ax.text(1.2, np.mean(energy_values),
                    fr"$B = {B_field}\,\mathrm{{T}}$",
                    rotation=90, fontsize=12)

        # Achsen konfigurieren
        ax.set_ylabel(r"Energie $E_{lm_{l}}$ (eV)")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xlim(-0.8, 1.4)
        ax.set_ylim(kern_position - y_padding / 2, max(energy_values) + y_padding / 2)

        # Y-Achsen-Ticks anpassen
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))

        # Titel
        ax.set_title(f"Energieniveaudiagramm: Zeeman-Effekt ($l={l}$)", pad=20)

        # Zusätzliche Informationen
        fig.text(0.02, 0.02,
                 r"$\Delta E = \mu_B g_J B m_J$",
                 fontsize=10)

        # Layout optimieren
        plt.tight_layout()

        plt.show()


def plot_example_magnetic_rpd():
    plotter = MagneticRadialProbabilityPlotter()
    n, l, m = 2, 1, 1
    # n, l, m = 2, 1, 0 --> BEI M=0 BLEIBT DAS ORBITALGLEICH DA MAGNETFELD ENTLANG Z ACHSE GEHT!!!!!
    #n, l, m = 2, 1, -1
    configs = [
        (2, 1, 1),
        (2, 1, 0),
        (2, 1, -1),
               ]
    Z = 1
    B_fields = [1_000, 10_000, 100_0000, 1_000_000, 10_000_000]

    for config in configs:
        plotter.plot_rpd_magnetic_comparison(config[0], config[1], config[2], Z, B_fields)


# Beispielnutzung
def plot_example_rpd():
    plotter = RadialProbabilityPlotter()

    configs = [
        (3, 0),  # 3s
        (3, 1),  # 3p
        (3, 2)   # 3d
    ]

    plotter.plot_rpd(configs, Z=1, r_max=40)


def plot_energy_difference():
    plotter = MagneticEnergyDifferencePlotter()
    energies = plotter.calculate_energies(2, 1, -50000)
    plotter.plot_energy_levels(energies)


def main():
    electrons = int(input("elektronenzahl: "))
    #plot_example_rpd()
    #plot_example_magnetic_rpd()
    #plot_energy_difference()

    visual_dict = {
        "angle": 50,
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.3,
        "num_points": 600000,
        "magnetic_field": 1000,
        "field_lines": False,
        "info": False,
        "Störtheorie Näherung": False
    }
    visualizer = ElectronDensityVisualizer(visual_dict=visual_dict)
    visualizer.visualize_atom(electrons)



    print("\nVisualisierung gestartet...")
    print("Steuerung:")
    print("- Linke Maustaste: Rotation")
    print("- Rechte Maustaste: Zoom")
    print("- Mittlere Maustaste: Pan")
    app.run()


if __name__ == '__main__':
    main()
