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

        density *= electron_count / (2.0 * np.sum(density) / num_points)

        #density = density + np.max(density) * 0.01
        max_density = np.max(density)
        if max_density > 1e-10:
            density = density / max_density
        else:
            density = np.ones_like(density) * 0.1

        density = np.nan_to_num(density, nan=0.1, posinf=1.0, neginf=0.0)
        #density = np.clip(density, 0.01, 1)

        # Apply threshold
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
        r2_expect = (n ** 2 * (5 * n ** 2 + 1)) * (0.529e-10) ** 2  # in m²
        dia_energy = (self.e ** 2 * b_field ** 2 / (8 * self.m_e)) * r2_expect

        return base_energy + zeeman_energy + dia_energy

    def scaling_factor_magnetic_field(self, E_new, E_base):
        return np.sqrt(abs(E_new)/abs(E_base))

    def first_order_apporoximation_old(self, n, l, r, Z, m, theta, phi, field):
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

    def first_order_apporoximation(self, n, l, r, Z, m, theta, phi, field):
        """
        Vereinfachte Näherung für den Zeeman-Effekt mit Normierung.

        Parameters:
        -----------
        n, l, m : int
            Quantenzahlen
        r, theta, phi : array
            Kugelkoordinaten
        Z : int
            Kernladungszahl
        field : float
            Magnetfeldstärke
        """
        # Berechne ungestörte Wellenfunktion
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        psi_0 = R_nl * Y_lm

        # Berechne Störterme
        # Paramagnetischer Term
        H_para = self.mu_B * field * m

        # Diamagnetischer Term (In Kugelkoordinaten) (x² + y² = r²sin²(θ))
        r_perp_squared = r * r * np.sin(theta) * np.sin(theta)
        H_dia = (self.e ** 2 * field ** 2 / (8 * self.m_e)) * r_perp_squared

        # Kombinierte gestörte Wellenfunktion (Zähler)
        psi = psi_0 * (1 + H_para + H_dia)

        # Normierung (Nenner)
        # Volumenelement in Kugelkoordinaten
        dV = r ** 2 * np.sin(theta)

        # Berechne Normierungsfaktor
        norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))

        # Normiere die Wellenfunktion
        if norm > 0:
            psi = psi / norm

        return psi

    def total_correction_up_to_second_order(self, n, l, m, B, r, theta, phi):
        """
        Berechnet die Wellenfunktion mit Korrekturen bis zur zweiten Ordnung
        """
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        psi_1 = self.first_order_correction(n, l, m, B, r, theta, phi) - psi_0  # Nur die Korrektur
        psi_2 = self.second_order_correction(n, l, m, B, r, theta, phi) - psi_0  # Nur die Korrektur

        # Kombiniere alle Beiträge
        psi_total = psi_0 + psi_1 + psi_2

        # Normierung
        dV = r ** 2 * np.sin(theta)
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))
        if norm > 0:
            psi_total = psi_total / norm

        return psi_total

    def calculate_matrix_element(self, psi_k, psi_0, B, r, theta, phi, m):
        # Zeeman Term mit m-Abhängigkeit
        zeeman = self.mu_B * B * m * np.conjugate(psi_k) * psi_0

        # Diamagnetischer Term
        # x² + y² = r²sin²θ
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        r_perp_squared = x ** 2 + y ** 2

        dia = (self.e ** 2 * B ** 2 / (8 * self.m_e)) * \
              np.conjugate(psi_k) * r_perp_squared * psi_0

        return zeeman + dia

    def first_order_correction(self, n, l, m, B, r, theta, phi):
        """
        First-order perturbation theory implementation für 1D Arrays
        """
        a0 = 5.29177e-11
        B_atomic = B * (a0 ** 2 * self.e / self.hbar)
        print(f"B_atomic: {B_atomic}")

        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        E_n = self.calculate_orbitals_energy(n, 1)

        psi_1 = np.zeros_like(psi_0, dtype=complex)

        dV = r ** 2 * np.sin(theta)

        contributions = []

        for n_prime in range(max(1, n - 2), n + 3):
            for l_prime in range(max(0, l - 1), min(n_prime, l + 2)):
                for m_prime in range(-l_prime, l_prime + 1):
                    if (n_prime, l_prime, m_prime) == (n, l, m):
                        continue

                    try:
                        psi_k = self.wave_func(n_prime, l_prime, m_prime, 1, r, theta, phi)
                        E_k = self.calculate_orbitals_energy(n_prime, 1)

                        H_k0 = self.calculate_matrix_element(psi_k, psi_0, B_atomic, r, theta, phi, m)
                        integral = np.sum(H_k0 * dV)

                        delta_E = E_k - E_n
                        if abs(delta_E) > 1e-10:
                            contribution = (integral / delta_E) * psi_k
                            psi_1 += contribution

                            contributions.append({
                                'state': (n_prime, l_prime, m_prime),
                                'magnitude': np.abs(integral / delta_E)
                            })

                            print(f"\nContribution from state ({n_prime},{l_prime},{m_prime}):")
                            print(f"Matrix element: {integral}")
                            print(f"Energy difference: {delta_E}")
                            print(f"Contribution magnitude: {np.abs(integral / delta_E)}")

                    except Exception as e:
                        print(f"Exception for state ({n_prime},{l_prime},{m_prime}): {str(e)}")
                        continue

        if contributions:
            print("\nLargest contributions:")
            for contrib in sorted(contributions, key=lambda x: x['magnitude'], reverse=True)[:5]:
                print(f"State {contrib['state']}: {contrib['magnitude']}")

        psi_total = psi_0 + psi_1
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))
        if norm > 0:
            psi_total = psi_total / norm
            print(f"\nFinal normalization factor: {norm}")

        return psi_total


    def check_normalization(self, n, l, r, Z, m, theta, phi, field):
        """
        Überprüft die Normierung der Wellenfunktion durch Integration
        der Wahrscheinlichkeitsdichte über den gesamten Raum.
        """
        # Berechne Wellenfunktion
        psi = self.first_order_approximation(n, l, r, Z, m, theta, phi, field)

        # Wahrscheinlichkeitsdichte
        density = np.abs(psi) ** 2

        # Volumenelement in Kugelkoordinaten
        dV = r ** 2 * np.sin(theta)

        # Berechne Integral über gesamten Raum
        total_prob = np.sum(density * dV)

        print(f"Gesamtwahrscheinlichkeit: {total_prob}")
        print(f"Abweichung von 1: {abs(1 - total_prob)}")

        return total_prob

    def wave_func_magneticfield(self, n, l, r, Z, m, theta, phi, field, approx):
        if approx:
            print("approx")
            # Unnormierte Version (alte Implementierung)
            psi_old = self.first_order_apporoximation_old(n, l, r, Z, m, theta, phi, field)
            density_old = np.abs(psi_old) ** 2

            # Normierte Version (neue Implementierung)
            psi_new = self.total_correction_up_to_second_order(n, l, r, Z, m, theta, phi, field)
            density_new = np.abs(psi_new) ** 2

            # Volumenelement
            dV = r ** 2 * np.sin(theta)

            # Integrale berechnen
            total_old = np.sum(density_old * dV)
            total_new = np.sum(density_new * dV)

            print("Vergleich der Normierung:")
            print(f"Alte Version - Gesamtwahrscheinlichkeit: {total_old}")
            print(f"Neue Version - Gesamtwahrscheinlichkeit: {total_new}")

            # Maximale Werte
            max_old = np.max(density_old)
            max_new = np.max(density_new)

            print(f"\nMaximale Dichte:")
            print(f"Alte Version: {max_old}")
            print(f"Neue Version: {max_new}")

            return psi_new

        else:
            print("no approx")
            psi_1 = self.first_order_correction(n, l, m, field, r, theta, phi)
            return psi_1

    def probability_density_magnetic_field(self, n, l, r, Z, m, theta, phi, field):
        psi = self.wave_func_magneticfield(n, l, r, Z, m, theta, phi, field,
                                           approx=self.visual_dict["Störtheorie Näherung"])

        density = np.abs(psi) ** 2

        dV = r ** 2 * np.sin(theta)
        norm = np.sum(density * dV)

        if norm > 1e-10:
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
        self.view.camera.distance = 3
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
        if electron_count == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])


        r_scale = n * (1 + 0.5 * l)
        r, theta, phi = self.generate_grid(n, Z, num_points)
        r = r * r_scale

        if magnetic_field != 0:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, magnetic_field)
        else:
            density = self.probability_density(n, l, m, Z, r, theta, phi)

        density *= electron_count
        density = density / np.max(density) if np.max(density) > 0 else density

        l_dependent_threshold = threshold * (1 - 0.1 * l)  # Lower threshold for higher angular momentum
        mask = density >= l_dependent_threshold

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

        for data in sorted(orbital_data, key=lambda x: (x['n'], x['l'])):
            self.add_orbital_visualization(
                data['x'], data['y'], data['z'],
                data['density'],
                data['n'], data['l'], data['m']
            )

        if self.visual_dict["field_lines"]:
            self.add_magnetic_field_lines()

    def add_orbital_visualization(self, x, y, z, density, n, l, m):

        colors = self.get_density_color(density)

        if self.visual_dict["point_size"] == "dynamic":
            base_size = self.visual_dict["point_size"]
            n_scale = 1 + 0.2 * (n - 1)
            l_scale = 1 - 0.1 * l
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

    def calculate_rpd_magnetic_directional(self, n, l, m, Z, r, theta, B_field):
        """
        Berechnet die richtungsabhängige radiale Wahrscheinlichkeitsverteilung im Magnetfeld
        """
        # Basis-Wellenfunktion
        psi_0 = self.wave_func(n, l, m, Z, r, theta, np.zeros_like(theta))

        # Paramagnetischer Term (nur in x-y-Ebene wirksam)
        H_para = self.mu_B * B_field * m * np.sin(theta)

        # Diamagnetischer Term (r_perp = r*sin(theta))
        r_perp_squared = r ** 2 * np.sin(theta) ** 2
        H_dia = (self.e ** 2 * B_field ** 2 / (8 * self.m_e)) * r_perp_squared

        # Gestörte Wellenfunktion mit richtungsabhängiger Korrektur
        psi = psi_0 * (1 + H_para + H_dia)

        # Normierung
        dV = r ** 2 * np.sin(theta)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))
        if norm > 0:
            psi = psi / norm

        # Radiale Wahrscheinlichkeitsverteilung
        rpd = r ** 2 * np.abs(psi) ** 2

        return rpd

    def calculate_rpd_magnetic(self, n, l, m, Z, r, theta, B_field):
        """
        Berechnet die radiale Wahrscheinlichkeitsverteilung im Magnetfeld
        mit der verbesserten normierten Störungstheorie
        """
        # Basis-Wellenfunktion
        psi_0 = self.wave_func(n, l, m, Z, r, theta, np.zeros_like(theta))

        # Störterme
        # Paramagnetischer Term
        H_para = self.mu_B * B_field * m

        # Diamagnetischer Term
        r_perp_squared = r ** 2 * np.sin(theta) ** 2
        H_dia = (self.e ** 2 * B_field ** 2 / (8 * self.m_e)) * r_perp_squared

        # Gestörte Wellenfunktion
        psi = psi_0 * (1 + H_para + H_dia)

        # Normierung
        dV = r ** 2 * np.sin(theta)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))
        if norm > 0:
            psi = psi / norm

        # Radiale Wahrscheinlichkeitsverteilung
        rpd = r ** 2 * np.abs(psi) ** 2

        return rpd

    def calculate_rpd_magnetic_direction_full_first_order(self, n, l, m, Z, r, theta, B_field):
        # Debug der Array-Shapes
        print(f"Shape of r: {r.shape}")
        print(f"Shape of theta: {theta.shape}")

        # Erstellen der korrekten Meshgrids
        r_mesh, theta_mesh = np.meshgrid(r, theta, indexing='ij')
        print(f"Shape of r_mesh: {r_mesh.shape}")
        print(f"Shape of theta_mesh: {theta_mesh.shape}")

        phi = np.zeros_like(theta)  # Behalten ich bei

        psi_corrected = self.first_order_correction(n, l, m, B_field, r, theta, phi)
        psi_0 = self.wave_func(n, l, m, Z, r, theta, phi)

        print(f"Shape of psi_corrected: {psi_corrected.shape}")

        # Berechnung von dV mit korrektem Broadcasting
        dV = r_mesh ** 2 * np.sin(theta_mesh)
        print(f"Shape of dV: {dV.shape}")

        # Vergleich mit korrektem Broadcasting
        if np.allclose(psi_corrected, psi_0, atol=1e-8):
            print("BITTE NICHT!!!!!! Die Wellenfunktionen sind (fast) identisch.")
        else:
            print("Danke! Die Wellenfunktionen sind unterschiedlich.")

            # Normierung mit korrektem Broadcasting
            norm_psi_corrected = np.sqrt(np.sum(np.abs(psi_corrected) ** 2 * dV))
            norm_psi_0 = np.sqrt(np.sum(np.abs(psi_0) ** 2 * dV))

            print(f"Norm von psi_corrected: {norm_psi_corrected}")
            print(f"Norm von psi_0: {norm_psi_0}")

            if not np.isclose(norm_psi_corrected, 1, atol=1e-8):
                print("WARNUNG: psi_corrected ist nicht normiert!")
            if not np.isclose(norm_psi_0, 1, atol=1e-8):
                print("WARNUNG: psi_0 ist nicht normiert!")

            difference = psi_corrected - psi_0
            max_diff = np.max(np.abs(difference))
            mean_diff = np.mean(np.abs(difference))
            print(f"Maximaler Unterschied: {max_diff}")
            print(f"Mittlerer Unterschied: {mean_diff}")

            relative_diff = np.abs(difference / (psi_0 + 1e-10))
            max_rel_diff = np.max(relative_diff)
            print(f"Maximale relative Abweichung: {max_rel_diff}")

        # RPD Berechnung mit korrektem Broadcasting
        rpd = r_mesh ** 2 * np.abs(psi_corrected) ** 2

        return rpd

    def plot_first_order_wavefunction(self, n, l, m, Z, B_field, r_max=20, num_points=1000):
        """
        Eine separate Funktion zur Visualisierung der Wellenfunktion erster Ordnung
        """
        # Erstelle Gitter
        r = np.linspace(0, r_max, num_points)
        theta_xy = np.pi / 2 * np.ones_like(r)  # x-y Ebene
        theta_z = np.zeros_like(r)  # z-Richtung
        phi = np.zeros_like(r)

        # Berechne ungestörte Wellenfunktionen
        #psi_0_xy = self.wave_func(n, l, m, Z, r, theta_xy, phi)
        #psi_0_z = self.wave_func(n, l, m, Z, r, theta_z, phi)
        psi_0_xy = self.total_correction_up_to_second_order(n, l, m, 0, r, theta_xy, phi)
        psi_0_z = self.total_correction_up_to_second_order(n, l, m, 0, r, theta_z, phi)


        # Berechne gestörte Wellenfunktionen

        #psi_B_xy = self.total_correction_up_to_second_order(n, l, m, B_field, r, theta_xy, phi)
        #psi_B_z = self.total_correction_up_to_second_order(n, l, m, B_field, r, theta_z, phi)
        psi_B_xy = self.first_order_correction(n, l, m, B_field, r, theta_xy, phi)
        psi_B_z = self.first_order_correction(n, l, m, B_field, r, theta_z, phi)


        # Berechne Wahrscheinlichkeitsdichten
        density_0_xy = r ** 2 * np.abs(psi_0_xy) ** 2
        density_B_xy = r ** 2 * np.abs(psi_B_xy) ** 2
        density_0_z = r ** 2 * np.abs(psi_0_z) ** 2
        density_B_z = r ** 2 * np.abs(psi_B_z) ** 2

        # Normierung
        max_density = max([np.max(density_0_xy), np.max(density_B_xy),
                           np.max(density_0_z), np.max(density_B_z)])

        density_0_xy /= max_density
        density_B_xy /= max_density
        density_0_z /= max_density
        density_B_z /= max_density

        # Erstelle Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot für x-y Ebene
        ax1.plot(r, density_0_xy, 'b--', label='B = 0 T', linewidth=2)
        ax1.plot(r, density_B_xy, 'r-', label=f'B = {B_field} T', linewidth=2)
        ax1.set_title(f'x-y-Ebene (θ = π/2)\nn={n}, l={l}, m={m}')
        ax1.set_xlabel('Radius (Bohr)')
        ax1.set_ylabel('Normierte Wahrscheinlichkeitsdichte')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot für z-Richtung
        ax2.plot(r, density_0_z, 'b--', label='B = 0 T', linewidth=2)
        ax2.plot(r, density_B_z, 'r-', label=f'B = {B_field} T', linewidth=2)
        ax2.set_title(f'z-Richtung (θ = 0)\nn={n}, l={l}, m={m}')
        ax2.set_xlabel('Radius (Bohr)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Debug-Ausgaben
        print(f"Maximum Werte:")
        print(f"xy-Ebene: B=0: {np.max(density_0_xy):.6f}, B={B_field}: {np.max(density_B_xy):.6f}")
        print(f"z-Richtung: B=0: {np.max(density_0_z):.6f}, B={B_field}: {np.max(density_B_z):.6f}")
        print(f"\nIntegrierte Wahrscheinlichkeiten:")
        print(f"xy-Ebene: B=0: {np.trapz(density_0_xy, r):.6f}, B={B_field}: {np.trapz(density_B_xy, r):.6f}")
        print(f"z-Richtung: B=0: {np.trapz(density_0_z, r):.6f}, B={B_field}: {np.trapz(density_B_z, r):.6f}")

        # Setze vernünftige Grenzen
        for ax in [ax1, ax2]:
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, r_max)

        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)

    def plot_rpd_magnetic_directional(self, n, l, m, Z, B_field, approx=True, r_max=10, num_points=1000):
        """
        Erstellt separate Plots für x-y-Ebene und z-Richtung
        """
        r = np.linspace(0, r_max, num_points)

        # Winkel für x-y-Ebene (θ = π/2) und z-Richtung (θ = 0)
        theta_xy = np.pi / 2 * np.ones_like(r)
        theta_z = np.zeros_like(r)

        # Berechne RPD mit existierender Methode
        if approx:
            rpd_0_xy = self.calculate_rpd_magnetic_directional(n, l, m, Z, r, theta_xy, 0)
            rpd_B_xy = self.calculate_rpd_magnetic_directional(n, l, m, Z, r, theta_xy, B_field)
            rpd_0_z = self.calculate_rpd_magnetic_directional(n, l, m, Z, r, theta_z, 0)
            rpd_B_z = self.calculate_rpd_magnetic_directional(n, l, m, Z, r, theta_z, B_field)
        else:
            # Nutze die full-first-order Methode
            rpd_0_xy = self.calculate_rpd_magnetic_direction_full_first_order(n, l, m, Z, r, theta_xy, 0)
            rpd_B_xy = self.calculate_rpd_magnetic_direction_full_first_order(n, l, m, Z, r, theta_xy, B_field)
            rpd_0_z = self.calculate_rpd_magnetic_direction_full_first_order(n, l, m, Z, r, theta_z, 0)
            rpd_B_z = self.calculate_rpd_magnetic_direction_full_first_order(n, l, m, Z, r, theta_z, B_field)

        # Debug-Ausgabe
        print(f"RPD Shapes - xy: {rpd_0_xy.shape}, z: {rpd_0_z.shape}")
        print(f"RPD Max Values - xy_0: {np.max(rpd_0_xy)}, xy_B: {np.max(rpd_B_xy)}")
        print(f"RPD Max Values - z_0: {np.max(rpd_0_z)}, z_B: {np.max(rpd_B_z)}")

        # Wenn die RPDs 2D sind, nehmen wir die relevante Dimension
        if rpd_0_xy.ndim > 1:
            rpd_0_xy = np.mean(rpd_0_xy, axis=1)  # oder axis=0, je nach Struktur
            rpd_B_xy = np.mean(rpd_B_xy, axis=1)
            rpd_0_z = np.mean(rpd_0_z, axis=1)
            rpd_B_z = np.mean(rpd_B_z, axis=1)

        # Normierung auf das Maximum aller Kurven
        max_rpd = max([np.max(rpd_0_xy), np.max(rpd_B_xy),
                       np.max(rpd_0_z), np.max(rpd_B_z)])

        if max_rpd > 0:
            rpd_0_xy = rpd_0_xy / max_rpd
            rpd_B_xy = rpd_B_xy / max_rpd
            rpd_0_z = rpd_0_z / max_rpd
            rpd_B_z = rpd_B_z / max_rpd

        # Erstelle Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot für x-y-Ebene
        ax1.plot(r, rpd_0_xy, 'b--', label='B = 0 T', linewidth=2)
        ax1.plot(r, rpd_B_xy, 'r-', label=f'B = {B_field} T', linewidth=2)
        ax1.set_title(f'x-y-Ebene (θ = π/2)\nn={n}, l={l}, m={m}')
        ax1.set_xlabel('Radius (Bohr)')
        ax1.set_ylabel('Normierte Dichte')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot für z-Richtung
        ax2.plot(r, rpd_0_z, 'b--', label='B = 0 T', linewidth=2)
        ax2.plot(r, rpd_B_z, 'r-', label=f'B = {B_field} T', linewidth=2)
        ax2.set_title(f'z-Richtung (θ = 0)\nn={n}, l={l}, m={m}')
        ax2.set_xlabel('Radius (Bohr)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Einheitliche Formatierung
        for ax in [ax1, ax2]:
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, r_max)

        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)

    def plot_rpd_magnetic_comparison(self, n, l, m, Z, B_fields, r_max=50, num_points=5000):
        """
        Vergleicht die RPD für verschiedene Magnetfeldstärken
        """
        # Erzeuge r-Gitter und Winkelkoordinaten
        r = np.linspace(0, r_max, num_points)
        print(r.max())
        theta = np.pi / 2

        fig = plt.figure(figsize=(12, 8))
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
        # Set style manually for better control
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.facecolor'] = '#f0f0f0'

        # Berechne und plotte ungestörten Fall
        rpd_0 = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, 0)


        max_rpd = np.max(rpd_0)  # Normierung auf B=0 Fall
        plt.plot(r, rpd_0 / max_rpd, label='B = 0 T',
                 linestyle='--', color='black', linewidth=2)

        # Berechne und plotte für verschiedene Feldstärken
        colors = plt.cm.viridis(np.linspace(0, 1, len(B_fields)))
        for B, color in zip(B_fields, colors):
            rpd = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, B)
            rpd_norm = rpd / max_rpd
            plt.plot(r, rpd_norm, label=f'B = {B} T',
                     color=color, linewidth=2)

            # Energieverschiebung berechnen
            E_shift = self.calculate_orbitals_energy_magneticfield(n, m, B, Z) - \
                      self.calculate_orbitals_energy(n, Z)
            print(f"Energieverschiebung für B = {B}T: {E_shift:.6f} eV")

        plt.title(f'Radiale Wahrscheinlichkeitsverteilung im Magnetfeld\n' +
                  f'(n={n}, l={l}, m={m}, Z={Z})',
                  fontsize=14, pad=20)
        plt.xlabel('Radius (Bohr)', fontsize=12)
        plt.ylabel('Normierte Wahrscheinlichkeitsdichte', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, framealpha=0.9)

        plt.ylim(-0.05, 1.05)

        max_relevant_r = r[np.where(rpd_0 > max_rpd * 0.01)[0][-1]]
        plt.xlim(0, max_relevant_r * 1.2)

        plt.tight_layout()

        plt.show()

    def plot_comparison_series(self, n_values, l_values, m_values, Z, B_field, r_max=10):
        """
        Erstellt eine Serie von Vergleichsplots für verschiedene Quantenzahlen
        """
        num_plots = len(n_values)
        rows = int(np.ceil(num_plots / 2))
        cols = min(2, num_plots)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (n, l, m) in enumerate(zip(n_values, l_values, m_values)):
            if i < len(axes):
                # Konfiguriere jeden Subplot
                self._plot_single_comparison(n, l, m, Z, B_field, r_max, ax=axes[i])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def _plot_single_comparison(self, n, l, m, Z, B_field, r_max, ax):
        """
        Hilfsfunktion für einzelnen Vergleichsplot
        """
        r = np.linspace(0, r_max, 1000)
        theta = np.pi / 2

        rpd_0 = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, 0)
        rpd_B = self.calculate_rpd_magnetic(n, l, m, Z, r, theta, B_field)

        max_rpd = max(np.max(rpd_0), np.max(rpd_B))

        ax.plot(r, rpd_0 / max_rpd, 'b--', label='B = 0 T')
        ax.plot(r, rpd_B / max_rpd, 'r-', label=f'B = {B_field} T')

        ax.set_title(f'n={n}, l={l}, m={m}')
        ax.set_xlabel('Radius (Bohr)')
        ax.set_ylabel('Normierte Dichte')
        ax.grid(True, alpha=0.3)
        ax.legend()


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

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))

        ax.set_title(f"Energieniveaudiagramm: Zeeman-Effekt ($l={l}$)", pad=20)

        fig.text(0.02, 0.02,
                 r"$\Delta E = \mu_B g_J B m_J$",
                 fontsize=10)

        plt.tight_layout()

        plt.show()


def plot_example_magnetic_rpd():
    plotter = MagneticRadialProbabilityPlotter()

    B_fields = [1000, 5000, 10000]
    plotter.plot_rpd_magnetic_comparison(n=2, l=1, m=1, Z=1, B_fields=B_fields)

    n_values = [1, 2, 2, 3]
    l_values = [0, 0, 1, 1]
    m_values = [0, 0, 1, 1]
    plotter.plot_comparison_series(n_values, l_values, m_values, Z=1, B_field=300)

    plotter.plot_rpd_magnetic_directional(n=2, l=0, m=0, Z=1, B_field=1010, approx=False, r_max=500, num_points=1000)

    plotter.plot_first_order_wavefunction(n=2, l=0, m=0, Z=1, B_field=80000000)



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
    plot_example_magnetic_rpd()
    #plot_energy_difference()

    print("Und?")

    visual_dict = {
        "angle": 50,
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.3,
        "num_points": 1030000,
        "magnetic_field": 300000000000,
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
