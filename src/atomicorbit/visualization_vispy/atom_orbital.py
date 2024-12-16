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
        r = np.random.exponential(scale=n ** 2 / Z, size=num_points)
        theta = np.arccos(2 * np.random.random(num_points) - 1)
        phi = 2 * np.pi * np.random.random(num_points)
        return r, theta, phi

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

    def probability_density_magnetic_field(self, n, l, r, Z, m, theta, phi, field):
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

        # Gestörte Wellenfunktion
        psi = psi_0 * (1 + correction)
        print("ATOM IN MAGNETFELD")
        # Wahrscheinlichkeitsdichte
        density = np.abs(psi) ** 2

        # Normierung
        if density.max() != 0:
            density = density / density.max()

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
        self.view.camera.distance = 5
        self.axis = scene.XYZAxis(parent=self.view.scene, )
        self.atomic_config = AtomicConfiguration()
        self.visual_dict = visual_dict
        label_distance = 1.2
        self.x_label = scene.Text("x", pos=[label_distance, 0, 0], color='red',
                                  font_size=20, parent=self.view.scene)
        self.y_label = scene.Text("y", pos=[0, label_distance, 0], color='green',
                                  font_size=20, parent=self.view.scene)

        self.z_label = scene.Text("z", pos=[0, 0, label_distance], color='blue',
                                  font_size=20, parent=self.view.scene)



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

    def calculate_rpd_magnetic(self, n, l, m, Z, r, B_field):
        """
        Berechnet die radiale Wahrscheinlichkeitsverteilung im Magnetfeld
        P(r) = r^2 * |R(r)|^2 mit magnetfeldabhängiger Skalierung
        """
        E_base = self.calculate_orbitals_energy(n, Z)
        E_new = self.calculate_orbitals_energy_magneticfield(n, m, B_field, Z)
        scaling_function = self.scaling_factor_magnetic_field(E_base=E_base, E_new=E_new)

        r_scaled = r / scaling_function
        R = self.radial_function(n, l, r_scaled, Z)
        scaling_factor = 1 / (scaling_function ** 3)

        rpd = r ** 2 * np.abs(R) ** 2 * scaling_factor
        return rpd

    def plot_rpd_magnetic_comparison(self, n, l, m, Z, B_fields, r_max=10, num_points=100000):
        """
        Vergleicht die RPD für verschiedene Magnetfeldstärken mit verbesserter Skalierung
        """
        max_B = max(B_fields)
        E_base = self.calculate_orbitals_energy(n, Z)
        E_new = self.calculate_orbitals_energy_magneticfield(n, m, max_B, Z)
        scaling = self.scaling_factor_magnetic_field(E_base=E_base, E_new=E_new)
        r_max = r_max * scaling

        r = np.linspace(0, r_max, num_points)

        plt.figure(figsize=(15, 10))

        all_rpds = []
        for B in [0] + B_fields:
            rpd = self.calculate_rpd_magnetic(n, l, m, Z, r, B)
            all_rpds.append(rpd)

        global_max = max(np.max(rpd) for rpd in all_rpds)

        plt.plot(r, all_rpds[0] / global_max,
                 label='B = 0 T', linestyle='--', color='black')

        colors = plt.cm.viridis(np.linspace(0, 1, len(B_fields)))
        for i, (B, color) in enumerate(zip(B_fields, colors)):
            rpd_norm = all_rpds[i + 1] / global_max
            plt.plot(r, rpd_norm, label=f'B = {B} T', color=color)

            E_shift = self.calculate_orbitals_energy_magneticfield(n, m, B, Z) - self.calculate_orbitals_energy(n, Z)
            print(f"Energieverschiebung für B = {B}T: {E_shift:.6f} eV")

        plt.title(f'Radiale Wahrscheinlichkeitsverteilung im Magnetfeld\n(n={n}, l={l}, m={m}, Z={Z})')
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
    plot_example_rpd()
    plot_example_magnetic_rpd()
    #plot_energy_difference()

    visual_dict = {
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.3,
        "num_points": 600000,
        "magnetic_field": 0,
        "field_lines": False,
        "info": False
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
