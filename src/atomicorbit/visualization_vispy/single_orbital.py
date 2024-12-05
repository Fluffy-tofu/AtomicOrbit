import numpy as np
from vispy import app, scene
from src.atomicorbit.visualization.interactive_3d_plot import orbital_wavefunction, create_orbital_data
from src.atomicorbit.visualization_vispy.atom_orbital import GeneralFunctions


class SingleOrbital(GeneralFunctions):
    def __init__(self, visual_dict):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(2000, 1200), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 5
        self.axis = scene.XYZAxis(parent=self.view.scene)
        self.visual_dict = visual_dict

    def calculate_single_orbital_points(self, n, l, m, Z, num_points=100000, threshold=0.1, magnetic_field=False):
        """
        Parameter:
        -----------
        n: int
            Hauptquantenzahl (Energieniveau)
        l: int
            Nebenquantenzahl (Orbitalform)
        m: int
            Magnetische Quantenzahl (Orbitalorientierung)
        Z: int
            Ordnungszahl (Kernladung)
        num_points: int
            Anzahl der zu generierenden Punkte zur Visualisierung
        threshold: float
            Mindestwert für die Wahrscheinlichkeitsdichte (0 bis 1)

        returns:
        --------
        tuple: (x, y, z, dichte) Koordinaten und Dichtewerte zur Visualisierung
        """


        # kugel koordinaten-system erstellen
        r, theta, phi = self.generate_grid(n, Z, num_points)

        if not magnetic_field:
            density = self.probability_density(n, l, m, Z, r, theta, phi)

        else:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, field=10000000)

        # Normalisieren
        density = density / np.max(density)

        # density = density + np.max(density) * 0.01 offset das nicht null ist

        density = np.nan_to_num(density, nan=0.1, posinf=1.0, neginf=0.0)
        density = np.clip(density, 0.01, 1)

        mask = density >= threshold

        x, y, z = self.convert_cartesian(r, theta, phi)

        return x[mask], y[mask], z[mask], density[mask]

    def calculate_optimal_Z(self, n, l, m):
        """ Berechnung von Z bei einzel Orbitalen """

        base_Z = n / 1.5

        l_factor = 1 + (l / (n * 2))

        m_factor = 1 + (abs(m) / (l + 1)) if l > 0 else 1

        Z = base_Z * l_factor * m_factor

        Z = np.clip(Z, 0.5, 5.0)

        return Z

    def visualize_atom(self, n, l, m):
        points_plotted = 0
        points_plotted = self.add_orbitals_single(n, l, m, self.calculate_optimal_Z(n, l, m),
                                                  points_plotted,
                                                  difference_wavefunctions=self.visual_dict["show_difference_wavefunctions"])

        if self.visual_dict["magnetic_field_comparison"] and not self.visual_dict["show_difference_wavefunctions"]:
            points_plotted = self.add_orbitals_single(n, l, m, self.calculate_optimal_Z(n, l, m),
                                                      points_plotted,
                                                      magnetic_field=True)

        self.info_window = self.add_info_window(points_plotted=points_plotted)

    def add_orbitals_single(self, n, l, m, Z, points_plotted, magnetic_field=False,
                            difference_wavefunctions=False):
        x, y, z, density = self.calculate_single_orbital_points(n, l, m, Z,
                                                                threshold=self.visual_dict["prob_threshold"],
                                                                num_points=self.visual_dict["num_points"])

        if magnetic_field or difference_wavefunctions:
            x, y, z, density_magnetic_field = self.calculate_single_orbital_points(n, l, m, Z,
                                                                          threshold=self.visual_dict["prob_threshold"],
                                                                          num_points=self.visual_dict["num_points"],
                                                                          magnetic_field=True)

        if difference_wavefunctions:
            try:
                density = density - density_magnetic_field
            except ValueError:
                density = density[:len(density_magnetic_field)]

                density = density - density_magnetic_field

        points_plotted += len(x)
        if len(x) > 0:
            if magnetic_field and not difference_wavefunctions:
                colors = self.get_density_color_magnetic_field(density_magnetic_field)
            elif difference_wavefunctions:
                colors = np.column_stack((1.0, 0.9, 0.0, 0.2))

            else:
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


    def add_info_window(self, points_plotted):
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


def main():
    visual_dict = {
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.1,
        "num_points": 1000000,
        "magnetic_field_comparison": False,
        "show_difference_wavefunctions": False,
    }
    n = int(input("n: "))
    l = int(input("l: "))
    m = int(input("m: "))

    visualizer = SingleOrbital(visual_dict=visual_dict)
    visualizer.visualize_atom(n, l, m)
    app.run()


if __name__ == "__main__":
    main()
