import numpy as np
from vispy import app, scene
from src.atomicorbit.visualization.interactive_3d_plot import orbital_wavefunction, create_orbital_data
from src.atomicorbit.visualization_vispy.atom_orbital import GeneralFunctions


class SingleOrbital(GeneralFunctions):
    def __init__(self, visual_dict):
        super().__init__(visual_dict)
        self.canvas = scene.SceneCanvas(keys='interactive', size=(2000, 1200), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 5
        self.axis = scene.XYZAxis(parent=self.view.scene)
        self.visual_dict = visual_dict

    def calculate_single_orbital_points(self, n, l, m, Z, field=0, num_points=100000,
                                        threshold=0.1, magnetic_field=False):
        """
        Improved point calculation with consistent array handling
        """
        r, theta, phi = self.generate_grid(n, Z, num_points)

        if not magnetic_field:
            density = self.probability_density(n, l, m, Z, r, theta, phi)
        else:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, field)

        # Normalize density
        max_density = np.max(density)
        if max_density > 0:
            density = density / max_density

        # Create mask and apply it consistently
        mask = density >= threshold

        # Convert to cartesian coordinates
        x, y, z = self.convert_cartesian(r, theta, phi)

        # Apply mask to all arrays simultaneously
        x = x[mask]
        y = y[mask]
        z = z[mask]
        density = density[mask]

        return x, y, z, density

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
        """
        Synchronized version of add_orbitals_single that ensures color and point arrays match exactly
        """
        # Calculate base orbital points
        x, y, z, density = self.calculate_single_orbital_points(n, l, m, Z,
                                                                threshold=self.visual_dict["prob_threshold"],
                                                                num_points=self.visual_dict["num_points"])

        if magnetic_field or difference_wavefunctions:
            x_mag, y_mag, z_mag, density_magnetic_field = self.calculate_single_orbital_points(
                n, l, m, Z,
                field=self.visual_dict["magnectic_field"],
                threshold=self.visual_dict["prob_threshold"],
                num_points=self.visual_dict["num_points"],
                magnetic_field=True
            )

            # Create unified mask for both datasets
            if difference_wavefunctions:
                min_size = min(len(x), len(x_mag))
                if min_size == 0:
                    return points_plotted

                # Truncate arrays to same size
                x = x[:min_size]
                y = y[:min_size]
                z = z[:min_size]
                density = density[:min_size]
                density_magnetic_field = density_magnetic_field[:min_size]
                density = density - density_magnetic_field

        points_plotted += len(x)

        if len(x) > 0:
            # Generate colors using the exact same density array used for points
            if magnetic_field and not difference_wavefunctions:
                colors = np.zeros((len(x), 4))
                for i in range(len(x)):
                    red = np.clip(2 - density[i] * 2, 0, 1)
                    blue = np.clip(density[i] * 2, 0, 1)
                    green = np.clip(1 - np.abs(density[i] - 0.5) * 2, 0, 1)
                    alpha = np.clip(density[i] * 0.8 + 0.2, 0, 1)
                    colors[i] = [red, green, blue, alpha]
            elif difference_wavefunctions:
                colors = np.ones((len(x), 4))
                colors[:, 1] = 0.9
                colors[:, 2] = 0
                colors[:, 3] = 0.2
            else:
                colors = np.zeros((len(x), 4))
                for i in range(len(x)):
                    red = np.clip(density[i] * 2, 0, 1)
                    blue = np.clip(2 - density[i] * 2, 0, 1)
                    green = np.clip(1 - np.abs(density[i] - 0.5) * 2, 0, 1)
                    alpha = np.clip(density[i] * 0.8 + 0.2, 0, 1)
                    colors[i] = [red, green, blue, alpha]

            state = self.visual_dict["state"]
            not_see_inside = self.visual_dict["not_see_inside"]
            blend = self.visual_dict["blend"]
            edges = self.visual_dict["edges"]
            point_size = self.visual_dict["point_size"]

            if point_size == "dynamic":
                point_size = 10 * density + 3

            edge_color = None if edges else colors

            # Verify array lengths match before creating scatter plot
            if len(colors) != len(x):
                print(f"Warning: Color array length {len(colors)} doesn't match point array length {len(x)}")
                # Ensure arrays match by truncating to shorter length
                min_len = min(len(colors), len(x))
                colors = colors[:min_len]
                x = x[:min_len]
                y = y[:min_len]
                z = z[:min_len]

            scatter = scene.visuals.Markers()
            scatter.set_gl_state(state, depth_test=not_see_inside, blend=blend)

            points = np.column_stack((x, y, z))
            scatter.set_data(
                points,
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
        "magnectic_field": 1000000000000000,
        "magnetic_field_comparison": True,
        "show_difference_wavefunctions": False,
        "Störtheorie Näherung": True
    }
    n = int(input("n: "))
    l = int(input("l: "))
    m = int(input("m: "))

    visualizer = SingleOrbital(visual_dict=visual_dict)
    visualizer.visualize_atom(n, l, m)
    app.run()


if __name__ == "__main__":
    main()
