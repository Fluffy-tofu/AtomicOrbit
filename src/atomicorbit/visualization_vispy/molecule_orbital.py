from src.atomicorbit.visualization_vispy.atom_orbital import GeneralFunctions
from vispy import app, scene
import numpy as np


class MoleculeOrbital(GeneralFunctions):
    def __init__(self, visual_dict):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(2000, 1200), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 5
        self.axis = scene.XYZAxis(parent=self.view.scene)
        self.visual_dict = visual_dict

    def diatomic_molecule(self, angstroms, atom, num_points=100000, threshold=0.1):
        r, theta, phi = self.generate_grid(1, 1, num_points)
        x1, y1, z1 = self.convert_cartesian(r, theta, phi)
        distance = angstroms

        if atom == "hydrogen":
            x1, y1, z1 = self.convert_cartesian(r, theta, phi)
            x1 += distance

            x2, y2, z2 = self.convert_cartesian(r, theta, phi)
            #y2 += distance

            wave_1 = self.wave_func(n=1, l=0, m=0, Z=1, theta=theta, phi=phi,
                                    r=np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
            wave_2 = self.wave_func(n=1, l=0, m=0, Z=1, theta=theta, phi=phi,
                                    r=np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))

            wave = wave_2 - wave_1
            density = abs(wave) ** 2
            if density.max() != 0:
                density /= density.max()

            mask = density >= threshold
            x, y, z, density = x1[mask], y1[mask], z1[mask], density[mask]

            return (x, y, z, density)

    def add_orbital(self, atom, angstroms):
        """Visualize molecular orbital."""
        orbital = self.diatomic_molecule(angstroms, atom,
                                         num_points=self.visual_dict["num_points"],
                                         threshold=self.visual_dict["prob_threshold"])
        x, y, z, density = orbital

        print("Points after thresholding:", len(x))

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

    def visualize_atom(self, atom, angstroms):
        self.add_orbital(atom, angstroms)


def main():
    visual_dict = {
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.1,
        "num_points": 5000000,
        "magnetic_field_comparison": False,
        "show_difference_wavefunctions": False,
    }
    visualizer = MoleculeOrbital(visual_dict=visual_dict)
    visualizer.visualize_atom("hydrogen", 0.74)
    app.run()


if __name__ == '__main__':
    main()

