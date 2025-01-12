from src.atomicorbit.visualization_vispy.atom_orbital import ElectronDensityVisualizer
import os
import numpy as np
import cv2
from vispy import app, scene
from tqdm import tqdm


class OrbitalAnimator:
    def __init__(self, visualizer_class, visual_dict, output_path="output"):
        self.output_path = output_path
        self.frame_path = os.path.join(output_path, "frames")
        os.makedirs(self.frame_path, exist_ok=True)

        self.visualizer_class = visualizer_class
        self.visual_dict = visual_dict.copy()

        # Animation parameters
        self.frame_count = 0
        self.total_frames = 0
        self.B_fields = None
        self.camera_angle = None
        self.pbar = None
        self.first_frame = True

    def setup_animation(self, B_field_range=(0, 1e6), num_frames=100):
        start_field, end_field = B_field_range
        self.B_fields = np.linspace(start_field, end_field, num_frames)
        self.camera_angle = np.linspace(-70, 70, num_frames)
        self.total_frames = num_frames
        self.frame_count = 0

        # Initialize progress bar
        self.pbar = tqdm(total=num_frames, desc="Generating frames")

        # Create visualizer
        self.visualizer = self.visualizer_class(visual_dict=self.visual_dict)

        # Set up timer for frame capture
        self.timer = app.Timer(interval=0.1, connect=self.update, start=True)

    def update(self, event):
        if self.frame_count >= self.total_frames:
            self.timer.stop()
            self.pbar.close()
            self.create_video()
            self.cleanup_frames()
            self.visualizer.clear_cache()  # Cache am Ende leeren
            app.quit()
            return

        # Update magnetic field
        self.visual_dict["magnetic_field"] = self.B_fields[self.frame_count]
        self.visual_dict["angle"] = 90

        # Clear previous visualization
        for child in self.visualizer.view.scene.children[:]:
            child.parent = None

        # Create new visualization
        self.visualizer.visualize_atom(1)

        # Save frame
        filename = os.path.join(self.frame_path, f"frame_{self.frame_count:04d}.png")
        img = self.visualizer.canvas.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(filename, img)

        self.frame_count += 1
        self.pbar.update(1)

    def create_video(self, output_filename="animation.mp4", fps=20):
        """Erstellt ein Video aus den gespeicherten Frames."""
        print("\nCreating video...")
        frame_files = sorted([f for f in os.listdir(self.frame_path) if f.endswith('.png')])
        if not frame_files:
            raise ValueError("Keine Frames gefunden!")

        # Lese erste Frame für die Dimensionen
        first_frame = cv2.imread(os.path.join(self.frame_path, frame_files[0]))
        height, width, _ = first_frame.shape

        # Video Writer Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            os.path.join(self.output_path, output_filename),
            fourcc, fps, (width, height)
        )

        # Frames zum Video hinzufügen
        for frame_file in tqdm(frame_files, desc="Creating video"):
            frame = cv2.imread(os.path.join(self.frame_path, frame_file))
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as: {os.path.join(self.output_path, output_filename)}")

    def cleanup_frames(self):
        """Löscht die temporären Frame-Dateien."""
        print("Cleaning up temporary files...")
        for file in os.listdir(self.frame_path):
            os.remove(os.path.join(self.frame_path, file))
        os.rmdir(self.frame_path)


# Beispielverwendung:
if __name__ == "__main__":
    visual_dict = {
        "angle": 0,
        "state": "translucent",
        "not_see_inside": True,
        "blend": True,
        "edges": False,
        "point_size": 1,
        "prob_threshold": 0.3,
        "num_points": 1000000,
        "magnetic_field": 0,
        "field_lines": False,
        "info": False
    }

    animator = OrbitalAnimator(
        visualizer_class=ElectronDensityVisualizer,
        visual_dict=visual_dict
    )

    animator.setup_animation(
        B_field_range=(0, 10000),
        num_frames=100
    )
    app.run()
