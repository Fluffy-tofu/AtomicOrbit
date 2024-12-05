import numpy as np
from vispy import app, scene
from src.atomicorbit.visualization.interactive_3d_plot import orbital_wavefunction, create_orbital_data


# Reshape the arrays and combine them with probability as color
def prepare_orbital_data(x_norm, y_norm, z_norm, norm_prob, threshold=0.1, sample_rate=20):
    # Sample the arrays
    x_sampled = x_norm[::sample_rate, ::sample_rate, ::sample_rate]
    y_sampled = y_norm[::sample_rate, ::sample_rate, ::sample_rate]
    z_sampled = z_norm[::sample_rate, ::sample_rate, ::sample_rate]
    prob_sampled = norm_prob[::sample_rate, ::sample_rate, ::sample_rate]

    # Flatten and combine coordinates
    points = np.column_stack((
        x_sampled.flatten(),
        y_sampled.flatten(),
        z_sampled.flatten()
    ))

    # Flatten probability
    probs = prob_sampled.flatten()

    # Create mask for significant probability values
    mask = probs > threshold

    # Apply mask to both points and probabilities
    points = points[mask]
    probs = probs[mask]

    # Create colorful RGBA colors
    colors_rgba = np.zeros((len(probs), 4), dtype=np.float32)

    # Create a colorful gradient based on probability
    # This creates a transition from blue (low probability) to red (high probability)
    colors_rgba[:, 0] = probs  # Red increases with probability
    colors_rgba[:, 1] = 0.5 * probs  # Some green for mixing
    colors_rgba[:, 2] = 1 - probs  # Blue decreases with probability
    colors_rgba[:, 3] = probs  # Full opacity

    print(f"Number of points being plotted: {len(points)}")
    return points, colors_rgba


n, l, m = 4, 1, 1
x, y, z, phi, theta, r = create_orbital_data(n, l, m, "single")
prob = orbital_wavefunction(n=n, l=l, m=m, r=r, theta=theta, phi=phi)

max_prob = np.max(prob)
if max_prob > 0:
    norm_prob = prob / max_prob

x_max = np.max(x)
y_max = np.max(y)
z_max = np.max(z)

if x_max > 0:
    x_norm = x / x_max
    y_norm = y / y_max
    z_norm = z / z_max

# Prepare the data
points, colors = prepare_orbital_data(x_norm, y_norm, z_norm, norm_prob,
                                      threshold=0.1,
                                      sample_rate=5)

# Create the visualization
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
view = canvas.central_widget.add_view()

# Create scatter plot
scatter = scene.visuals.Markers()
scatter.set_gl_state('translucent', depth_test=False)
scatter.set_data(points,
                 edge_width=0,
                 edge_color=None,
                 face_color=colors,
                 size=10)

# Set up view
view.add(scatter)
view.camera = 'turntable'

# Add axes (optional)
axis = scene.visuals.XYZAxis(parent=view.scene)

# Show the plot
canvas.show()
app.run()