import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm
import math
import colorsys
from plotly.subplots import make_subplots
from atomicorbit.orbital_maths.generell_functions.radial_part_wavefunc import R_nl
from atomicorbit.orbital_maths.generell_functions.spherical_harmonic_func import Y_lm


def orbital_wavefunction(n, l, m, r, theta, phi):
    """Calculates the wavefunction for a given orbital"""
    R = R_nl(n=n, l=l, r=r)
    Y = Y_lm(m=m, l=l, phi=phi, theta=theta)
    return (R * Y) ** 2


def plot_orbital_3d(n, l, m, title, threshold=0.03, grid_size=60):
    x, y, z, phi, theta, r = create_orbital_data(n, l, m, "single")
    print("Calculating Wavefunction...")
    prob = orbital_wavefunction(n=n, l=l, m=m, r=r, theta=theta, phi=phi)
    max_prob = np.max(prob)
    if max_prob > 0:
        norm_prob = prob / max_prob
    else:
        print("Warnung: Maximale Wahrscheinlichkeitsdichte ist Null oder NaN")
        return None

    if len(x) > 10000:
        indices = np.random.choice(len(x), 10000, replace=False)
        x, y, z, norm_prob = x[indices], y[indices], z[indices], norm_prob[indices]

    # Create a regular 3D grid
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    zi = np.linspace(z.min(), z.max(), grid_size)
    X, Y, Z = np.meshgrid(xi, yi, zi)

    print("Interpolating Values...")
    interp = NearestNDInterpolator(list(zip(x.flatten(), y.flatten(), z.flatten())), norm_prob.flatten())
    PI = interp(X.flatten(), Y.flatten(), Z.flatten()).reshape(X.shape)

    fig = go.Figure()
    print("Staring the plot of the Orbital...")
    # Isosurface
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=PI.flatten(),
        isomin=threshold,
        isomax=1,
        surface_count=3,
        colorscale='Viridis',
        opacity=0.6,
        name='Electron probability'
    ))

    # Nucleus
    fig.add_trace(go.Mesh3d(
        x=[0, 5, -5, 0, 0],
        y=[0, 0, 0, 5, -5],
        z=[5, 0, 0, 0, 0],
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[3, 4, 4, 3],
        color='red',
        opacity=0.8,
        name='Nucleus'
    ))

    max_range = np.max([x.max(), y.max(), z.max(), -x.min(), -y.min(), -z.min()])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (pm)',
            yaxis_title='Y (pm)',
            zaxis_title='Z (pm)',
            aspectmode='cube',
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range]),
        ),
        width=800,
        height=800,
        margin=dict(r=20, l=10, b=10, t=40)
    )

    return fig


def plot_multiple_orbitals(electron_count, grid_size=70, max_orbitals=100):
    n_list, l_list, m_list, s_list = calculate_quantum_numbers(electron_count=electron_count)
    orbital_data, titles = create_orbital_data(n_list, l_list, m_list, "multiple")
    num_orbitals = min(len(orbital_data), max_orbitals)
    cols = math.ceil(math.sqrt(num_orbitals))
    rows = math.ceil(num_orbitals / cols)

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scene'}] * cols] * rows,
        subplot_titles=titles[:num_orbitals]
    )

    for i, (x, y, z, prob) in tqdm(enumerate(orbital_data[:num_orbitals]), total=num_orbitals, desc="Plotting multiple orbitals"):
        row = i // cols + 1
        col = i % cols + 1

        max_prob = np.max(prob)
        if max_prob > 0:
            norm_prob = prob / max_prob
        else:
            print(f"Warning: Maximum probability density is zero or NaN for orbital {i}")
            continue

        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        zi = np.linspace(z.min(), z.max(), grid_size)
        X, Y, Z = np.meshgrid(xi, yi, zi)

        interp = NearestNDInterpolator(list(zip(x.flatten(), y.flatten(), z.flatten())), norm_prob.flatten())
        PI = interp(X.flatten(), Y.flatten(), Z.flatten()).reshape(X.shape)

        # Isosurface
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=PI.flatten(),
                isomin=0.1,
                isomax=0.8,
                surface_count=5,
                colorscale='Viridis',
                opacity=0.6,
            ),
            row=row, col=col
        )

        # Nucleus
        fig.add_trace(
            go.Mesh3d(
                x=[0], y=[0], z=[0],
                alphahull=0,
                color='red',
                opacity=0.8,
                name='Nucleus'
            ),
            row=row, col=col
        )

        # Update layout for each subplot
        fig.update_scenes(
            aspectmode='cube',
            xaxis_title='X (pm)',
            yaxis_title='Y (pm)',
            zaxis_title='Z (pm)',
            row=row, col=col
        )

    fig.update_layout(height=400 * rows, width=400 * cols, title_text="Multiple Orbitals")
    return fig


def plot_multiple_orbitals_single_plot(electron_count, grid_size=60):
    fig = go.Figure()

    n_list, l_list, m_list, s_list = calculate_quantum_numbers(electron_count)
    orbital_data, titles = create_orbital_data(n_list, l_list, m_list, "all")

    max_range = max([np.max([np.abs(data[0]).max(), np.abs(data[1]).max(), np.abs(data[2]).max()]) for data in orbital_data])

    sorted_data = sorted(zip(orbital_data, titles, n_list, l_list, m_list),
                         key=lambda x: (-x[2], -x[3], -abs(x[4])))

    for i, (data, title, n, l, m) in tqdm(enumerate(sorted_data), total=len(sorted_data), desc="Plotting multiple orbitals in one graph"):
        x, y, z, prob = data
        max_prob = np.max(prob)
        if max_prob > 0:
            norm_prob = prob / max_prob
        else:
            print(f"Warning: Maximum probability density is zero or NaN for orbital {title}")
            continue

        xi = np.linspace(-max_range, max_range, grid_size)
        yi = np.linspace(-max_range, max_range, grid_size)
        zi = np.linspace(-max_range, max_range, grid_size)
        X, Y, Z = np.meshgrid(xi, yi, zi)

        interp = NearestNDInterpolator(list(zip(x.flatten(), y.flatten(), z.flatten())), norm_prob.flatten())
        PI = interp(X.flatten(), Y.flatten(), Z.flatten()).reshape(X.shape)

        #color = f'rgb({255 - (n*50)%256}, {255 - (l*80)%256}, {255 - (abs(m)*100)%256})'
        #color = f'rgb({128 + (n * 20) % 128}, {128 + (l * 25) % 128}, {128 + (abs(m) * 50) % 128})'
        #color = spectral_color(n=n, l=l, m=m)
        max_n = max(n_list)  # Assuming n_list contains all n values
        #color = enhanced_blue_orbital_scheme(n=n, l=l, m=m, max_n=max_n)
        #colorscale = [[i / 100, darker_blue_orbital_scheme(n, l, m, max_n)] for i in range(101)]

        # Isosurface
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=PI.flatten(),
                isomin=0.1,
                isomax=0.8,
                surface_count=3,
                colorscale=[[i / 100, darker_blue_orbital_scheme(n, l, m, max_n)] for i in range(101)],
                opacity=0.5,
                name=title,
                caps=dict(x_show=False, y_show=False, z_show=False)  # Hide caps to see through the surfaces
            )
        )

    # Nucleus
    fig.add_trace(
        go.Mesh3d(
            x=[0], y=[0], z=[0],
            alphahull=0,
            color='red',
            opacity=0.8,
            name='Nucleus'
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis_title='X (pm)',
            yaxis_title='Y (pm)',
            zaxis_title='Z (pm)',
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range]),
        ),
        width=800,
        height=800,
        title_text="Multiple Orbitals in Single Plot"
    )
    return fig


def create_orbital_data(n_values, l_values, m_values, type_plot):
    """
    Simplified and fast orbital data generation.
    """
    orbital_data = []
    titles = []

    if type_plot == "single":
        points_r = 300
        points_theta = 300
        points_phi = 600
    else:
        points_r = 150
        points_theta = 150
        points_phi = 300

    if type_plot != "single":
        highest_n = max(n_values)
    else:
        highest_n = n_values

    stop = highest_n * 1e-9

    if type_plot == "multiple" or type_plot == "all":
        for n, l, m in tqdm(zip(n_values, l_values, m_values), total=len(n_values), desc="Calculating orbitals"):
            r = np.linspace(0, stop, points_r)
            theta = np.linspace(0, np.pi, points_theta)
            phi = np.linspace(0, 2 * np.pi, points_phi)

            r, theta, phi = np.meshgrid(r, theta, phi)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            prob_d = orbital_wavefunction(r=r, n=n, l=l, m=m, theta=theta, phi=phi)

            orbital_data.append((x, y, z, prob_d))
            titles.append(f"Orbital (n={n}, l={l}, m={m})")

        return orbital_data, titles

    else:
        r = np.linspace(0, stop, points_r)
        theta = np.linspace(0, np.pi, points_theta)
        phi = np.linspace(0, 2 * np.pi, points_phi)

        r, theta, phi = np.meshgrid(r, theta, phi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return x, y, z, phi, theta, r


def darker_blue_orbital_scheme(n, l, m, max_n):
    base_hue = 0.6  # blue

    saturation = 0.8 - (n / (max_n * 2))
    saturation = max(0.3, saturation)

    value = 0.2 + (n / max_n) * 0.5

    hue_variation = (l * 0.015 + abs(m) * 0.008) % 0.08
    final_hue = (base_hue + hue_variation) % 1.0

    r, g, b = colorsys.hsv_to_rgb(final_hue, saturation, value)

    r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return f'rgb({r}, {g}, {b})'


def calculate_quantum_numbers(electron_count):
    n = 1
    electrons_left = electron_count
    n_list = []
    l_list = []
    m_list = []
    s_list = []

    while electrons_left > 0:
        for l in range(n):
            for m in range(-l, l+1):
                for s in [-0.5, 0.5]:
                    if electrons_left > 0:
                        if s == -0.5 or electrons_left == 1:
                            n_list.append(n)
                            l_list.append(l)
                            m_list.append(m)
                            s_list.append(s)
                        electrons_left -= 1
                    else:
                        break
                if electrons_left == 0:
                    break
            if electrons_left == 0:
                break
        n += 1

    return n_list, l_list, m_list, s_list
