import numpy as np
import matplotlib.pyplot as plt

def plot_simulation_heatmaps_spin(simulation_data, num_frames=4):
    """
    Plot multiple frames of the 2D quantum simulation (spin up & spin down)
    as side-by-side heatmaps.

    Parameters:
        simulation_data: dict
            Output from simulate_2d_spin function containing 'x', 'y', 't',
            'prob_up', 'prob_down'.
        num_frames: int
            Number of timeslices to display.
    """
    # Convert lists back to numpy arrays
    x = np.array(simulation_data['x'])
    y = np.array(simulation_data['y'])
    t = np.array(simulation_data['t'])
    prob_up = np.array(simulation_data['prob_up'])       # shape: (n_steps, Nx, Ny)
    prob_down = np.array(simulation_data['prob_down'])   # shape: (n_steps, Nx, Ny)

    n_steps = len(t)

    # Pick frames evenly spaced in time
    frame_indices = np.linspace(0, n_steps - 1, num_frames, dtype=int)

    # Create figure with subplots:
    # Each "row" is a specific time frame. We show spin-up in the left subplot,
    # spin-down in the right subplot for that time.
    fig, axes = plt.subplots(nrows=num_frames, ncols=2,
                             figsize=(12, 5 * num_frames),
                             squeeze=False)

    for row, frame_idx in enumerate(frame_indices):
        ax_up = axes[row, 0]
        ax_down = axes[row, 1]

        # Spin Up
        im_up = ax_up.pcolormesh(x, y, prob_up[frame_idx], shading='auto', cmap='viridis')
        ax_up.set_title(f'Spin Up, t = {t[frame_idx]:.2f}')
        ax_up.set_xlabel('x')
        ax_up.set_ylabel('y')
        plt.colorbar(im_up, ax=ax_up)

        # Spin Down
        im_down = ax_down.pcolormesh(x, y, prob_down[frame_idx], shading='auto', cmap='viridis')
        ax_down.set_title(f'Spin Down, t = {t[frame_idx]:.2f}')
        ax_down.set_xlabel('x')
        ax_down.set_ylabel('y')
        plt.colorbar(im_down, ax=ax_down)

    plt.tight_layout()
    return fig


def plot_potential_spin(x, y,
                       mag_barrier_center_x=0.0,
                       mag_barrier_center_y=0.0,
                       mag_barrier_strength=5.0,
                       mag_barrier_width=0.5,
                       use_gravity=True,
                       g=9.81,
                       m=1.0):
    """
    Visualize the spin-dependent potential.  We'll show the spin-up potential
    (common potential + +mag_barrier) and spin-down potential 
    (common potential + -mag_barrier).
    """
    X, Y = np.meshgrid(x, y, indexing='xy')
    magnetic_barrier = mag_barrier_strength * np.exp(
        -(((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2)
          / (2 * mag_barrier_width**2))
    )

    # Common potential
    if use_gravity:
        V_common = m * g * Y
    else:
        V_common = 0.0

    # Spin up: V_up
    V_up = V_common + magnetic_barrier
    # Spin down: V_down
    V_down = V_common - magnetic_barrier

    # Plot them side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.pcolormesh(x, y, V_up, shading='auto', cmap='viridis')
    ax1.set_title('Spin-Up Potential')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.pcolormesh(x, y, V_down, shading='auto', cmap='viridis')
    ax2.set_title('Spin-Down Potential')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig


# if __name__ == "__main__":
#     # Example usage (with mock data). 
#     # In practice, you would do something like:
#     #
#     #   sim_data = simulate_2d_spin(...)
#     #   plot_simulation_heatmaps_spin(sim_data, num_frames=6)
#     #   plt.show()
#     #
#     Nx, Ny = 64, 64
#     x = np.linspace(-10, 10, Nx)
#     y = np.linspace(-10, 10, Ny)

#     # Mock spin potentials
#     fig_pot = plot_potential_spin(x, y)
#     plt.show()

#     # Mock data
#     n_steps = 30
#     t = np.linspace(0, 3, n_steps)
#     prob_up_mock = np.random.rand(n_steps, Nx, Ny)
#     prob_down_mock = np.random.rand(n_steps, Nx, Ny)
#     sim_data = {
#         'x': x.tolist(),
#         'y': y.tolist(),
#         't': t.tolist(),
#         'prob_up': prob_up_mock.tolist(),
#         'prob_down': prob_down_mock.tolist(),
#     }
#     fig_heatmaps = plot_simulation_heatmaps_spin(sim_data, num_frames=4)
#     plt.show()
