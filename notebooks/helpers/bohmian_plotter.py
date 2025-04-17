import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_trajectories(trajectories_up, trajectories_down, params):
    """
    Plot Bohmian trajectories for spin up, spin down, and a combined plot.
    
    Parameters
    ----------
    trajectories_up : ndarray, shape (n_traj, N_t, 2)
        Bohmian trajectories for the spin up component.
    trajectories_down : ndarray, shape (n_traj, N_t, 2)
        Bohmian trajectories for the spin down component.
    params : dict
        Simulation parameters used to set plot limits and the magnetic barrier.
    """
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot spin up trajectories.
    for traj in trajectories_up:
        axs[0].plot(traj[:, 0], traj[:, 1], alpha=0.7)
    axs[0].set_title("Spin Up Bohmian Trajectories")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    
    # Plot spin down trajectories.
    for traj in trajectories_down:
        axs[1].plot(traj[:, 0], traj[:, 1], alpha=0.7)
    axs[1].set_title("Spin Down Bohmian Trajectories")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    
    # Combined plot: spin up in red, spin down in blue.
    for traj in trajectories_up:
        axs[2].plot(traj[:, 0], traj[:, 1], color='red', alpha=0.7)
    for traj in trajectories_down:
        axs[2].plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.7)
    axs[2].set_title("Combined Bohmian Trajectories")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    
    # Draw the magnetic barrier as a circle on all subplots.
    R = params['mag_barrier_width'] / 2.0  # radius of the barrier
    for ax in axs:
        circle = Circle((params['mag_barrier_center_x'], params['mag_barrier_center_y']),
                        R, color='green', fill=False, lw=2)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlim(params['x_min'], params['x_max'])
        ax.set_ylim(params['y_min'], params['y_max'])
    
    plt.tight_layout()
    plt.show()
