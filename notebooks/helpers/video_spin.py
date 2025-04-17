import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import datetime

def create_animation_spin(simulation_data, output_file='quantum_evolution_spin.mp4', fps=30):
    """
    Create an MP4 animation of the two-component (spin up, spin down) quantum evolution.

    Parameters:
        simulation_data: dict
            Output from simulate_2d_spin function, containing keys:
            'x', 'y', 't', 'prob_up', 'prob_down'
        output_file: str
            Name of the output MP4 file
        fps: int
            Frames per second for the animation
    """
    x = np.array(simulation_data['y'])
    y = np.array(simulation_data['x'])
    prob_up = np.array(simulation_data['prob_up'])     # Shape: (n_steps, Nx, Ny)
    prob_down = np.array(simulation_data['prob_down']) # Shape: (n_steps, Nx, Ny)
    t = np.array(simulation_data['t'])

    # Figure with two subplots: top for spin up, bottom for spin down
    # Calculate figure size to maintain square axes
    width = 8  # width in inches
    # Height should account for both squares plus some space for titles and colorbars
    height = width * 1.9  # 2.2 factor gives room for titles and colorbars
    
    fig, (ax_up, ax_down) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # fig.tight_layout()

    # Find global min and max for proper scaling
    prob_up_min = np.min(prob_up)
    prob_up_max = np.max(prob_up)
    prob_down_min = np.min(prob_down)
    prob_down_max = np.max(prob_down)

    # Initial plots with explicit vmin and vmax
    im_up = ax_up.pcolormesh(x, y, prob_up[0], shading='auto', cmap='viridis',
                            vmin=prob_up_min, vmax=prob_up_max)
    cbar_up = plt.colorbar(im_up, ax=ax_up)
    ax_up.set_title(f'Spin Up, t = {t[0]:.2f}')
    ax_up.set_xlabel('x')
    ax_up.set_ylabel('y')

    im_down = ax_down.pcolormesh(x, y, prob_down[0], shading='auto', cmap='viridis',
                                vmin=prob_down_min, vmax=prob_down_max)
    cbar_down = plt.colorbar(im_down, ax=ax_down)
    ax_down.set_title(f'Spin Down, t = {t[0]:.2f}')
    ax_down.set_xlabel('x')
    ax_down.set_ylabel('y')

    # Make both axes square
    ax_up.set_aspect('equal')
    ax_down.set_aspect('equal')


    # Update function for animation
    def update(frame):
        # Pre-compute frame data and their ranges
        up_frame = prob_up[frame]
        down_frame = prob_down[frame]
        
        # Update data
        im_up.set_array(up_frame.ravel())
        im_down.set_array(down_frame.ravel())
        
        # Update titles
        ax_up.set_title(f'Spin Up, t = {t[frame]:.2f}')
        ax_down.set_title(f'Spin Down, t = {t[frame]:.2f}')
        
        # Update color scales only if significant change
        up_max = np.max(up_frame)
        if up_max > 0.01:  # Only rescale if there's significant probability
            im_up.set_clim(vmin=0, vmax=up_max)
            
        down_max = np.max(down_frame)
        if down_max > 0.01:  # Only rescale if there's significant probability
            im_down.set_clim(vmin=0, vmax=down_max)

        return im_up, im_down

    # Create animation - remove the interval parameter as it's irrelevant for saving
    anim = FuncAnimation(
        fig, update,
        frames=len(prob_up),
        blit=True,
        cache_frame_data=False
    )

    # Force FFmpeg to maintain exact timing
    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=3000,
        extra_args=[
            '-vsync', 'cfr'  # Constant frame rate
        ]
    )

    anim.save(output_file, writer=writer)
    plt.close()

    return output_file


if __name__ == "__main__":
    # Example usage:
    # Suppose you have run your spin simulation:
    # simulation_data = simulate_2d_spin(...)
    # Then call:
    import os
    output_dir = "simulations"
    os.makedirs(output_dir, exist_ok=True)

    # Mock data example (comment this out if you have real data):
    Nx, Ny, n_steps = 64, 64, 50
    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    t = np.linspace(0, 1, n_steps)
    prob_up_mock = np.random.rand(n_steps, Nx, Ny)
    prob_down_mock = np.random.rand(n_steps, Nx, Ny)
    simulation_data = {
        'x': x.tolist(),
        'y': y.tolist(),
        't': t.tolist(),
        'prob_up': prob_up_mock.tolist(),
        'prob_down': prob_down_mock.tolist(),
    }

    mp4_file = create_animation_spin(simulation_data,
                                     output_file=os.path.join(
                                         output_dir,
                                         f'{datetime.datetime.now().strftime("%Y%m%d_%H-%M")}_spin_evolution.mp4'
                                     ),
                                     fps=10)
    print(f"Animation saved to {mp4_file}")
