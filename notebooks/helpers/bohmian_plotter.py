import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
def plot_trajectories(trajectories, params, till_frame=None, simulation_data=None, alpha=1.0+0j, beta=0.0+0j, same_chart=False):
    """
    Plot Bohmian trajectories for spin up, spin down, and a combined plot.
    
    Parameters
    ----------
    trajectories : ndarray, shape (n_traj, N_t, 2)
        Bohmian trajectories
    params : dict
        Simulation parameters used to set plot limits and the magnetic barrier.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    # Plot spin trajectories.
    for traj in trajectories:
        if till_frame is not None:
            ax.plot(traj[:till_frame, 0], traj[:till_frame, 1], alpha=0.2, color='white')
        else:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.2, color='white')
    ax.set_title("Bohmian Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Draw the magnetic barrier as a circle on all subplots.
    R = params['mag_barrier_width'] / 2.0  # radius of the barrier
    
    circle = Circle((params['mag_barrier_center_x'], params['mag_barrier_center_y']),
                    R, color='green', fill=False, lw=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(params['x_min'], params['x_max'])
    ax.set_ylim(params['y_min'], params['y_max'])
    
    # add <psi|psi>
    if simulation_data is not None:

        # global spin coefficients, re‑normalise just in case
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha, beta = alpha / norm, beta / norm

        x = np.asarray(simulation_data['x'])
        y = np.asarray(simulation_data['y'])

        psi_up = np.asarray(simulation_data['psi_up_list'])
        psi_down = np.asarray(simulation_data['psi_down_list'])

        if till_frame is not None:  
            psi_up   = alpha * psi_up[till_frame]
            psi_down = beta  * psi_down[till_frame]
        else:
            psi_up   = alpha * psi_up[-1]
            psi_down = beta  * psi_down[-1]

        # probability density ρ(x,y,t_k)
        rho = (np.abs(psi_up)**2 + np.abs(psi_down)**2)

        im = ax.pcolormesh(x, y, rho, shading='auto', cmap='viridis')
        ax.set_title(r"Probability Density $\langle \psi | \psi \rangle$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        ax.set_xlim(params['x_min'], params['x_max'])
        ax.set_ylim(params['y_min'], params['y_max'])

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm   # nicer density scale

def plot_velocity_field(simulation_data,
                        frame_indices,
                        params,
                        trajectories=None,
                        stride=4,            # decimate grid for quiver arrows
                        background='rho',    # 'rho' | 'speed' | None
                        alpha=1.0+0j, beta=0.0+0j,
                        quiver_kwargs=None,
                        stream=False):
    """
    Visualise the Bohmian guidance field v = j / ρ.

    Parameters
    ----------
    simulation_data : dict
        Must contain 'psi_up_list', 'psi_down_list', 'x', 'y', 't'
        in the same shape you used for `compute_bohmian_trajectories`.
    frame_indices : int or 1‑D iterable of ints
        Frame(s) you want to look at – 0 ≡ first time step.
    params : dict
        Same parameter dictionary you passed to the simulator.
        Needs keys used below for axes limits & magnetic barrier.
    trajectories : ndarray, optional
        (n_traj, N_t, 2) array returned by `compute_bohmian_trajectories`.
        If given, the trajectories up to the selected frame(s) are drawn.
    stride : int
        Every *stride*‑th grid point is used for arrows (keeps plots readable).
    background : {'rho', 'speed', None}
        • 'rho'   – show the probability density ρ as a log‑heatmap  
        • 'speed' – show |v| (useful for spotting spikes)  
        • None    – no background
    alpha, beta : complex
        Global spin coefficients (same meaning as in your integrator).
    quiver_kwargs : dict
        Extra matplotlib kwargs forwarded to `ax.quiver`.
    stream : bool
        If True, draw a streamplot instead of a quiver.
    """
    # --- unpack basic data --------------------------------------------------
    x = np.asarray(simulation_data['x'])
    y = np.asarray(simulation_data['y'])
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    psi_up_all   = np.asarray(simulation_data['psi_up_list'])
    psi_down_all = np.asarray(simulation_data['psi_down_list'])

    # normalise spin coeffs just in case
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha/norm, beta/norm

    # make sure we can iterate over frame_indices
    if np.isscalar(frame_indices):
        frame_indices = [frame_indices]

    if quiver_kwargs is None:
        quiver_kwargs = dict(scale=50, headwidth=3, headlength=4, minlength=0.5)

    Ny, Nx = len(y), len(x)
    X, Y   = np.meshgrid(x, y)         # for plotting

    hbar = params.get('hbar', 1.0)
    m    = params.get('m',   1.0)
    pref = hbar / m
    eps  = 1e-14                       # avoid divide‑by‑zero

    n_cols = len(frame_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6), squeeze=False)

    for ax, k in zip(axes[0], frame_indices):
        # --- spinor at this frame ------------------------------------------
        psi_up   = alpha * psi_up_all[k]
        psi_down = beta  * psi_down_all[k]

        rho = (np.abs(psi_up)**2 + np.abs(psi_down)**2) + eps

        dpsi_up_dx   = np.gradient(psi_up,   dx, axis=1)
        dpsi_up_dy   = np.gradient(psi_up,   dy, axis=0)
        dpsi_down_dx = np.gradient(psi_down, dx, axis=1)
        dpsi_down_dy = np.gradient(psi_down, dy, axis=0)

        jx = pref * np.imag(np.conj(psi_up) * dpsi_up_dx +
                            np.conj(psi_down) * dpsi_down_dx)
        jy = pref * np.imag(np.conj(psi_up) * dpsi_up_dy +
                            np.conj(psi_down) * dpsi_down_dy)

        vx = jx / rho
        vy = jy / rho
        speed = np.sqrt(vx**2 + vy**2)

        # --- background ----------------------------------------------------
        if background == 'rho':
            im = ax.pcolormesh(X, Y, rho,
                               shading='auto', cmap='viridis',
                               norm=LogNorm(vmin=rho.max()*1e-6, vmax=rho.max()))
            fig.colorbar(im, ax=ax, label=r'$\rho$')
        elif background == 'speed':
            im = ax.pcolormesh(X, Y, speed,
                               shading='auto', cmap='magma',
                               norm=LogNorm(vmin=speed.max()*1e-6, vmax=speed.max()))
            fig.colorbar(im, ax=ax, label='|v|')

        # --- quiver / streamplot ------------------------------------------
        skip = (slice(None, None, stride), slice(None, None, stride))
        if stream:
            ax.streamplot(x, y, vx, vy, density=1.2, linewidth=1, arrowsize=1.2)
        else:
            ax.quiver(X[skip], Y[skip], vx[skip], vy[skip], **quiver_kwargs)

        # --- optional trajectories ----------------------------------------
        if trajectories is not None:
            for traj in trajectories:
                ax.plot(traj[:k+1, 0], traj[:k+1, 1],
                        color='white', lw=0.8, alpha=0.7)

        # --- magnetic barrier ---------------------------------------------
        R = params['mag_barrier_width'] / 2.0
        circ = Circle((params['mag_barrier_center_x'],
                       params['mag_barrier_center_y']),
                      R, fc='none', ec='green', lw=2)
        ax.add_patch(circ)

        # --- cosmetics -----------------------------------------------------
        ax.set_aspect('equal')
        ax.set_xlim(params['x_min'], params['x_max'])
        ax.set_ylim(params['y_min'], params['y_max'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Frame {k}  (t = {simulation_data["t"][k]:.3f})')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def quick_quiver(simulation_data,                # dict returned by your solver
                 k,                              # frame index  (0 = first)
                 params,                         # same dict you already use
                 stride=3,                       # decimate grid for clarity
                 alpha=1.0+0j, beta=0.0+0j,      # global spin coeffs
                 speed_cap=None):                # optional hard cap on |v|
    """
    Plot the Bohmian guidance field v = j/ρ for a single frame k.

    Examples
    --------
    >>> quick_quiver(simulation_data, 75, params)          # basic use
    >>> quick_quiver(simulation_data, 40, params, stride=2,
    ...               speed_cap=5)                         # denser arrows
    """

    # ----- unpack grid & wavefunction for this frame -----------------------
    x = np.asarray(simulation_data['x'])
    y = np.asarray(simulation_data['y'])
    dx, dy = x[1] - x[0], y[1] - y[0]

    psi_up   = np.asarray(simulation_data['psi_up_list'][k])
    psi_down = np.asarray(simulation_data['psi_down_list'][k])

    # normalise (α,β) just in case
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm
    psi_up   = alpha * psi_up
    psi_down = beta  * psi_down

    # ----- probability density & current ----------------------------------
    eps  = 1e-14
    rho  = (np.abs(psi_up)**2 + np.abs(psi_down)**2) + eps

    dpsi_up_dx   = np.gradient(psi_up,   dx, axis=1)
    dpsi_up_dy   = np.gradient(psi_up,   dy, axis=0)
    dpsi_down_dx = np.gradient(psi_down, dx, axis=1)
    dpsi_down_dy = np.gradient(psi_down, dy, axis=0)

    hbar = params.get('hbar', 1.0)
    m    = params.get('m',   1.0)
    pref = hbar / m

    jx = pref * np.imag(np.conj(psi_up) * dpsi_up_dx +
                        np.conj(psi_down) * dpsi_down_dx)
    jy = pref * np.imag(np.conj(psi_up) * dpsi_up_dy +
                        np.conj(psi_down) * dpsi_down_dy)

    # quick mask
    # vx = jx / rho
    # vy = jy / rho

    # masking vx/vy where rho (wavefunction probability density) is too small
    rho_min = 1e-6 * rho.max()
    mask    = rho > rho_min
    vx      = np.zeros_like(jx);  vy = np.zeros_like(jy)
    vx[mask] = jx[mask] / rho[mask]
    vy[mask] = jy[mask] / rho[mask]

    # optional safety cap
    if speed_cap is not None:
        vx = np.clip(vx, -speed_cap, speed_cap)
        vy = np.clip(vy, -speed_cap, speed_cap)

    # ----- plot ------------------------------------------------------------
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(6, 6))

    skip = (slice(None, None, stride), slice(None, None, stride))
    ax.quiver(X[skip], Y[skip], vx[skip]/2, vy[skip]/2,
              scale=50, headwidth=3, headlength=3, minlength=0.5)

    # magnetic barrier (if any)
    R = params['mag_barrier_width'] / 2.0
    circ = Circle((params['mag_barrier_center_x'],
                   params['mag_barrier_center_y']),
                  R, fc='none', ec='green', lw=2)
    ax.add_patch(circ)

    ax.set_aspect('equal')
    ax.set_xlim(params['x_min'], params['x_max'])
    ax.set_ylim(params['y_min'], params['y_max'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Quiver field – frame {k} (t = {simulation_data["t"][k]:.3f})')

    plt.tight_layout()
    plt.show()
