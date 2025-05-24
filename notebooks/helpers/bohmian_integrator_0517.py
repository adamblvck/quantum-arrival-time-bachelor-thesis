import numpy as np
from scipy.interpolate import RegularGridInterpolator

def rk4(pos, dt, fx, fy):
    """ Runge-Kutta 4th order integrator """
    def vel(p):
        return np.array([fx((p[1], p[0])), fy((p[1], p[0]))])

    k1 = vel(pos)
    k2 = vel(pos + 0.5*dt*k1)
    k3 = vel(pos + 0.5*dt*k2)
    k4 = vel(pos +     dt*k3)
    return pos + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def wrap_position(pos, x_min, x_max, y_min, y_max):
    """Wrap position to stay within simulation boundaries"""
    x, y = pos
    x_wrapped = x_min + (x - x_min) % (x_max - x_min)
    y_wrapped = y_min + (y - y_min) % (y_max - y_min)
    return np.array([x_wrapped, y_wrapped])

def compute_bohmian_trajectories(
        simulation_data, params,
        alpha=1.0+0j, beta=0.0+0j,
        n_trajectories=10, random_seed=0):
    """
    Bohmian trajectories for a 2‑component Pauli spinor.

    The guidance equation uses the *total* probability density

        ρ = |ψ_up|² + |ψ_down|²         (1)

    and the *total* current density

        j = (ħ/m) Im[ ψ_up* ∇ψ_up + ψ_down* ∇ψ_down ]   (2)

    Optional global spin coefficients α, β allow you to examine
    spinors αψ_up + βψ_down without re‑running the TDSE.

    Parameters
    ----------
    simulation_data : dict returned by `simulate_2d_spin`
        *must* contain ``'psi_up_list'`` and ``'psi_down_list'``.
    params : dict with keys ``'x0', 'y0', 'sigma'``  (for spawning)
    alpha, beta : complex
        Global spin amplitudes (normalisation is enforced internally).
    n_trajectories : int
    random_seed : int

    Returns
    -------
    trajectories : ndarray  (n_trajectories, N_t, 2)
    """

    # --- grids & time axis --------------------------------------------------
    x = np.asarray(simulation_data['x'])
    y = np.asarray(simulation_data['y'])
    t = np.asarrawy(simulation_data['t'])
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]
    Ny, Nx = len(y), len(x)
    N_t = len(t)
    
    # Get simulation boundaries for wrapping
    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]

    # --- spinor amplitudes --------------------------------------------------
    psi_up_all   = np.asarray(simulation_data['psi_up_list'])    # (N_t, Ny, Nx) complex
    psi_down_all = np.asarray(simulation_data['psi_down_list'])  # (N_t, Ny, Nx) complex

    # global spin coefficients, re‑normalise just in case
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm

    # --- initial ensemble ---------------------------------------------------
    rng = np.random.default_rng(random_seed)
    x0, y0, sigma = params['x0'], params['y0'], params['sigma']
    starts = np.column_stack([
        rng.normal(x0, sigma, n_trajectories),
        np.full(n_trajectories, fill_value=y0)
    ])

    traj = np.zeros((n_trajectories, N_t, 2), dtype=float)
    traj[:, 0, :] = starts

    # constants for current
    hbar = params.get('hbar', 1.0)
    m    = params.get('m', 1.0)

    # n_sub = 2                         # Euler sub‑steps per frame - not used anymore
    # sub_dt = dt
    eps = 1e-20                       # to avoid 0/0

    # --- main loop ----------------------------------------------------------
    for k in range(N_t - 1): # how many frames it take to get to the barrier

        # combine spinor with the chosen (α, β)
        psi_up   = alpha * psi_up_all[k]
        psi_down = beta  * psi_down_all[k]

        # probability density ρ(x,y,t_k)
        rho = (abs(psi_up)**2 + abs(psi_down)**2) + eps

        # spatial derivatives (used meshgrid indexing='xy' so axis=1 -> x, axis=0 -> y)
        dpsi_up_dx   = np.gradient(psi_up,   dx, axis=1)
        dpsi_up_dy   = np.gradient(psi_up,   dy, axis=0)
        dpsi_down_dx = np.gradient(psi_down, dx, axis=1)
        dpsi_down_dy = np.gradient(psi_down, dy, axis=0)

        pref = hbar / m
        jx = pref * np.imag(np.conj(psi_up) * dpsi_up_dx +
                            np.conj(psi_down) * dpsi_down_dx)
        jy = pref * np.imag(np.conj(psi_up) * dpsi_up_dy +
                            np.conj(psi_down) * dpsi_down_dy)

        # guidance velocity field - velocity = current / density
        # vx = jx / rho            
        # vy = jy / rho

        # masking vx/vy where rho (wavefunction probability density) is too small
        rho_min = 1e-6 * rho.max()
        mask    = rho > rho_min
        vx      = np.zeros_like(jx)
        vy      = np.zeros_like(jy)
        vx[mask] = jx[mask] / rho[mask]
        vy[mask] = jy[mask] / rho[mask]

        speed_cap = 10*np.percentile(np.sqrt(vx**2 + vy**2), 99)   # robust cap
        vx_safe = vx # np.clip(vx, -speed_cap, speed_cap)
        vy_safe = vy # np.clip(vy, -speed_cap, speed_cap)

        interp_vx = RegularGridInterpolator((y,x), vx_safe,
                                            bounds_error=False)
        interp_vy = RegularGridInterpolator((y,x), vy_safe,
                                            bounds_error=False)

        # --- inside the time loop, after vx, vy are ready -----------------
        # rho_min = 1e-8 * rho.max()            # much smaller
        # speed_cap = 5000.0                       # max |v|, in grid units / dt
        # vx = np.clip(vx, -speed_cap, speed_cap, where=rho<rho_min)
        # vy = np.clip(vy, -speed_cap, speed_cap, where=rho<rho_min)

        # ensure to take a minium rho value, to avoid sudden velocity spikes
        # interp_vx = RegularGridInterpolator((y, x), np.where(rho<rho_min, 0, vx))
        # interp_vy = RegularGridInterpolator((y, x), np.where(rho<rho_min, 0, vy))

        # interpolators expect (y, x) order for query points
        # interp_vx = RegularGridInterpolator((y, x), vx)
        # interp_vy = RegularGridInterpolator((y, x), vy)

        # propagate all trajectories with two Euler half‑steps
        for n in range(n_trajectories):
            pos = traj[n, k].copy()          # (x, y)

            # ------ one RK‑4 step over dt ---------
            new_pos = rk4(pos, dt, interp_vx, interp_vy)
            # --------------------------------------
            
            # Wrap position to stay within simulation boundaries
            new_pos = wrap_position(new_pos, x_min, x_max, y_min, y_max)

            # n euler substeps
            # for _ in range(n_sub):
            #     v_x = interp_vx((pos[1], pos[0]))
            #     v_y = interp_vy((pos[1], pos[0]))
            #     pos[0] += sub_dt * v_x
            #     pos[1] += sub_dt * v_y
            
            traj[n, k + 1] = new_pos

    return traj