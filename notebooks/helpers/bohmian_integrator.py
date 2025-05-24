import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scipy.fft import fft2, ifft2          # modern pocketfft backend

def euler_or_heun(pos, dt, vx_f, vy_f, use_heun=False):
    """
    One time‑step with either
      • simple Euler (use_heun = False)   Δr = v_k dt
      • Heun / improved Euler (use_heun = True):
            Δr = ½ (v_k + v_k+1) dt
        which is the trapezoidal rule, i.e. O(dt²) accurate.
    vx_f, vy_f … callables returning velocity at (y, x) for *frame k*.
    """
    # --- Euler predictor -----------------------------------------------
    v0 = np.array([vx_f((pos[1], pos[0])),
                   vy_f((pos[1], pos[0]))])
    pos1 = pos + dt * v0                  # provisional end‑point

    if not use_heun:                      # plain Euler – we're done
        return pos1

    # --- Heun corrector -------------------------------------------------
    v1 = np.array([vx_f((pos1[1], pos1[0])),
                   vy_f((pos1[1], pos1[0]))])
    return pos + 0.5 * dt * (v0 + v1)

def velocity_nearest(vx_grid, vy_grid, x, y, x0, y0, dx, dy):
    """return v at (x,y) by picking the closest grid point"""
    j  = int(round((x - x0) / dx))    # column index
    i  = int(round((y - y0) / dy))    # row index
    return np.array([vx_grid[i, j], vy_grid[i, j]])

def velocity_bilinear(vx, vy, x, y, x0, y0, dx, dy):
    """manual bilinear sample of vx, vy"""
    u = (x - x0) / dx
    v = (y - y0) / dy
    j0, i0 = int(np.floor(u)), int(np.floor(v))
    du, dv = u - j0, v - i0

    # clamp indices so we don't run off the array
    j1, i1 = min(j0+1, vx.shape[1]-1), min(i0+1, vx.shape[0]-1)

    vx_val = ((1-du)*(1-dv)*vx[i0,j0] + du*(1-dv)*vx[i0,j1] +
              (1-du)*dv*vx[i1,j0]     + du*dv*vx[i1,j1])
    vy_val = ((1-du)*(1-dv)*vy[i0,j0] + du*(1-dv)*vy[i0,j1] +
              (1-du)*dv*vy[i1,j0]     + du*dv*vy[i1,j1])
    return np.array([vx_val, vy_val])

def rk4(pos, dt, fx, fy):
    """ Runge-Kutta 4th order integrator """
    def vel(p):
        return np.array([fx((p[1], p[0])), fy((p[1], p[0]))])

    k1 = vel(pos)
    k2 = vel(pos + 0.5*dt*k1)
    k3 = vel(pos + 0.5*dt*k2)
    k4 = vel(pos +     dt*k3)
    return pos + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

# some online source recommended the following implementation of RK4
# with time-dependent velocity fields
def rk4_time_interp(pos, dt,
                    vx_k, vy_k,      # interpolators at frame k
                    vx_k1, vy_k1,    # interpolators at frame k+1
                    x_min, x_max, y_min, y_max):   # add boundary parameters
    """Fourth‑order RK with *time‑dependent* RHS v(r,t) and periodic boundary conditions."""

    def v(p, lam):
        """Return v at position p and fractional time λ ∈ [0,1]."""
        # Wrap the position before interpolation
        p_wrapped = wrap_position(p, x_min, x_max, y_min, y_max)
        w = 1.0 - lam
        vx = w * vx_k((p_wrapped[1], p_wrapped[0])) + lam * vx_k1((p_wrapped[1], p_wrapped[0]))
        vy = w * vy_k((p_wrapped[1], p_wrapped[0])) + lam * vy_k1((p_wrapped[1], p_wrapped[0]))
        return np.array([vx, vy], dtype=float)

    k1 = v(pos,           0.0)
    k2 = v(pos + 0.5*dt*k1, 0.5)
    k3 = v(pos + 0.5*dt*k2, 0.5)
    k4 = v(pos +     dt*k3, 1.0)
    return pos + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def wrap_position(pos, x_min, x_max, y_min, y_max):
    """Wrap position to stay within simulation boundaries"""
    x, y = pos
    x_wrapped = x_min + (x - x_min) % (x_max - x_min)
    y_wrapped = y_min + (y - y_min) % (y_max - y_min)
    return np.array([x_wrapped, y_wrapped])

def max_cfl(vx_all, vy_all, dt, dx, dy):
    vmax = np.max(np.sqrt(vx_all**2 + vy_all**2))
    return vmax * dt / min(dx, dy)

def compute_bohmian_trajectories(
        simulation_data, params,
        alpha=1.0+0j, beta=0.0+0j,
        n_trajectories=10, random_seed=0):
    """
    Bohmian trajectories for a 2‑component Pauli spinor.

    The guidance equation uses the *total* probability density

        ρ = |psi_up|² + |psi_down|²         (1)

    and the *total* current density

        j = (ħ/m) Im[ psi_up* ∇psi_up + psi_down* ∇psi_down ]   (2)

    Optional global spin coefficients α, β allow you to examine
    spinors alpha * psi_up + beta * psi_down without re‑running the TDSE.

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
    t = np.asarray(simulation_data['t'])
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
        rng.normal(y0, sigma, n_trajectories) # np.full(n_trajectories, fill_value=y0)
    ])

    traj = np.zeros((n_trajectories, N_t, 2), dtype=float)
    traj[:, 0, :] = starts

    # constants for current
    hbar = params.get('hbar', 1.0)
    m    = params.get('m', 1.0)

    # n_sub = 2                         # Euler sub‑steps per frame - not used anymore
    # sub_dt = dt
    eps = 1e-20                       # to avoid 0/0

    vx_all = np.empty_like(psi_up_all, dtype=float)
    vy_all = np.empty_like(psi_down_all, dtype=float)

    # Rebuild k-space grid
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)      # shape (Nx,)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)      # shape (Ny,)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')# shapes (Ny, Nx)

    # --- pre compute velocities for all frames ---------
    # --- pre compute velocities for all frames ---------
    for k in range(N_t): # how many frames it take to get to the barrier

        # combine spinor with the chosen (α, β)
        # psi_up   = alpha * psi_up_all[k]
        # psi_down = beta  * psi_down_all[k]

        # probability density rho(x,y,t_k)
        # rho = (abs(psi_up)**2 + abs(psi_down)**2) + eps

        # # spatial derivatives (used meshgrid indexing='xy' so axis=1 -> x, axis=0 -> y)
        # dpsi_up_dx   = np.gradient(psi_up,   dx, axis=1)
        # dpsi_up_dy   = np.gradient(psi_up,   dy, axis=0)
        # dpsi_down_dx = np.gradient(psi_down, dx, axis=1)
        # dpsi_down_dy = np.gradient(psi_down, dy, axis=0)

        # pref = hbar / m
        # jx = pref * np.imag(np.conj(psi_up) * dpsi_up_dx +
        #                     np.conj(psi_down) * dpsi_down_dx)
        # jy = pref * np.imag(np.conj(psi_up) * dpsi_up_dy +
        #                     np.conj(psi_down) * dpsi_down_dy)

        # # masking vx/vy where rho (wavefunction probability density) is too small
        # rho_min = 1e-12 * rho.max()
        # # mask    = rho > rho_min
        # # vx      = np.zeros_like(jx)
        # # vy      = np.zeros_like(jy)

        # vx = jx / (rho + rho_min)
        # vy = jy / (rho + rho_min)
        
        # vx[mask] = jx[mask] / rho[mask]
        # vy[mask] = jy[mask] / rho[mask]

        # speed_cap = 10*np.percentile(np.sqrt(vx**2 + vy**2), 99)   # robust cap
        # vx_all[k] = vx # np.clip(vx, -speed_cap, speed_cap)
        # vy_all[k] = vy # np.clip(vy, -speed_cap, speed_cap)

        # ------------------ Okay... so perhaps I need to combine the components first ------------------
        # psi = alpha*psi_up_all[k] + beta*psi_down_all[k]

        # # prob density
        # rho = np.abs(psi)**2
        # rho_min = 1e-12 * rho.max()

        # dpsi_dx = np.gradient(psi, dx, axis=1)   # x-derivative
        # dpsi_dy = np.gradient(psi, dy, axis=0)   # y-derivative  (rows = y)

        # pref = hbar / m
        # jx = pref * np.imag(np.conj(psi) * dpsi_dx)
        # jy = pref * np.imag(np.conj(psi) * dpsi_dy)

        # vx = jx / (rho)
        # vy = jy / (rho)

        # vx_all[k] = vx
        # vy_all[k] = vy

        #  I'm switching to spectral method
        # combine only for density prefactors
        # amp_up, amp_dn = alpha, beta          # already (re)normalised

        psi_up   = psi_up_all[k]
        psi_down = psi_down_all[k]

        rho = np.abs(psi_up)**2 + np.abs(psi_down)**2 + 1e-20


        # spectral momemnta calculation
        psi_k_up = np.fft.fft2(psi_up)
        psi_k_down = np.fft.fft2(psi_down)

        # spatial derivatives
        dpsi_up_dx   = np.fft.ifft2(1j*KX*psi_k_up)
        dpsi_up_dy   = np.fft.ifft2(1j*KY*psi_k_up)
        dpsi_dn_dx   = np.fft.ifft2(1j*KX*psi_k_down)
        dpsi_dn_dy   = np.fft.ifft2(1j*KY*psi_k_down)

        pref = hbar / m
        jx = pref * (
                np.imag(np.conj(psi_up) * dpsi_up_dx) +
                np.imag(np.conj(psi_down) * dpsi_dn_dx)
            )
        jy = pref * (
                np.imag(np.conj(psi_up) * dpsi_up_dy) +
                np.imag(np.conj(psi_down) * dpsi_dn_dy)
            )

        vx_all[k] = jx / rho
        vy_all[k] = jy / rho

    # --- Compute trajectories, using a midpoint method ---------
    # === trajectory loop ======================================================
    # -------- main propagation -----------------------------------------
    # for k in range(N_t-1):

    #     vx_interp = RegularGridInterpolator((y,x), vx_all[k], fill_value=0.0)
    #     vy_interp = RegularGridInterpolator((y,x), vy_all[k], fill_value=0.0)

    #     for n in range(n_trajectories):
    #         p = traj[n,k]
    #         p = euler_or_heun(p, dt, vx_interp, vy_interp, use_heun=True)
    #         # optional periodic wrap
    #         p = wrap_position(p, x[0], x[-1], y[0], y[-1])
    #         traj[n,k+1] = p


    # def step_euler(pos, dt, vxg, vyg):
    #     x, y = pos
    #     vx, vy = velocity_nearest(vxg, vyg, x, y, x_min, y_min, dx, dy)
    #     return pos + dt*np.array([vx, vy])

    # for k in range(N_t-1):
    #     vxg, vyg = vx_all[k], vy_all[k]      # just use frame‑k field
    #     for n in range(n_trajectories):
    #         traj[n,k+1] = wrap_position(step_euler(traj[n,k], dt, vxg, vyg),
    #                                     x_min, x_max, y_min, y_max)

    sub = 4                    # 2 gives trapezoidal, 4 is usually plenty
    sub_dt = dt / sub

    C   = max_cfl(vx_all, vy_all, dt, dx, dy)

    print("CFL parameter:", C)

    target = 0.25              # we want C ≲ 0.25 for safety
    sub = int(np.ceil(C / target))   # 49 sub-steps

    sub_dt = dt / sub

    vmax = np.max(np.sqrt(vx_all**2 + vy_all**2))
    dx   = x[1]-x[0];  dt_eff = dt/sub
    print("C after fix =", vmax*dt_eff/min(dx,dy))   # should print ≤ 0.3

    for k in range(N_t-1):
        vx0 = RegularGridInterpolator((y,x), vx_all[k]  , fill_value=0.0)
        vy0 = RegularGridInterpolator((y,x), vy_all[k]  , fill_value=0.0)
        vx1 = RegularGridInterpolator((y,x), vx_all[k+1], fill_value=0.0)
        vy1 = RegularGridInterpolator((y,x), vy_all[k+1], fill_value=0.0)

        for n in range(n_trajectories):
            traj[n,k+1] = wrap_position(
                rk4_time_interp(traj[n,k], dt, vx0, vy0, vx1, vy1, x_min, x_max, y_min, y_max),
                x_min, x_max, y_min, y_max)

    # substepping to meat CFL condition
    # for k in range(N_t - 1):
    #     # interpolators for the two consecutive snapshots
    #     vx0 = RegularGridInterpolator((y, x), vx_all[k]  , fill_value=0.0)
    #     vy0 = RegularGridInterpolator((y, x), vy_all[k]  , fill_value=0.0)
    #     vx1 = RegularGridInterpolator((y, x), vx_all[k+1], fill_value=0.0)
    #     vy1 = RegularGridInterpolator((y, x), vy_all[k+1], fill_value=0.0)

    #     for n in range(n_trajectories):
    #         p = traj[n, k]

    #         # ------- sub-steps with *time-blended* velocity -----------------
    #         for s in range(sub):
    #             lam = (s + 0.5) / sub        # λ = 0½, 1½, … mid-point of slice
    #             w   = 1.0 - lam
    #             vx  = w * vx0((p[1], p[0])) + lam * vx1((p[1], p[0]))
    #             vy  = w * vy0((p[1], p[0])) + lam * vy1((p[1], p[0]))
    #             p  += sub_dt * np.array([vx, vy])
    #         # ----------------------------------------------------------------

    #         traj[n, k+1] = wrap_position(p, x_min, x_max, y_min, y_max)

    # for k in range(N_t-1):
    #     vx = vx_all[k]; vy = vy_all[k]             # field frozen for sub-steps
    #     vx_int = RegularGridInterpolator((y,x), vx, fill_value=0.0)
    #     vy_int = RegularGridInterpolator((y,x), vy, fill_value=0.0)

    #     for n in range(n_trajectories):
    #         p = traj[n,k]
    #         for _ in range(sub):
    #             p = p + sub_dt * np.array([vx_int((p[1],p[0])),
    #                                     vy_int((p[1],p[0]))])
                
    #         traj[n,k+1] = wrap_position(p, x_min, x_max, y_min, y_max)

    return traj
   
def get_velocity_field(
        simulation_data, params,
        alpha=1.0+0j, beta=0.0+0j,
        n_trajectories=10, random_seed=0):
    """
    Bohmian trajectories for a 2‑component Pauli spinor.

    The guidance equation uses the *total* probability density

        ρ = |psi_up|² + |psi_down|²         (1)

    and the *total* current density

        j = (ħ/m) Im[ psi_up* ∇psi_up + psi_down* ∇psi_down ]   (2)

    Optional global spin coefficients α, β allow you to examine
    spinors alpha * psi_up + beta * psi_down without re‑running the TDSE.

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
    t = np.asarray(simulation_data['t'])
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
        rng.normal(x0, sigma/2, n_trajectories),
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

    vx_all = np.empty_like(psi_up_all, dtype=float)
    vy_all = np.empty_like(psi_up_all, dtype=float)

    # Rebuild k-space grid
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)      # shape (Nx,)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)      # shape (Ny,)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')# shapes (Ny, Nx)

    # --- pre compute velocities for all frames ---------
    for k in range(N_t): # how many frames it take to get to the barrier  

        # combine only for density prefactors
        # amp_up, amp_dn = alpha, beta          # already (re)normalised

        psi_up   = psi_up_all[k]
        psi_down = psi_down_all[k]

        rho = np.abs(psi_up)**2 + np.abs(psi_down)**2 + 1e-20


        # spectral momemnta calculation
        psi_k_up = np.fft.fft2(psi_up)
        psi_k_down = np.fft.fft2(psi_down)

        # spatial derivatives
        dpsi_up_dx   = np.fft.ifft2(1j*KX*psi_k_up)
        dpsi_up_dy   = np.fft.ifft2(1j*KY*psi_k_up)
        dpsi_dn_dx   = np.fft.ifft2(1j*KX*psi_k_down)
        dpsi_dn_dy   = np.fft.ifft2(1j*KY*psi_k_down)

        pref = hbar / m
        jx = pref * (
                np.imag(np.conj(psi_up) * dpsi_up_dx) +
                np.imag(np.conj(psi_down) * dpsi_dn_dx)
            )
        jy = pref * (
                np.imag(np.conj(psi_up) * dpsi_up_dy) +
                np.imag(np.conj(psi_down) * dpsi_dn_dy)
            )

        vx_all[k] = jx / rho
        vy_all[k] = jy / rho

        # # ------------------ Okay... so perhaps I need to combine the components first ------------------
        # psi = alpha*psi_up_all[k] + beta*psi_down_all[k]

        # # prob density
        # rho = np.abs(psi)**2
        # rho_min = 1e-12 * rho.max()

        # dpsi_dx = np.gradient(psi, dx, axis=1)   # x-derivative
        # dpsi_dy = np.gradient(psi, dy, axis=0)   # y-derivative  (rows = y)

        # pref = hbar / m
        # jx = pref * np.imag(np.conj(psi) * dpsi_dx)
        # jy = pref * np.imag(np.conj(psi) * dpsi_dy)

        # vx = jx / (rho)
        # vy = jy / (rho)

        # vx_all[k] = vx
        # vy_all[k] = vy

    return vx_all, vy_all

import numpy as np
import matplotlib.pyplot as plt

def analyze_barrier_passage(traj, t, R, center=(0.0, 0.0)):
    """
    Analyze Bohmian trajectories to find times spent inside a circular barrier.

    Parameters
    ----------
    traj : ndarray, shape (n_trajectories, N_t, 2)
        The x,y positions of each trajectory at each time t.
    t : ndarray, shape (N_t,)
        The time points corresponding to the trajectory frames.
    R : float
        Radius of the circular barrier region.
    center : tuple (x0, y0)
        Center of the circle.

    Returns
    -------
    inside_mask : ndarray, shape (n_trajectories, N_t), dtype=bool
        Mask indicating for each trajectory & time whether it's inside the circle.
    times_inside : ndarray, shape (n_trajectories,)
        Total time each trajectory spent inside the circle.
    avg_time_inside : float
        Average time spent inside over all trajectories.
    """
    x0, y0 = center
    dt = t[1] - t[0]
    # Compute radial distance from center at each frame
    dx = traj[:, :250, 0] - x0
    dy = traj[:, :250, 1] - y0 # using 250 frame before it hits the border of the simulation
    r = np.sqrt(dx**2 + dy**2)

    # Mask of being inside the circle
    inside_mask = (r <= R)

    # Time inside = count of True * dt
    times_inside = inside_mask.sum(axis=1) * dt
    avg_time_inside = times_inside.mean()

    return inside_mask, times_inside, avg_time_inside

def plot_trajectories_with_barrier(traj, inside_mask, R, center=(0.0, 0.0)):
    """
    Plot trajectories, coloring segments inside the barrier in red, outside in blue.
    """
    x0, y0 = center
    fig, ax = plt.subplots(figsize=(6,6))
    # Draw the barrier
    circle = plt.Circle((x0, y0), R, color='k', linestyle='--', fill=False, lw=1.5)
    ax.add_patch(circle)

    n_traj, N_t, _ = traj.shape

    N_t = 250

    for i in range(n_traj):
        mask = inside_mask[i]
        start = 0
        # Segment trajectories by contiguous mask values
        for j in range(N_t):
            if j == N_t-1 or mask[j] != mask[j+1]:
                seg = traj[i, start:j+1]
                color = 'red' if mask[j] else 'blue'
                ax.plot(seg[:,0], seg[:,1], color=color, lw=1)
                start = j+1

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Bohmian Trajectories\n(red = inside barrier)')
    
    return fig, ax

def plot_density_inside(traj, inside_mask, bins=100, extent=None):
    """
    Plot a 2D density (histogram) of all trajectory points that lie inside the barrier.
    """
    # Collect all inside points
    all_inside = np.vstack([traj[i][inside_mask[i]] for i in range(traj.shape[0])])
    if all_inside.size == 0:
        raise ValueError("No points found inside the barrier.")

    H, xedges, yedges = np.histogram2d(
        all_inside[:,0], all_inside[:,1], bins=bins, range=extent
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        H.T,
        origin='lower',
        extent=(
            xedges[0], xedges[-1],
            yedges[0], yedges[-1]
        ),
        aspect='equal',
        interpolation='nearest'
    )
    fig.colorbar(im, ax=ax, label='Counts')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Density of Trajectory Points Inside Barrier')
    plt.show()
