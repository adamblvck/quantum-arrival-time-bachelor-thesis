import numpy as np
import matplotlib.pyplot as plt

def measure_precession_angle(sim_data, dx, dy, hbar=1.0, calc_sz=False):
    """
    Given sim_data which contains lists of complex wavefunction arrays
    psi_up_list[t], psi_down_list[t], compute:
       - <sigma_x>(t)
       - <sigma_y>(t)
       - theta_x(t) = atan2(<sigma_y>, <sigma_x>)
    for each time step.
    Return arrays time_list, sx_list, sy_list, theta_list
    that you can plot or animate.
    """

    psi_up_list   = sim_data["psi_up_list"]   # shape [t][Nx,Ny] (complex)
    psi_down_list = sim_data["psi_down_list"] # shape [t][Nx,Ny] (complex)
    time_list     = sim_data["t"]             # shape [n_steps]

    n_steps = len(time_list)
    sx_list = np.zeros(n_steps, dtype=np.float64)
    sy_list = np.zeros(n_steps, dtype=np.float64)
    sz_list = np.zeros(n_steps, dtype=np.float64)
    theta_list = np.zeros(n_steps, dtype=np.float64)

    # We assume wavefunction is normalized so that sum(|psi|^2)*dx*dy = 1
    # Then <sigma_x> in [-1,+1], <sigma_y> in [-1,+1].

    for i in range(n_steps):
        psi_up   = psi_up_list[i]
        psi_down = psi_down_list[i]

        # Overlap for cross terms:
        #   overlap = ∫ psi_up^*(r) psi_down(r) dr
        overlap_x = np.sum(np.conjugate(psi_up) * psi_down + np.conjugate(psi_down) * psi_up)
        overlap_y = np.sum(np.conjugate(psi_up) * psi_down - np.conjugate(psi_down) * psi_up)
        # multiply by dx*dy for the integral
        overlap_x *= (dx*dy)
        overlap_y *= (dx*dy)

        # <sigma_x> = overlap_up_dn + conj => 2 Re(overlap)
        sig_x = overlap_x.real
        # <sigma_y> = 2 Im(overlap)
        sig_y = overlap_y.imag

        
        
        sx_list[i] = sig_x
        sy_list[i] = sig_y

        # complex retrieve angle
        theta_list[i] = np.angle(sig_x + 1j * sig_y)
        # theta_unwrapped = np.unwrap(theta)

        # theta_list[i] = np.arctan2(sig_y, sig_x)  # angle in [-pi, pi]


        if calc_sz == True:
            sz_list[i] = np.sum((np.abs(psi_up)**2 - np.abs(psi_down)**2)) * dx * dy

    if calc_sz == True:
        return time_list, sx_list, sy_list, sz_list, theta_list
    else:
        return time_list, sx_list, sy_list, theta_list


def plot_precession(time_list, sx_list, sy_list, theta_list):
    """Make some simple plots of the net spin over time."""
    fig, axs = plt.subplots(3, 1, figsize=(6,8), sharex=True)

    axs[0].plot(time_list, sx_list, label=r'$\langle \sigma_x \rangle$')
    axs[0].legend(loc='best'); axs[0].grid(True)

    axs[1].plot(time_list, sy_list, label=r'$\langle \sigma_y \rangle$')
    axs[1].legend(loc='best'); axs[1].grid(True)

    axs[2].plot(time_list, theta_list, label=r'Precession angle $\theta$')
    axs[2].legend(loc='best'); axs[2].grid(True)
    axs[2].set_xlabel("Time")

    fig.tight_layout()

    return fig, axs

def calc_larmor_frequency(params, hbar=1.0):
	"""
	Compute the approximate Larmor frequency (omega_L) from simulation parameters.
	
	The code uses 'mag_barrier_strength' as the potential amplitude for spin-up,
	and the opposite sign for spin-down, producing an energy splitting of
	2 * mag_barrier_strength at the barrier peak. Then:
		omega_L = (energy difference) / hbar = 2 * mag_barrier_strength / hbar
	
	Parameters
	----------
	params : dict
		A dictionary with simulation parameters, expected to include:
			'mag_barrier_strength': float
				Peak amplitude of the spin-dependent barrier potential.
	hbar : float
		Reduced Planck's constant in our chosen units (default 1.0).
	
	Returns
	-------
	float
		The Larmor frequency in the same time units used by the simulation.
	"""
	v0 = params['mag_barrier_strength']
	delta_E = 2.0 * v0  # spin-up sees +v0, spin-down sees -v0
	omega_L = delta_E / hbar
	return omega_L


def weighted_B_and_larmor(sim_data, params,
                          t0_idx, t1_idx,
                          hbar: float = 1.0):
    """
    Probability-weighted magnetic field ⟨B⟩ and average
    Larmor frequency ⟨ω_L⟩ over the time-slice [t0_idx, t1_idx].

    Parameters
    ----------
    sim_data : dict
        Output of `simulate_2d_spin` that contains the lists
        'psi_up_list', 'psi_down_list', 'x', 'y'.
    params : dict
        Parameter dictionary used in the simulation (needs the
        *same* keys you passed to `simulate_2d_spin`,
        at least the barrier-related ones).
    t0_idx, t1_idx : int
        Indices of the frames that delimit the time window
        (inclusive).  They can be the same if you want a
        single-snapshot estimate.
    hbar : float, optional
        ℏ in the simulation units (default 1.0).

    Returns
    -------
    (B_avg, omega_L_avg) : tuple(float, float)
        Probability-weighted field and the corresponding
        average Larmor frequency.
    """

    # ---- 1. set up spatial grid and B(x,y) --------------------------
    x = np.asarray(sim_data["x"])
    y = np.asarray(sim_data["y"])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')

    B0   = params["mag_barrier_strength"]
    xc   = params.get("mag_barrier_center_x", 0.0)
    yc   = params.get("mag_barrier_center_y", 0.0)
    w    = params["mag_barrier_width"]
    typ  = params.get("barrier_type", "gaussian")

    if typ == "gaussian":
        B = B0 * np.exp(-((X-xc)**2 + (Y-yc)**2)/(2.0*w**2))
    elif typ == "square":
        B = B0 * np.where(((X-xc)**2 + (Y-yc)**2) <= w**2, 1.0, 0.0)
    elif typ == "circular":
        B = B0 * np.where(((X-xc)**2 + (Y-yc)**2) <= (w/2.0)**2, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown barrier_type {typ!r}")

    dA   = dx * dy                       # area element
    N1   = 0.0                           # ∑|ψ|² B dA dt (numerator)
    N0   = 0.0                           # ∑|ψ|² dA dt   (denominator)

    up_list   = sim_data["psi_up_list"]
    down_list = sim_data["psi_down_list"]

    # ---- 2. accumulate over the chosen frames ----------------------
    for idx in range(t0_idx, t1_idx + 1):
        psi_up   = np.asarray(up_list[idx])
        psi_down = np.asarray(down_list[idx])
        rho      = np.abs(psi_up)**2 + np.abs(psi_down)**2  # total probability
        N1      += np.sum(rho * B) * dA
        N0      += np.sum(rho)      * dA

    if N0 == 0:
        raise RuntimeError("Total probability in selected frames is zero.")

    B_avg       = N1 / N0
    omega_L_avg = 2.0 * B_avg / hbar    #   ΔE = 2B  ⇒  ω = ΔE/ℏ

    return B_avg, omega_L_avg

def frame_weighted_B_and_larmor(sim_data, params, hbar: float = 1.0):
    """
    Probability-weighted magnetic field ⟨B⟩ and average
    Larmor frequency ⟨ω_L⟩ over the time-slice [t0_idx, t1_idx].

    Parameters
    ----------
    sim_data : dict
        Output of `simulate_2d_spin` that contains the lists
        'psi_up_list', 'psi_down_list', 'x', 'y'.
    params : dict
        Parameter dictionary used in the simulation (needs the
        *same* keys you passed to `simulate_2d_spin`,
        at least the barrier-related ones).
    t0_idx, t1_idx : int
        Indices of the frames that delimit the time window
        (inclusive).  They can be the same if you want a
        single-snapshot estimate.
    hbar : float, optional
        ℏ in the simulation units (default 1.0).

    Returns
    -------
    (B_avg, omega_L_avg) : tuple(float, float)
        Probability-weighted field and the corresponding
        average Larmor frequency.
    """

    # ---- 1. set up spatial grid and B(x,y) --------------------------
    t = np.asarray(sim_data["t"])
    x = np.asarray(sim_data["x"])
    y = np.asarray(sim_data["y"])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')

    B0   = params["mag_barrier_strength"]
    xc   = params.get("mag_barrier_center_x", 0.0)
    yc   = params.get("mag_barrier_center_y", 0.0)
    w    = params["mag_barrier_width"]
    typ  = params.get("barrier_type", "gaussian")

    if typ == "gaussian":
        B = B0 * np.exp(-((X-xc)**2 + (Y-yc)**2)/(2.0*w**2))
    elif typ == "square":
        B = B0 * np.where(((X-xc)**2 + (Y-yc)**2) <= w**2, 1.0, 0.0)
    elif typ == "circular":
        B = B0 * np.where(((X-xc)**2 + (Y-yc)**2) <= (w/2.0)**2, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown barrier_type {typ!r}")

    dA   = dx * dy                       # area element

    up_list   = sim_data["psi_up_list"]
    down_list = sim_data["psi_down_list"]

    N1 = np.zeros(len(t))
    N0 = np.zeros(len(t))

    # ---- 2. accumulate over the chosen frames ----------------------
    for idx in range(len(t)):
        psi_up   = np.asarray(up_list[idx])
        psi_down = np.asarray(down_list[idx])
        rho      = np.abs(psi_up)**2 + np.abs(psi_down)**2  # total probability

        # vector summing
        N1[idx] = np.sum(rho * B) * dA
        N0[idx] = np.sum(rho) * dA + 1e-24

    B_avg = N1 / N0
    omega_L_avg = 2.0 * B_avg / hbar    #   ΔE = 2B  ⇒  ω = ΔE/ℏ

    return B_avg, omega_L_avg