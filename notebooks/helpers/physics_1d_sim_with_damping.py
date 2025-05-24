import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fft import fft, ifft, fftfreq
from helpers.absorber_functions import *
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ------------------
# 2) Potential Functions
# ------------------
def gravity_potential(m, z, g=1.0):
    return m * g * z

def delta_barrier(z, z0=0.0, alpha=5.0, dz=0.01):
    """Approximate a delta potential by a very narrow Gaussian."""
    sigma = dz / 2
    return alpha * np.exp(-((z - z0)**2)/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

def gaussian_barrier(z, z0=0.0, V0=5.0, sigma=0.5):
    """A Gaussian potential barrier centered at z0."""
    return V0 * np.exp(-(z - z0)**2/(2*sigma**2))

# ------------------
# 3) Initial Wavefunction
# ------------------
def initial_wavefunction(z, z0=10.0, p0=0.0, sigma0=1.0):
    """Gaussian wave packet with center z0, momentum p0, width sigma0."""
    norm_factor = 1.0 / (2.0 * np.pi * sigma0**2)**0.25
    return norm_factor \
           * np.exp(-(z - z0)**2 / (4.0 * sigma0**2)) \
           * np.exp(1j * p0 * (z - z0) / hbar)

# ------------------
# 4) Split-Operator Step
# ------------------
def split_operator_step(psi, V, k, m, dt_seg):
    """Perform one split-operator time step: half potential -> full kinetic -> half potential."""
    # half step potential
    psi *= np.exp(-0.5j * V * dt_seg / hbar)

    # full-drift in momentum space
    psi_k  = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * hbar * k**2 * dt_seg / m)   # e^{-i T dt / ħ}
    psi    = np.fft.ifft(psi_k)

    # another half step  potential
    psi *= np.exp(-0.5j * V * dt_seg / hbar)
    return psi

# ------------------
# 4) Yoshida-Step
# ------------------

hbar = 1.0 # or whatever you are using
y1 = 1.0 / (2.0 - 2.0**(1.0/3.0)) #  +1.3512071919596578
y2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))  #  -1.7024143839193153
YOSHIDA4_COEFFS = (y1, y2, y1) # symmetric → 4-th order

def yoshida4_step(psi, V, k, m, dt):
    """
    Forest–Ruth / Yoshida explicit 4-th-order symplectic step.
    
    Performs three symmetric Strang sub-steps with coefficients
    (y1, y2, y1) where y2 < 0 is unavoidable for any explicit 4-th-order
    method of this type.
    """
    for c in YOSHIDA4_COEFFS:
        psi = split_operator_step(psi, V, k, m, c * dt)
    return psi

def apply_parabolic_dwell_potential(V_tot, z, z_dwell_start, z_bin_end, strength=40, g=1):
    """
    Z-Dwell time starts at -z_dwell_start and ends at 0.

    """
    c = -strength
    D = z_dwell_start
    a = c / D**2
    b = g - 2*D*a

    V_tot[:z_bin_end] = a*z[:z_bin_end]**2 + b*z[:z_bin_end] + c

    return V_tot

# ------------------
# 5) Simulation Function
# ------------------

def simulate_n(
    *,
    barrier_type: str = "delta",
    n_steps: int = 200,
    dt: float = 0.01,
    x_min: float = -20,
    x_max: float = 20,
    Nx: int = 512,
    apply_parabolic_dwell: bool = False,
    z_dwell_start: float = -10.0,
    strength: float = 40.0,
    barrier_params: Optional[dict] = None,
    z0_packet: float = 10.0,
    p0_packet: float = 0.0,
    sigma0_packet: float = 1.0,
    absorber_type: str = "none",
    absorber_params: Optional[dict] = None,
    m: float = 1.0,
    g: float = 1.0,
    use_yoshida: bool = False,
):
    """Propagate ψ and track absorption.

    Returns
    -------
    z : ndarray, shape (Nx,)
        Spatial grid.
    t_vals : ndarray, shape (n_steps,)
        Times at which |ψ|² is stored.
    prob : ndarray, shape (n_steps, Nx)
        Probability density |ψ|² at each time‑step *before* absorption of
        that step is applied.
    absorbed_frac : ndarray, shape (n_steps,)
        Cumulative fraction of the initial probability that has been
        absorbed up to and including each step.
    """
    if barrier_params is None:
        barrier_params = {}
    if absorber_params is None:
        absorber_params = {}

    # ------------------------------------------------------------------
    # Grid
    z = np.linspace(x_min, x_max, Nx)
    dz = z[1] - z[0]

    # ------------------------------------------------------------------
    # External + barrier potential (placeholder helpers provided by user)
    if barrier_type == "delta":
        V_bar = delta_barrier(z, **barrier_params, dz=dz)
    else:
        V_bar = gaussian_barrier(z, **barrier_params)
    V_tot = gravity_potential(m, z, g) + V_bar

    if apply_parabolic_dwell:
        V_tot = apply_parabolic_dwell_potential(V_tot, z, z_dwell_start, int(Nx/2+z_dwell_start/dz), strength=strength)

    # ------------------------------------------------------------------
    # Absorber
    absorber = build_absorber(z, absorber_type=absorber_type,
                              x_min=x_min, x_max=x_max,
                              **absorber_params)
    if absorber.cap is not None:
        V_tot = V_tot - 1j * absorber.cap  # add once to hamiltonian

    # ------------------------------------------------------------------
    # Kinetic factor in k‑space
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dz)

    # ------------------------------------------------------------------
    # Initial wavefunction
    psi = initial_wavefunction(z, z0=z0_packet, p0=p0_packet, sigma0=sigma0_packet)
    absorber.begin(psi, dz)

    # ------------------------------------------------------------------
    # Storage arrays
    prob = np.empty((n_steps, Nx), dtype=np.float64)
    absorbed_frac = np.empty(n_steps, dtype=np.float64)
    psi_history = np.empty((n_steps, Nx), dtype=np.complex128)

    # ------------------------------------------------------------------
    # Time loop
    for step in range(n_steps):
        # record probability density *before* this step's absorption
        prob[step] = np.abs(psi) ** 2
        psi_history[step] = psi.copy()

        # split‑operator step
        if use_yoshida:
            psi = yoshida4_step(psi, V_tot, k, m, dt)
        else:
            psi = split_operator_step(psi, V_tot, k, m, dt)

        # multiplicative mask, if any  (absorbs *after* evolution for step)
        if absorber.mask is not None:
            psi *= absorber.mask

        absorbed_frac[step] = absorber.update(psi, dz)

    # ------------------------------------------------------------------
    t_vals = np.arange(n_steps) * dt
    return z, t_vals, prob, absorbed_frac, psi_history, V_tot

# ------------------
# 6) Simulation Function Yoshida - dt^4th order
# ------------------
def simulate_n_yoshida(barrier_type='delta',
                       n_steps=200,
                       dt=0.01,
                       x_min=-20, x_max=20, Nx=512,
                       barrier_params=None,
                       z0_packet=10.0, p0_packet=0.0):
    if barrier_params is None:
        barrier_params = {}

    # grid
    z = np.linspace(x_min, x_max, Nx)
    dz = z[1] - z[0]

    # potential
    if barrier_type == 'delta':
        Vb = delta_barrier(z, **barrier_params, dz=dz)
    else:
        Vb = gaussian_barrier(z, **barrier_params)
    V = gravity_potential(z) + Vb

    # precompute k^2
    k   = 2*np.pi * fftfreq(Nx, d=dz)
    ksq = k**2

    # init wavepacket
    psi = initial_wavefunction(z, z0=z0_packet, p0=p0_packet, sigma0=1.0)
    prob = np.zeros((n_steps, Nx))

    # time evolution
    for t in range(n_steps):
        prob[t] = np.abs(psi)**2
        psi     = yoshida_step(psi, V, ksq, dt)

    t_vals = np.arange(n_steps) * dt
    return z, t_vals, prob


# ------------------
# x) Calculate probability current at Z_0
# ------------------
def calculate_probability_current(psi_history, z_array, hbar=1, m=1):
    """
    Calculate the probability current at Z_0.
    """

    prob_current = (hbar / (m)) * np.imag(
        np.conj(psi_history) * np.gradient(psi_history, axis=0)
    )

    return prob_current

# ------------------
# 6) 2D Heatmap Plot
# ------------------
def plot_spacetime(prob_arr, z_array, t_vals, title=None, log_scale=False, z_min=None, z_max=None):
    """
    Produce a heatmap of |psi(z,t)|^2 over space (x-axis) and time (y-axis).
    
    prob_arr : shape (N_time, N_space)
               prob_arr[i, j] = |psi(z_j, t_i)|^2
    z_array  : spatial grid, shape (N_space,)
    t_vals   : time array, shape (N_time,)
    z_min    : minimum z value to display (optional)
    z_max    : maximum z value to display (optional)
    """
    # Use provided z_min/z_max if specified, otherwise use full range
    z_min = z_array[0] if z_min is None else z_min
    z_max = z_array[-1] if z_max is None else z_max
    t_min, t_max = t_vals[0], t_vals[-1]

    plt.figure(figsize=(8,6))
    plt.imshow(
        prob_arr.T,
        extent=[t_min, t_max, z_array[0], z_array[-1]],
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )
    if log_scale:
        plt.colorbar(label=r'$|\psi(z,t)|^2$', norm=LogNorm())
    else:
        plt.colorbar(label=r'$|\psi(z,t)|^2$')
    
    # Set the z-axis limits explicitly
    plt.ylim(z_min, z_max)
    
    plt.xlabel('t')
    plt.ylabel('z')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Probability Density')
    plt.show()

def create_quantum_evolution_video(
    prob_arr, 
    z_array, 
    t_vals, 
    output_filename='quantum_evolution.mp4',
    z_range=None,
    t_range=None,
    fps=30,
    dpi=150,
    log_scale=False
):
    """
    Create a video showing the quantum evolution with a heatmap and probability distribution.
    
    Parameters:
    -----------
    prob_arr : ndarray, shape (N_time, N_space)
        Probability density array where prob_arr[i, j] = |psi(z_j, t_i)|^2
    z_array : ndarray, shape (N_space,)
        Spatial grid points
    t_vals : ndarray, shape (N_time,)
        Time values
    output_filename : str
        Name of the output MP4 file
    z_range : tuple, optional
        (z_min, z_max) for plotting range. If None, uses full range
    t_range : tuple, optional
        (t_min, t_max) for plotting range. If None, uses full range
    fps : int
        Frames per second for the output video
    dpi : int
        Resolution of the output video
    log_scale : bool
        Whether to use logarithmic scale for the heatmap
    """
    # Handle ranges
    z_min, z_max = z_range if z_range is not None else (z_array[0], z_array[-1])
    t_min, t_max = t_range if t_range is not None else (t_vals[0], t_vals[-1])
    
    # Create mask for z and t ranges
    z_mask = (z_array >= z_min) & (z_array <= z_max)
    t_mask = (t_vals >= t_min) & (t_vals <= t_max)
    
    # Filter arrays based on ranges
    z_plot = z_array[z_mask]
    t_plot = t_vals[t_mask]
    prob_plot = prob_arr[t_mask][:, z_mask]
    
    # Set up the figure with 21:9 aspect ratio
    fig = plt.figure(figsize=(21, 9))
    gs = GridSpec(1, 2, width_ratios=[1, 2], figure=fig)
    
    # Create subplots
    ax_dist = fig.add_subplot(gs[0])  # Probability distribution
    ax_heat = fig.add_subplot(gs[1])  # Heatmap
    
    # Initialize heatmap
    if log_scale:
        heatmap = ax_heat.imshow(
            prob_plot.T,
            extent=[t_min, t_max, z_min, z_max],
            aspect='auto',
            cmap='viridis',
            origin='lower',
            norm=LogNorm()
        )
    else:
        heatmap = ax_heat.imshow(
            prob_plot.T,
            extent=[t_min, t_max, z_min, z_max],
            aspect='auto',
            cmap='viridis',
            origin='lower'
        )
    
    # Add colorbar
    plt.colorbar(heatmap, ax=ax_heat, label=r'$|\psi(z,t)|^2$')
    
    # Set up probability distribution plot
    line, = ax_dist.plot([], [], 'b-', lw=2)
    ax_dist.set_xlim(z_min, z_max)
    ax_dist.set_ylim(0, 1.1 * np.max(prob_plot))
    
    # Set labels
    ax_dist.set_xlabel('z')
    ax_dist.set_ylabel(r'$|\psi(z)|^2$')
    ax_heat.set_xlabel('t')
    ax_heat.set_ylabel('z')
    
    # Add vertical line to heatmap
    time_line = ax_heat.axvline(x=t_min, color='r', ls='--')
    
    def animate(frame):
        current_time = t_plot[frame]
        
        # Update probability distribution plot
        line.set_data(z_plot, prob_plot[frame])
        
        # Update time line
        time_line.set_xdata([current_time, current_time])
        
        # Update title with current time
        ax_heat.set_title(f't = {current_time:.2f}')
        
        return line, time_line
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=len(t_plot),
        interval=1000/fps,  # interval in milliseconds
        blit=True
    )
    
    # Save animation
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=dpi)
    plt.close()