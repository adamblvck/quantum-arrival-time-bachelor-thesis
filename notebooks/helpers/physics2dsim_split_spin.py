import numpy as np
from scipy.fft import fft2, ifft2          # modern pocketfft backend

def simulate_2d_spin(
    x_min=-20, x_max=20, Nx=128,
    y_min=-20, y_max=20, Ny=128,
    n_steps=100, dt=0.01,
    hbar=1.0, m=1.0,
    # Parameters for the magnetic barrier (spin-dependent potential)
    mag_barrier_center_x=0.0,
    mag_barrier_center_y=0.0,
    mag_barrier_strength=5.0,
    mag_barrier_width=0.5,
    barrier_type='gaussian',
    # If you still want "gravity" or any other shared potential
    use_gravity=True,
    g=9.81,
    # Initial wavepacket parameters (same for spin up component)
    x0=0.0, y0=10.0, p0x=0, p0y=0, sigma=1.0
):
    """
    2D quantum simulation with two spin components (spin-1/2).
    Uses the split-operator method for time evolution and includes
    a magnetic barrier that is +V for spin up and -V for spin down.

    Parameters
    ----------
    x_min, x_max, Nx : float, float, int
        Spatial domain endpoints and number of grid points in x.
    y_min, y_max, Ny : float, float, int
        Spatial domain endpoints and number of grid points in y.
    n_steps : int
        Number of time steps.
    dt : float
        Time step.
    hbar, m : float
        Reduced Planck's constant and particle mass.
    mag_barrier_center_x, mag_barrier_center_y : float
        Coordinates of the center of the magnetic barrier.
    mag_barrier_strength : float
        Peak amplitude of the magnetic barrier.
    mag_barrier_width : float
        Gaussian width of the magnetic barrier.
    use_gravity : bool
        Whether or not to include a uniform gravitational potential m*g*y.
    g : float
        Gravitational acceleration (if use_gravity=True).
    x0, y0 : float
        Initial center position of the wavepacket (spin-up component).
    p0x, p0y : float
        Initial momentum components of the wavepacket.
    sigma : float
        Width of the Gaussian wavepacket.

    Returns
    -------
    dict :
        A dictionary with keys:
            "x", "y", "t" : 1D arrays of the spatial and time grids.
            "prob_up", "prob_down" : lists of 2D arrays (|ψ_up|², |ψ_down|²) over time.
            "jx_up", "jy_up" : lists of 2D arrays for probability-current components (spin up).
            "jx_down", "jy_down" : lists of 2D arrays for probability-current components (spin down).
    """

    # --- 1) Set up spatial grids ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')  # shapes: (Nx, Ny)

    # --- 2) Define spin-dependent potential ---
    #     We assume a magnetic barrier that is +barrier for spin up,
    #     and -barrier for spin down.  If gravity is true, we add m*g*Y
    #     for both spin components equally (no spin dependence).


    # --- Gaussian barrier --- 
    magnetic_barrier = np.zeros_like(X)

    if barrier_type == 'gaussian':
        magnetic_barrier = mag_barrier_strength * np.exp(
            -(((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2)
            / (2 * mag_barrier_width**2))
        )
    elif barrier_type == 'square':
        magnetic_barrier = mag_barrier_strength * np.where(
            ((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2) <= mag_barrier_width**2,
            1.0, 0.0
        )
    elif barrier_type == 'circular':    
        magnetic_barrier = mag_barrier_strength * np.where(
            ((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2) <= (mag_barrier_width/2)**2,
            1.0, 0.0
        )
    else:
        raise ValueError(f"Invalid barrier type: {barrier_type}")

    # --- Square Barrier ---
    # magnetic_barrier = mag_barrier_strength * np.where(
    #     ((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2) <= mag_barrier_width**2,
    #     1.0, 0.0
    # )

    # --- Circular Barrier ---
    # magnetic_barrier = mag_barrier_strength * np.where(
    #     ((X - mag_barrier_center_x)**2 + (Y - mag_barrier_center_y)**2) <= (mag_barrier_width/2)**2,
    #     1.0, 0.0
    # )

    if use_gravity:
        common_potential = m * g * Y  # same for both spins
    else:
        common_potential = 0.0

    # Spin-up sees V_up = common_potential + magnetic_barrier
    # Spin-down sees V_down = common_potential - magnetic_barrier
    V_up = common_potential + magnetic_barrier
    V_down = common_potential - magnetic_barrier

    # --- 3) Initialize the two-component wavefunction ---
    #     Spin up: a 2D Gaussian wavepacket
    #     Spin down: zero (you can adjust as you like for superpositions)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    gauss_envelope = norm * np.exp(
        -((X - x0)**2 + (Y - y0)**2) / (4.0 * sigma**2)
    )
    plane_wave = np.exp(1j * (p0x * (X - x0) + p0y * (Y - y0)) / hbar)

    # Initialize a 50-50 superposition state
    # Factor of 1/√2 ensures each component has 50% probability
    psi_up = (1/np.sqrt(2)) * gauss_envelope * plane_wave
    psi_down = (1/np.sqrt(2)) * gauss_envelope * plane_wave

    # --- 4) Build momentum space grids for the kinetic operator ---
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    # Kinetic phase factor: exp(-i * (hbar^2 k^2)/(2m) * dt / hbar) => exp(-i dt hbar k^2 / 2m)
    kinetic_phase = np.exp(-1j * dt * hbar * (KX**2 + KY**2) / (2.0 * m))

    # --- 5) Potential phase factors for half-step evolution ---
    # We do split-operator: U(dt) = exp(-i V dt/2hbar) exp(-i T dt/hbar) exp(-i V dt/2hbar)
    potential_phase_up = np.exp(-1j * V_up * dt / (2.0 * hbar))
    potential_phase_down = np.exp(-1j * V_down * dt / (2.0 * hbar))

    # --- 6) Time evolution storage ---
    t_arr = np.arange(n_steps) * dt


    # wavefunction storage
    psi_up_list = []
    psi_down_list = []

    # probability densities
    prob_up_storage = []
    prob_down_storage = []

    # Store wavefunctions - need to copy to avoid overwriting as psi_up and psi_down evolve over time
    psi_up_list.append(psi_up.copy())
    psi_down_list.append(psi_down.copy())

    # --- 7) Main time evolution loop ---
    for _ in range(n_steps):

        # Store probability densities
        prob_up_storage.append(np.abs(psi_up)**2)
        prob_down_storage.append(np.abs(psi_down)**2)

        # --- First half-step: potential operator ---
        psi_up *= potential_phase_up
        psi_down *= potential_phase_down

        # --- Full kinetic step: FFT -> multiply -> IFFT for each spin ---
        psi_up = ifft2( kinetic_phase * fft2(psi_up, workers=-1), workers=-1)
        psi_down = ifft2(kinetic_phase * fft2(psi_down, workers=-1), workers=-1)

        # --- Second half-step: potential operator ---
        psi_up *= potential_phase_up
        psi_down *= potential_phase_down

        # -- apply cap / boundaray absorption if needed - please
        #
        #
        #

        # Store wavefunctions - need to copy to avoid overwriting as psi_up and psi_down evolve over time
        psi_up_list.append(psi_up.copy())
        psi_down_list.append(psi_down.copy())


    # --- 8) Return data ---
    return {
        "x": x.tolist(), "y": y.tolist(), "t": t_arr.tolist(),
        
        "prob_up": [p.tolist() for p in prob_up_storage], # real
        "prob_down": [p.tolist() for p in prob_down_storage], # real
        
        "psi_up_list": [p.tolist() for p in psi_up_list],         # complex arrays
        "psi_down_list": [p.tolist() for p in psi_down_list],     # complex arrays
    }
