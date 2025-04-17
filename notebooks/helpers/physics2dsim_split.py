import numpy as np

def applyCap(psi, X, Y, x_min, x_max, y_min, y_max, dt, hbar, W0=1.0):
    """
    Apply a complex absorbing potential (CAP) to a 2D wavefunction.
    
    The CAP is defined such that in the central region (inside a margin) no
    absorption occurs, while near the edges the absorption grows quadratically.
    
    Parameters:
      psi      : 2D numpy array, the current wavefunction.
      X, Y     : 2D numpy arrays (meshgrids) for spatial coordinates.
      x_min, x_max, y_min, y_max : floats, the physical domain limits.
      dt       : float, the time step.
      hbar     : float, the reduced Planck constant.
      W0       : float, CAP strength parameter.
      
    Returns:
      psi      : 2D numpy array, the wavefunction after applying the CAP.
    """
    # Define margins (here 25% of the domain size in each direction)
    margin_x = 0.15 * (x_max - x_min)
    margin_y = 0.15 * (y_max - y_min)
    
    # Create arrays for the distance from the inner boundary of the absorbing region.
    # In the x-direction, the nonabsorbing region is [x_min + margin_x, x_max - margin_x]
    d_x = np.zeros_like(X)
    d_y = np.zeros_like(Y)
    
    # For x: left side absorption if x < x_min + margin_x
    mask_left = X < (x_min + margin_x)
    d_x[mask_left] = (x_min + margin_x) - X[mask_left]
    # For x: right side absorption if x > x_max - margin_x
    mask_right = X > (x_max - margin_x)
    d_x[mask_right] = X[mask_right] - (x_max - margin_x)
    
    # For y: bottom side absorption if y < y_min + margin_y
    mask_bottom = Y < (y_min + margin_y)
    d_y[mask_bottom] = (y_min + margin_y) - Y[mask_bottom]
    # For y: top side absorption if y > y_max - margin_y
    mask_top = Y > (y_max - margin_y)
    d_y[mask_top] = Y[mask_top] - (y_max - margin_y)
    
    # Define a quadratic CAP profile. Here we normalize the distances by half the margin.
    # The total CAP is taken as the sum of the contributions in x and y.
    W = W0 * ( (d_x/(margin_x/2))**2 + (d_y/(margin_y/2))**2 )
    
    # Compute the multiplicative factor coming from the CAP.
    factor = np.exp(-W * dt / hbar)
    
    # Apply the factor elementwise.
    psi *= factor
    return psi

def simulate_2d(x_min=-20, x_max=20, Nx=128,
                y_min=-20, y_max=20, Ny=128,
                n_steps=100, dt=0.01,
                hbar=1.0, m=1.0,
                barrier_center_y=0.0, barrier_strength=5.0, barrier_width=0.1,
                x0=-5.0, y0=-5.0, p0x=2.0, p0y=2.0, sigma=1.0,
                cap_strength=1.0):
    """
    2D quantum simulation using the split-operator method with a CAP.
    
    (Parameters as before, with an added 'cap_strength' to control the CAP.)
    """
    # Create the spatial grids.
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')  # X, Y shape: (Nx, Ny)
    
    # Define the potential V(x,y):
    # A Gaussian barrier along x (varying in y) plus gravitational potential in y.
    g = 9.81  # gravitational acceleration
    barrier = barrier_strength * np.exp(-((Y - barrier_center_y)**2) / (2 * barrier_width**2))
    V = barrier + m * g * Y  # gravity: V_grav = m*g*Y
    
    # Initial wavefunction: a 2D Gaussian wavepacket.
    norm = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    psi = norm * np.exp(-(((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2))) \
          * np.exp(1j * (p0x * (X - x0) + p0y * (Y - y0)) / hbar)
    
    # Build momentum space grids.
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Precompute phase factors for the kinetic and potential steps.
    kinetic_phase = np.exp(-1j * dt * hbar * (KX**2 + KY**2) / (2 * m))
    potential_phase = np.exp(-1j * V * dt / (2 * hbar))
    
    # Set up time array and storage for probability density.
    t_arr = np.arange(n_steps) * dt
    prob_storage = []
    
    # (Optional) Storage for probability current components.
    jx_storage = []
    jy_storage = []

    # Main time evolution loop using the split-operator method.
    for _ in range(n_steps):
        # Save the probability density |psi|^2 at the current time.
        prob_storage.append(np.abs(psi)**2)
        
        # Compute probability current components using central differences.
        dpsi_dx = np.gradient(psi, dx, axis=0)
        dpsi_dy = np.gradient(psi, dy, axis=1)
        jx = (hbar/(2*m)) * np.imag(np.conj(psi) * dpsi_dx)
        jy = (hbar/(2*m)) * np.imag(np.conj(psi) * dpsi_dy)
        jx_storage.append(jx)
        jy_storage.append(jy)
        
        # First half-step: potential operator.
        psi = potential_phase * psi
        # Full kinetic step: FFT -> multiply kinetic phase -> IFFT.
        psi = np.fft.ifft2( kinetic_phase * np.fft.fft2(psi) )
        # Second half-step: potential operator.
        psi = potential_phase * psi
        
        # Apply the CAP to absorb outgoing components.
        psi = applyCap(psi, X, Y, x_min, x_max, y_min, y_max, dt, hbar, W0=cap_strength)
    
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "t": t_arr.tolist(),
        "prob": [p.tolist() for p in prob_storage],
        "jx": [j.tolist() for j in jx_storage],
        "jy": [j.tolist() for j in jy_storage]
    }


import numpy as np

def simulate_2d_test_2(x_min=-20, x_max=20, Nx=128,
                y_min=-20, y_max=20, Ny=128,
                n_steps=100, dt=0.01,
                hbar=1.0, m=1.0,
                barrier_center_y=0.0, barrier_strength=5.0, barrier_width=0.1,
                x0=-5.0, y0=-5.0, p0x=0.0, p0y=0.0, sigma=1.0):
    """
    2D quantum simulation using the split-operator method.

    Parameters:
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
      barrier_center_x, barrier_strength, barrier_width : float
          Parameters for a Gaussian barrier in the x-direction.
      x0, y0 : float
          Initial center position of the wavepacket.
      p0x, p0y : float
          Initial momentum components.
      sigma : float
          Width of the Gaussian wavepacket.

    Returns:
      dict: with keys "x", "y", "t", and "prob" (the probability density evolution).
    """
    # Create the spatial grids.
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')  # X, Y shape: (Nx, Ny)
    
    # Define the potential V(x,y):
    # A Gaussian barrier along x plus gravitational potential in y.
    g = 9.81  # gravitational acceleration
    barrier = barrier_strength * np.exp(-((Y - barrier_center_y)**2) / (2 * barrier_width**2))
    V = barrier + m * g * Y  # gravity acts along the y-direction.
    
    # Initial wavefunction: a 2D Gaussian wavepacket.
    # Normalization factor for a 2D Gaussian is chosen so that ∫|psi|^2 dx dy = 1.
    norm = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    psi = norm * np.exp(-(((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2))) \
          * np.exp(1j * (p0x * (X - x0) + p0y * (Y - y0)) / hbar)
    
    # Build momentum space grids.
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    
    # Precompute phase factors for the kinetic and potential steps.
    kinetic_phase = np.exp(-1j * dt * hbar * (KX**2 + KY**2) / (2 * m))
    potential_phase = np.exp(-1j * V * dt / (2 * hbar))
    
    # Set up time array and storage for probability density.
    t_arr = np.arange(n_steps) * dt
    prob_storage = []
    
    # Main time evolution loop using the split-operator method.
    for _ in range(n_steps):
        # Save the probability density |psi|^2 at the current time.
        prob_storage.append(np.abs(psi)**2)
        
        # First half-step: potential operator.
        psi = potential_phase * psi
        # Full kinetic step: transform to momentum space, apply phase, then transform back.
        psi = np.fft.ifft2( kinetic_phase * np.fft.fft2(psi) )
        # Second half-step: potential operator.
        psi = potential_phase * psi

		# Apply the CAP to absorb outgoing components.
        psi = applyCap(psi, X, Y, x_min, x_max, y_min, y_max, dt, hbar, W0=2)	

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "t": t_arr.tolist(),
        "prob": [p.tolist() for p in prob_storage]
    }

def simulate_2d_2_particles(x_min=-20, x_max=20, Nx=128,
				y_min=-20, y_max=20, Ny=128,
				n_steps=100, dt=0.01,
				hbar=1.0, m=1.0,
				barrier_center_y=0.0, barrier_strength=5.0, barrier_width=0.1,
				x0=-5.0, y0=-5.0, p0x=2.0, p0y=2.0, sigma=1.0):
	"""
	2D quantum simulation using the split-operator method.

	Parameters:
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
	  barrier_center_y, barrier_strength, barrier_width : float
		  Parameters for a Gaussian barrier in the x-direction.
	  x0, y0 : float
		  Initial center position of the wavepacket.
	  p0x, p0y : float
		  Initial momentum components.
	  sigma : float
		  Width of the Gaussian wavepacket.

	Returns:
	  dict: with keys "x", "y", "t", and "prob" (the probability density evolution).
	"""
	# Create the spatial grids.
	x = np.linspace(x_min, x_max, Nx, endpoint=False)
	y = np.linspace(y_min, y_max, Ny, endpoint=False)
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	X, Y = np.meshgrid(x, y, indexing='ij')  # X, Y shape: (Nx, Ny)
	
	# Define the potential V(x,y):
	# A Gaussian barrier along x plus gravitational potential in y.
	g = 9.81  # gravitational acceleration
	barrier = barrier_strength * np.exp(-((Y - barrier_center_y)**2) / (2 * barrier_width**2))
	V = barrier + m * g * Y  # gravity: V_grav = m*g*x
	
	# Initial wavefunction: a 2D Gaussian wavepacket.
	# Normalization factor for 2D: 1/sqrt(2*pi*sigma^2)
	norm = 1.0 / (np.sqrt(2 * 2 * np.pi * sigma**2))  # Extra sqrt(2) in denominator

	# First particle
	psi_1 = norm * np.exp(-(((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2))) \
		* np.exp(1j * (p0x * (X - x0) + p0y * (Y - y0)) / hbar)

	# Second particle (moving in opposite x direction, spawned at mirror to x=0)
	psi_2 = norm * np.exp(-(((X + x0)**2 + (Y - y0)**2) / (4 * sigma**2))) \
		* np.exp(1j * (-p0x * (X + x0) + p0y * (Y - y0)) / hbar)


	psi = psi_1 + psi_2

	# Build momentum space grids.
	kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
	ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
	KX, KY = np.meshgrid(kx, ky, indexing='ij')
	
	# Precompute phase factors for the kinetic and potential steps.
	kinetic_phase = np.exp(-1j * dt * hbar * (KX**2 + KY**2) / (2 * m))
	potential_phase = np.exp(-1j * V * dt / (2 * hbar))
	
	# Set up time array and storage for probability density.
	t_arr = np.arange(n_steps) * dt
	prob_storage = []
	
	# Initialize storage for probability current components
	jx_storage = []
	jy_storage = []

	# Main time evolution loop using the split-operator method.
	for _ in range(n_steps):
		# Save the probability density |psi|^2 at the current time.
		prob_storage.append(np.abs(psi)**2)

		# Calculate probability current components
		# Using central difference for spatial derivatives
		dpsi_dx = np.gradient(psi, dx, axis=0)
		dpsi_dy = np.gradient(psi, dy, axis=1)
		
		# Calculate Jx and Jy components
		jx = (hbar/(2*m)) * np.imag(np.conj(psi) * dpsi_dx)
		jy = (hbar/(2*m)) * np.imag(np.conj(psi) * dpsi_dy)
		
		jx_storage.append(jx)
		jy_storage.append(jy)

		# First half-step: potential operator.
		psi = potential_phase * psi
		# Full kinetic step: FFT to momentum space, multiply, then IFFT back.
		psi = np.fft.ifft2( kinetic_phase * np.fft.fft2(psi) )
		# Second half-step: potential operator.
		psi = potential_phase * psi
	
	return {
		"x": x.tolist(),
		"y": y.tolist(),
		"t": t_arr.tolist(),
		"prob": [p.tolist() for p in prob_storage],
		"jx": [j.tolist() for j in jx_storage],
        "jy": [j.tolist() for j in jy_storage]
	}
