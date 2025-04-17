from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

# venv\Scripts\activate
# source venv/bin/activate
# pip install flask numpy
# python3 calculationServer.py
# deactivate
# pip freeze > requirements.txt

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes with no restrictions

def simulate_2d(x_min=-20, x_max=20, Nx=128,
				y_min=-20, y_max=20, Ny=128,
				n_steps=100, dt=0.01,
				hbar=1.0, m=1.0,
				barrier_center_x=0.0, barrier_strength=5.0, barrier_width=0.5,
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
	X, Y = np.meshgrid(x, y, indexing='ij')  # X, Y shape: (Nx, Ny)
	
	# Define the potential V(x,y):
	# A Gaussian barrier along x plus gravitational potential in y.
	g = 9.81  # gravitational acceleration
	barrier = barrier_strength * np.exp(-((X - barrier_center_x)**2) / (2 * barrier_width**2))
	V = barrier + m * g * Y  # gravity: V_grav = m*g*y
	
	# Initial wavefunction: a 2D Gaussian wavepacket.
	# Normalization factor for 2D: 1/sqrt(2*pi*sigma^2)
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
	
	# Main time evolution loop using the split-operator method.
	for _ in range(n_steps):
		# Save the probability density |psi|^2 at the current time.
		prob_storage.append(np.abs(psi)**2)
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
		"prob": [p.tolist() for p in prob_storage]
	}

@app.route('/simulate', methods=['GET'])
def simulate_api():
	"""
	GET API endpoint for the 2D simulation.
	
	Example query string:
	  http://127.0.0.1:5000/simulate?x_min=-20&x_max=20&Nx=128&y_min=-20&y_max=20&Ny=128&n_steps=100&dt=0.01&hbar=1.0&m=1.0
	  &barrier_center_x=0&barrier_strength=5.0&barrier_width=0.5&x0=-5&y0=-5&p0x=2.0&p0y=2.0&sigma=1.0
	"""
	try:
		x_min = float(request.args.get('x_min', -20))
		x_max = float(request.args.get('x_max', 20))
		Nx = int(request.args.get('Nx', 128))
		y_min = float(request.args.get('y_min', -20))
		y_max = float(request.args.get('y_max', 20))
		Ny = int(request.args.get('Ny', 128))
		n_steps = int(request.args.get('n_steps', 100))
		dt = float(request.args.get('dt', 0.01))
		hbar = float(request.args.get('hbar', 1.0))
		m = float(request.args.get('m', 1.0))
		barrier_center_x = float(request.args.get('barrier_center_x', 0.0))
		barrier_strength = float(request.args.get('barrier_strength', 5.0))
		barrier_width = float(request.args.get('barrier_width', 0.5))
		x0 = float(request.args.get('x0', -5.0))
		y0 = float(request.args.get('y0', -5.0))
		p0x = float(request.args.get('p0x', 2.0))
		p0y = float(request.args.get('p0y', 2.0))
		sigma = float(request.args.get('sigma', 1.0))
	except Exception as e:
		return jsonify({"error": str(e)})
	
	result = simulate_2d(
		x_min, x_max, Nx,
		y_min, y_max, Ny,
		n_steps, dt, hbar, m,
		barrier_center_x, barrier_strength, barrier_width,
		x0, y0, p0x, p0y, sigma
	)
	
	# print (result)
	
	return jsonify(result)

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=True)
