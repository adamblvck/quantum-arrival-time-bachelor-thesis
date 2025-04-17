import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import datetime

def plot_potential(x, y, barrier_center_y=0.0, barrier_strength=5.0, barrier_width=0.5, m=1.0, **kwargs):
	"""
	Visualize the potential (barrier + gravity) in the system.
	"""
	X, Y = np.meshgrid(x, y, indexing='xy')
	g = 9.81  # gravitational acceleration
	barrier = barrier_strength * np.exp(-((Y - barrier_center_y)**2) / (2 * barrier_width**2))
	V = barrier + m * g * Y  # gravity acts along the y-direction.
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
	
	# Plot 2D potential heatmap
	im1 = ax1.pcolormesh(x, y, V, shading='auto', cmap='viridis')
	ax1.set_title('Full Potential (Barrier + Gravity)')
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	plt.colorbar(im1, ax=ax1)
	
	# Plot barrier cross-section at y=0
	y_middle_idx = len(y) // 2
	ax2.plot(x, V[:, y_middle_idx])
	ax2.set_title('Potential Cross-section at x=0')
	ax2.set_xlabel('y')
	ax2.set_ylabel('V(0,y)')
	ax2.grid(True)
	
	plt.tight_layout()
	return fig

def create_animation(simulation_data, output_file='quantum_evolution.mp4', fps=30):
	"""
	Create an MP4 animation of the quantum evolution.
	
	Parameters:
		simulation_data: dict
			Output from simulate_2d function
		output_file: str
			Name of the output MP4 file
		fps: int
			Frames per second for the animation
	"""
	x = np.array(simulation_data['x'])
	y = np.array(simulation_data['y'])
	prob = np.array(simulation_data['prob'])
	t = np.array(simulation_data['t'])
	
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Initial plot
	im = ax.pcolormesh(x, y, prob[0].T, shading='auto', cmap='viridis')
	plt.colorbar(im, ax=ax)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	title = ax.set_title(f't = {t[0]:.2f}')
	
	# Animation update function
	def update(frame):
		im.set_array(prob[frame].T.ravel())
		title.set_text(f't = {t[frame]:.2f}')
		return im, title
	
	# Create animation
	anim = FuncAnimation(
		fig, update,
		frames=len(prob),
		interval=1000/fps,  # interval in milliseconds
		blit=True
	)
	
	# Slower -  Save animation
	writer = animation.FFMpegWriter(fps=fps, bitrate=2000)

	# Save animation with optimizations
	# writer = animation.FFMpegWriter(
	# 	fps=fps, 
	# 	bitrate=1000,  # Reduced bitrate
	# 	codec='libx264',  # Use a faster codec
	# 	extra_args=['-preset', 'ultrafast']  # Use ultrafast preset
	# )
	anim.save(output_file, writer=writer)
	plt.close()
	
	return output_file

# Usage:
if __name__ == "__main__":
	# Run simulation

	params = {
		'x_min': -20, 'x_max': 20, 'Nx': 128*2**1,
		'y_min': -20, 'y_max': 20, 'Ny': 128*2**1,
		'n_steps': 2000, 'dt': 0.01,
		'barrier_center_y': 0, 'barrier_strength': 2,
		'barrier_width': 1,
		'x0': -15, 'y0': 5, 'p0x': 1, 'p0y': 0.0
	}

	simulation_data = simulate_2d(**params) # defined somewhere else
	
	# Plot the potential
	fig_potential = plot_potential(
		np.array(simulation_data['x']),
		np.array(simulation_data['y'])
	)
	plt.show()
	
	# Create animation
	animation_file = create_animation(simulation_data, output_file=f'simulations/{datetime.datetime.now().strftime("%Y%m%d_%H-%M")}_quantum_evolution.mp4', fps=1/params['dt'])
	log_simulation_parameters(animation_file, params)
	print(f"Animation saved as: {animation_file}")