import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_simulation_heatmaps(simulation_data, num_frames=9):
	"""
	Plot multiple frames of the 2D quantum simulation as heatmaps.
	
	Parameters:
		simulation_data: dict
			Output from simulate_2d function containing 'x', 'y', 't', and 'prob'
		num_frames: int
			Number of frames to display (will be arranged in a square grid)
	"""
	# Convert lists back to numpy arrays
	x = np.array(simulation_data['x'])
	y = np.array(simulation_data['y'])
	t = np.array(simulation_data['t'])
	prob = np.array(simulation_data['prob'])
	
	# Calculate grid size for subplots (ceil of sqrt)
	grid_size = int(np.ceil(np.sqrt(num_frames)))
	
	# Create figure
	fig, axes = plt.subplots(grid_size, grid_size, 
							figsize=(15, 15),
							squeeze=False)
	
	# Select frames evenly spaced through time
	frame_indices = np.linspace(0, len(prob)-1, num_frames, dtype=int)
	
	# Plot each frame
	for idx, frame_idx in enumerate(frame_indices):
		row = idx // grid_size
		col = idx % grid_size
		ax = axes[row, col]
		
		# Create heatmap
		im = ax.pcolormesh(x, y, prob[frame_idx].T,
						  shading='auto',
						  cmap='viridis')
		
		# Add colorbar and title
		plt.colorbar(im, ax=ax)
		ax.set_title(f't = {t[frame_idx]:.2f}')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		
	# Remove empty subplots if any
	for idx in range(num_frames, grid_size*grid_size):
		row = idx // grid_size
		col = idx % grid_size
		fig.delaxes(axes[row, col])
	
	plt.tight_layout()
	return fig


def plot_potential(x, y, barrier_center_y=0.0, barrier_strength=5.0, barrier_width=0.5, m=1.0):
	"""
	Visualize the potential (barrier + gravity) in the system.
	"""
	X, Y = np.meshgrid(x, y, indexing='ij')
	g = 9.81
	barrier = barrier_strength * np.exp(-((X - barrier_center_y)**2) / (2 * barrier_width**2))
	V = barrier + m * g * X
	
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

