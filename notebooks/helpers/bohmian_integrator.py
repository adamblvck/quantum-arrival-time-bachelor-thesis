import numpy as np
from scipy.interpolate import RegularGridInterpolator

def compute_bohmian_trajectories(simulation_data, params, spin='up', 
                                 n_trajectories=10, random_seed=0):
    """
    Compute Bohmian trajectories from simulation data for the given spin
    using 2 Euler sub-steps per simulation frame.

    Parameters
    ----------
    simulation_data : dict
        Dictionary containing 'x', 'y', 't', and lists for probability densities 
        and current components (e.g. 'prob_up', 'jx_up', 'jy_up').
    params : dict
        Parameter dictionary containing initial positions (x0, y0) and sigma.
    spin : str
        Specify which spin component to use ('up' or 'down').
    n_trajectories : int
        The number of trajectories (or sample points) to simulate.
    random_seed : int
        For reproducibility.
        
    Returns
    -------
    trajectories : ndarray, shape (n_trajectories, N_time, 2)
        Array containing the trajectories for each initial point.
    """
    
    # Convert grid data to arrays.
    t = np.array(simulation_data['t'])
    xgrid = np.array(simulation_data['x'])
    ygrid = np.array(simulation_data['y'])
    dt = t[1] - t[0]
    N_t = len(t)
    
    # Choose the proper data for the selected spin.
    if spin.lower() == 'up':
        jx_all = np.array(simulation_data['jx_up'])
        jy_all = np.array(simulation_data['jy_up'])
        prob_all = np.array(simulation_data['prob_up'])
    elif spin.lower() == 'down':
        jx_all = np.array(simulation_data['jx_down'])
        jy_all = np.array(simulation_data['jy_down'])
        prob_all = np.array(simulation_data['prob_down'])
    else:
        raise ValueError("spin must be 'up' or 'down'")
    
    # Spawn initial positions. Here we use a Gaussian distribution centered at (x0, y0)
    # with standard deviation sigma.
    np.random.seed(random_seed)
    x0, y0 = params['x0'], params['y0']
    sigma = params['sigma']
    initial_positions = np.column_stack([
        np.random.normal(x0, sigma/2, n_trajectories),
        np.random.normal(y0, sigma/2, n_trajectories)
    ])
    
    # Prepare an array to store each trajectory:
    trajectories = np.zeros((n_trajectories, N_t, 2))
    trajectories[:, 0, :] = initial_positions  # initial positions at t=0

    # Number of Euler sub-steps per frame.
    n_sub = 2
    sub_dt = dt / n_sub
    
    # Loop over time steps and update the positions using two Euler sub-steps.
    for i in range(N_t - 1):
        # Compute the local velocity field at time i:
        # Add a tiny epsilon to avoid division-by-zero.
        epsilon = 1e-12  
        field_vx = jx_all[i] / (prob_all[i] + epsilon)
        field_vy = jy_all[i] / (prob_all[i] + epsilon)
        
        # Build 2D interpolators.
        # NOTE: The arrays are on the grid (y, x) because meshgrid was built with indexing='xy'
        # If using RegularGridInterpolator((xgrid, ygrid), field) then the function expects
        # query points as (x, y). Adjust the query ordering accordingly!
        interp_vx = RegularGridInterpolator((ygrid, xgrid), field_vx)
        interp_vy = RegularGridInterpolator((ygrid, xgrid), field_vy)
        
        # For each trajectory, update the position using two Euler sub-steps.
        for m in range(n_trajectories):
            pos = trajectories[m, i, :].copy()  # current position (x, y)
            for _ in range(n_sub):
                # Here, the interpolator is constructed with (xgrid, ygrid) and
                # if your velocity fields effectively are arranged in (y,x) order,
                # you may need to query with (pos[1], pos[0]). Adjust if necessary.
                v_x = interp_vx((pos[1], pos[0]))
                v_y = interp_vy((pos[1], pos[0]))
                
                # Euler update: use sub_dt (dt/2) for each sub-step.
                pos[0] += sub_dt * v_x
                pos[1] += sub_dt * v_y
            
            # Store the updated position at the next time index.
            trajectories[m, i+1, :] = pos
            
    return trajectories