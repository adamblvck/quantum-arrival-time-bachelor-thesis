import pickle
import datetime
import os

""" USAGE

# Saving the data
pickle_file = save_simulation_data(simulation_data, params)
print(f"Data saved to: {pickle_file}")

# Loading the data
sim_data, parameters, timestamp = load_simulation_data(pickle_file)

"""


def save_simulation_data(simulation_data, params, base_dir='simulations'):
    """
    Saves simulation data and parameters to a pickle file.
    
    Args:
        simulation_data (dict): Dictionary containing simulation results
        params (dict): Dictionary containing simulation parameters
        base_dir (str): Base directory for saving files
    
    Returns:
        str: Path to the saved pickle file
    """
    # Create simulations directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M")
    
    # Combine data and parameters into one dictionary
    data_to_save = {
        'simulation_data': simulation_data,
        'parameters': params,
        'timestamp': timestamp
    }
    
    # Create filename
    filename = os.path.join(base_dir, f'simulation_{timestamp}.pkl')
    
    # Save to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    return filename

def load_simulation_data(filename):
    """
    Loads simulation data and parameters from a pickle file.
    
    Args:
        filename (str): Path to the pickle file
    
    Returns:
        tuple: (simulation_data, params, timestamp)
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['simulation_data'], data['parameters'], data['timestamp']