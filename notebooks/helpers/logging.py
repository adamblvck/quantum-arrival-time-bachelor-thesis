import datetime

def log_simulation_parameters(output_file, params, extra_name=""):
    """
    Log simulation parameters to a log file.
    
    Parameters:
        output_file: str
            Path to the video file that was created
        params: dict
            Dictionary containing all simulation parameters
    """
    log_file = f'simulations/simulation_log_{extra_name}.txt'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create header of log entry
    log_entry = f"""
=== Simulation Run: {timestamp} ===
Video file: {output_file}
Parameters:"""

    # Iterate through all parameters and add them to the log
    for key, value in params.items():
        log_entry += f"\n    {key}: {value}"

    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(log_entry)