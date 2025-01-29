import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from world.math.quaternions import *
import yaml
import subprocess
import shutil
from visualization.plotter.parse_bin_file import parse_bin_file
import pyIGRF
import numpy as np
from numpy.testing import assert_allclose
import pytest
from build.world.pyphysics import ECI2GEOD


# simulation is energy consistent

# validate that translational orbital dynamics make sense

# validate frame conversions

# cross-check magnetic field with another reference
def test_magfieldECI():

    config_path = os.path.join(os.path.dirname(__file__), '..', 'montecarlo', 'configs', 'params.yaml')
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Modify some entries in the config_data
    # config_data['initial_attitude'] = [1,0,0,0]
    # config_data['initial_angular_rate'] = [1,1,1]
    config_data["debugFlags"]["bypass_controller"] = True
    config_data["dt"] = 1 # [s] time step
    config_data["MAX_TIME"] = 100
    
    # Save the modified config to a new file
    new_config_path = os.path.join(os.path.dirname(__file__), "UT1config.yaml")
    with open(new_config_path, 'w') as file:
        yaml.safe_dump(config_data, file)

    log_directory = os.path.join(os.path.dirname(__file__), "UT1configres")
    os.makedirs(log_directory, exist_ok=True)
    
    trial_command_dir = os.path.join(os.path.dirname(__file__), '..')
    env = os.environ.copy()
    env["TRIAL_DIRECTORY"] = log_directory
    env["PARAMETER_FILEPATH"] = new_config_path
    env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "300"  # Set timeout to 300 seconds
    subprocess.run(['python3', os.path.join(trial_command_dir, "sim.py")], env=env)

    # Load data from UT1configres/state_true.bin
    state_true_path = os.path.join(log_directory, "state_true.bin")
    data_dicts = parse_bin_file(state_true_path, 100)

    # Perform some checks or assertions on the loaded data
    assert data_dicts is not None
    # Delete the folder UT1configres
    shutil.rmtree(log_directory)
    # quaternions = np.array([data_dicts["q_w"], data_dicts["q_x"], data_dicts["q_y"], data_dicts["q_z"]]).T
    MagField = np.array([data_dicts["xMag ECI [T]"], data_dicts["yMag ECI [T]"], data_dicts["zMag ECI [T]"]]).T

    Pos_ECI = np.array([data_dicts["r_x ECI [m]"], data_dicts["r_y ECI [m]"], data_dicts["r_z ECI [m]"]]).T
    time_vec = np.array(data_dicts["Time [s]"])
    lon = np.zeros_like(time_vec)
    lat = np.zeros_like(time_vec)
    alt = np.zeros_like(time_vec)
    expectedBNED = np.zeros((len(time_vec), 3))
    for k in range(len(time_vec)):
        lon[k], lat[k], alt[k] = ECI2GEOD(Pos_ECI[k,:], time_vec[k])
        expectedBNED[k,:] =  pyIGRF.igrf_variation(lat, lon, alt, time_vec)
                
    #expected Bxyz
    

    # validate sun direction too
    rSun = np.array([data_dicts["rSun_x ECI [m]"], data_dicts["rSun_y ECI [m]"], data_dicts["rSun_z ECI [m]"]]).T
    

# Run the tests
if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])