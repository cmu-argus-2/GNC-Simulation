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

# UT1: define random quaternions in the config file and see how the sim responds
def test_quatinit():
    """
    Test the initialization and simulation of quaternions.
    This function performs the following steps:
    1. Loads a configuration file and modifies specific entries.
    2. Saves the modified configuration to a new file.
    3. Creates a log directory for simulation results.
    4. Sets up environment variables and runs the simulation script.
    5. Loads and parses the simulation results from a binary file.
    6. Performs assertions to check the validity of the loaded data.
    7. Deletes the log directory after the test is complete.
    Assertions:
    - Ensures that the parsed data is not None.
    - Checks that the norm of each quaternion is close to 1 within a tolerance of 1e-6.
    TODO: this is more of an integrated test than a unit test
    """

    config_path = os.path.join(os.path.dirname(__file__), '..', 'montecarlo', 'configs', 'params.yaml')
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Modify some entries in the config_data
    config_data['initial_attitude'] = [1,0,0,0]
    config_data['initial_angular_rate'] = [1,1,1]
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
    quaternions = np.array([data_dicts["q_w"], data_dicts["q_x"], data_dicts["q_y"], data_dicts["q_z"]]).T

    for quat in quaternions:
        assert np.isclose(np.linalg.norm(quat), 1, atol=1e-6)
    

"""
def test_quatnorm():
    q = np.array([1, 2, 3, 4])
    result = quatnorm(q)
    expected_result = np.array([0.18257419, 0.36514837, 0.54772256, 0.73029674])
    assert np.allclose(result, expected_result, atol=1e-6)
"""

def test_quatrotation():
    q = np.array([0.7071068, 0.7071068, 0, 0])
    R = quatrotation(q)
    expected_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(R, expected_R, atol=1e-6)


def test_quatconj():
    np.random.seed(0)  # For reproducibility
    random_quaternions = [np.random.rand(4) for _ in range(100)]
    normalized_quaternions = [quatnormalize(q) for q in random_quaternions]
    results = []
    for q in normalized_quaternions:
        q_conj = quatconj(q)
        result = hamiltonproduct(q_conj, q)
        results.append(result)

    expected_result = [np.array([1,0,0,0]) for _ in range(100)]
    assert np.allclose(result, expected_result, atol=1e-6)


def test_crossproduct():
    """
    Test the crossproduct function to ensure it correctly computes the cross product of two vectors.
    This test generates 100 pairs of random 3-dimensional vectors and computes their cross products
    using both the crossproduct function and numpy's built-in cross product function. The results are
    then compared to ensure they are close within a tolerance of 1e-6.
    The test ensures reproducibility by setting a random seed.
    Assertions:
        - The computed cross products from the crossproduct function should be close to the results
          from numpy's cross product function within a tolerance of 1e-6.
    """
    np.random.seed(0)  # For reproducibility
    random_vectors1 = [np.random.rand(3) for _ in range(100)]
    random_vectors2 = [np.random.rand(3) for _ in range(100)]
    result = np.zeros((3, 100))
    expected_Result = np.zeros((3, 100))
    for i, vec in enumerate(random_vectors1):
        skew_matrix = crossproduct(vec)

        result[:,i] = skew_matrix @ random_vectors2[i]
        expected_Result[:,i] = np.cross(vec, random_vectors2[i])
    
    assert np.allclose(result, expected_Result, atol=1e-6)


def test_hamiltonproduct():
    """
    Test the Hamilton product function to ensure it correctly rotates vectors using quaternions.
    This test generates 100 random quaternions and vectors, normalizes them, and then uses the 
    Hamilton product to rotate the vectors by the quaternions. The results are compared to the 
    expected results obtained by applying the corresponding rotation matrix derived from the 
    quaternions.
    The test asserts that the results from the Hamilton product are close to the expected results 
    within a tolerance of 1e-6.
    Steps:
    1. Generate 100 random quaternions and normalize them.
    2. Generate 100 random vectors and normalize them.
    3. For each quaternion and vector pair:
       a. Convert the vector to a quaternion with a zero scalar part.
       b. Rotate the vector quaternion using the Hamilton product with the quaternion and its conjugate.
       c. Extract the vector part of the resulting quaternion.
       d. Compute the expected rotated vector using the rotation matrix derived from the quaternion.
    4. Compare the results from the Hamilton product to the expected results.
    5. Assert that the results are close to the expected results within a tolerance of 1e-6.
    TODO: this compares two implemented quaternion-based rotation functions, but doesnt compare them to a "ground truth"
    """

    np.random.seed(0)  # For reproducibility
    random_quaternions = [np.random.rand(4) for _ in range(100)]
    normalized_quaternions = [quatnormalize(q) for q in random_quaternions]
    random_vectors = [np.random.rand(3) for _ in range(100)]
    normalized_vectors = [v / np.linalg.norm(v) for v in random_vectors]

    results = []
    expected_results = []
    for q, v in zip(normalized_quaternions, normalized_vectors):
        vquat = np.array([0, v[0], v[1], v[2]])
        result = hamiltonproduct(q, hamiltonproduct(vquat, quatconj(q)))
        results.append(result[1:])
        # body frame to inertial
        R = quatrotation(q)
        expected_results.append(R @ v)
    results = np.array(results)
    expected_results = np.array(expected_results)
    assert np.allclose(results, expected_results, atol=1e-6)


def test_rotmat2quat():
    np.random.seed(0)  # For reproducibility
    random_quaternions = [np.random.rand(4) for _ in range(100)]
    normalized_quaternions = [quatnormalize(q) for q in random_quaternions]

    for q in normalized_quaternions:
        R = quatrotation(q)
        q_back = rotmat2quat(R)
        if np.dot(q, q_back) < 0:  # Handle quaternion double-cover (q and -q represent the same rotation)
            q_back = -q_back
        assert np.allclose(q, q_back, atol=1e-6)

""" implemented because debugging call wasnt working, but it does now. keeping in case it craps out again
def run_debug():
    import test_math
    for i in dir(test_math):
        item = getattr(test_math, i)
        if callable(item) and item.__module__ == 'test_math' and i.startswith('test_'):
            item()
"""

# Run the tests
if __name__ == "__main__":
    # run_debug()
    # pytest.main(["-k", "not run_debug"])
    pytest.main([os.path.abspath(__file__)])