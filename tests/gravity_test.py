import sys
sys.path.append('../')

from build.tests.pymodels import grav_acc, grav_sph_acc, grav_J2_acc

import pytest
import numpy as np

# SPHERICAL GRAVITY TESTS
# Generate position vectors
test_cases = []
theta_space = np.linspace(0, 2*np.pi, 40)
phi_space = np.linspace(0, np.pi, 40)
for R in range(6371000, 7500000, 5000):
    for theta in theta_space:
        for phi in phi_space:
            test_cases.append((R*np.sin(phi)*np.cos(theta), R*np.sin(phi)*np.sin(theta), R*np.cos(phi)))

@pytest.mark.parametrize("v", test_cases)
def sph_test(v):
    v = np.array(v)
    v_hat = v/np.linalg.norm(v)

    sph_g = grav_sph_acc(v)

    sph_g_hat = np.array(sph_g)/np.linalg.norm(sph_g)

    mu = 398600.435507 * 1e9 # [https://ssd.jpl.nasa.gov/astro_par.html]
    expected_norm = mu/(np.linalg.norm(v))**2

    assert np.linalg.norm(v_hat + sph_g_hat) <= 1e-7 # unit vector of acc_g along negative position vector
    assert abs(np.linalg.norm(sph_g) - expected_norm) <= 1e-7 # acceleration magnitude = GM/r^2