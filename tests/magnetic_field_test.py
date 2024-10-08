import sys
sys.path.append('../')

from build.tests.pymodels import magnetic_field_sez, magnetic_field

import pytest
import numpy as np

# SEZ Magnetic field tests
# Read ground truth
truth_data = np.genfromtxt("magField_groundtruth.csv", delimiter=',')

# Test cases
test_cases = []
lat_space = np.linspace(np.pi/2, -np.pi/2, 37)
lon_space = np.linspace(-np.pi, np.pi, 361)
date_space = np.linspace(2020, 2024.5, 10)
i = 0
for date in date_space:
    for lat in lat_space:
        for lon in lon_space:
            test_cases.append(([lon, lat, 600000.0], date, list(truth_data[i,4:7])))
            i = i+1

@pytest.mark.parametrize("r, date, truth", test_cases)
def test(r, date, truth):
    field = np.array(magnetic_field_sez(r, date))
    truth = np.array([-truth[0], truth[1], -truth[2]])

    assert np.linalg.norm(field-truth)*1e-9 <= 1e-8 # field error against ground truth <= 10nT