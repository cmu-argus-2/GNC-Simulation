#!/usr/bin/python3

import time
import numpy as np
from mock_vision_model import *

width = 720     # px
height = 720    # px
f = 600.0e-3    # m
R = 6.371e6     # m
N = 100
cubesat_pos_in_ecef = np.array([0.0, 0.0, -1.1*R])
cubesat_att_in_ecef = np.array([1.0, 0.0, 0.0, 0.0])
camera_pos_in_cubesat = np.zeros(3)
camera_att_in_cubesat = np.array([1.0, 0.0, 0.0, 0.0])

cam = Camera(
    image_width=width,
    image_height=height,
    focal_length=f,
    position_in_cubesat_frame=camera_pos_in_cubesat,
    orientation_in_cubesat_frame=camera_att_in_cubesat
)

vision = MockVisionModel(
    camera=cam,
    max_correspondences=N,
    earth_radius=R
)

st = time.time()
correspondences = vision.get_measurement(
    cubesat_position_in_ecef=cubesat_pos_in_ecef,
    cubesat_attitude_in_ecef=cubesat_att_in_ecef
)
et = time.time() - st

for correspondence in correspondences:
    print(correspondence.ecef_coordinate/1000)

print(f"correspondence generation runtime: {et}")
