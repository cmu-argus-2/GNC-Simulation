#!/usr/bin/python3

import time
import yaml
import numpy as np
from mock_vision_model import *
from brahe.constants import R_EARTH as R


with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
camera_params = config["satellite"]["camera"]

N = 100
cubesat_pos_in_ecef = np.array([0.0, 0.0, -1.1*R])
cubesat_att_in_ecef = np.array([1.0, 0.0, 0.0, 0.0])
camera_pos_in_cubesat = np.zeros(3)
camera_att_in_cubesat = np.array([1.0, 0.0, 0.0, 0.0])
pixel_noise_mean = np.zeros(2)
pixel_noise_std_dev = np.array([10.0, 10.0 * camera_params["image_height"] / camera_params["image_width"]])

cam = Camera(
    image_width=camera_params["image_width"],
    image_height=camera_params["image_height"],
    focal_length=camera_params["focal_length"],
    position_in_cubesat_frame=camera_params["position_in_cubesat_frame"],
    orientation_in_cubesat_frame=camera_params["orientation_in_cubesat_frame"]
)

vision = MockVisionModel(
    camera=cam,
    max_correspondences=N,
    earth_radius=R,
    pixel_noise_mean=pixel_noise_mean,
    pixel_noise_std_dev=pixel_noise_std_dev,
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
