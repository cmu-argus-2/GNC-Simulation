import numpy as np
from scipy.spatial.transform import Rotation as R


class SunSensor:
    def __init__(self, sigma_angular_error):
        self.sigma_angular_error = sigma_angular_error

    def get_measurement(self, clean_signal):
        angular_error = self.sigma_angular_error * np.random.standard_normal()

        axis_of_angular_error = R.random().as_rotvec()
        axis_of_angular_error /= np.linalg.norm(axis_of_angular_error)

        perturbation = R.from_rotvec(axis_of_angular_error * angular_error)

        measurement = perturbation.as_matrix() @ clean_signal
        return measurement
