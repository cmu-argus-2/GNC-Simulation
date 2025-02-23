import numpy as np
from scipy.spatial.transform import Rotation as R


class SunSensor:
    def __init__(self, dt, sigma_angular_error, G_pd_b):
        self.sigma_angular_error = sigma_angular_error
        self.dt                  = dt
        self.last_meas_time      = -np.inf
        self.G_pd_b              = G_pd_b
        self.MAX_RANGE = 117000  # OPT4001
        self.THRESHOLD_ILLUMINATION_LUX = 3000

    def get_measurement(self, clean_signal):
        angular_error = self.sigma_angular_error * np.random.standard_normal()

        axis_of_angular_error = R.random().as_rotvec()
        axis_of_angular_error /= np.linalg.norm(axis_of_angular_error)

        perturbation = R.from_rotvec(axis_of_angular_error * angular_error)

        measurement = perturbation.as_matrix() @ clean_signal
        return measurement
