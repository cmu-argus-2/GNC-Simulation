import numpy as np
from scipy.spatial.transform import Rotation as R


class SunSensor:
    def __init__(self, dt, sigma_angular_error, ss_params):
        self.sigma_angular_error = sigma_angular_error
        self.dt                  = dt
        self.last_meas_time      = -np.inf
        self.num_photodiodes     = ss_params["num_photodiodes"]
        self.G_pd_b = np.array(ss_params["photodiode_normals"]).reshape(-1, 3)
        self.MAX_RANGE = 117000  # OPT4001
        self.THRESHOLD_ILLUMINATION_LUX = 3000
        self.last_measurement = np.zeros(3)

    def get_measurement(self, clean_signal):
        angular_error = self.sigma_angular_error * np.random.standard_normal()

        axis_of_angular_error = R.random().as_rotvec()
        axis_of_angular_error /= np.linalg.norm(axis_of_angular_error)

        perturbation = R.from_rotvec(axis_of_angular_error * angular_error)

        measurement = perturbation.as_matrix() @ clean_signal
        return measurement
