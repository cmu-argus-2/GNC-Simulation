import numpy as np
from scipy.spatial.transform import Rotation as R


class StarTracker:

    def __init__(self, sigma_error_deg):
        self.sigma_error_rad = np.deg2rad(sigma_error_deg)

    def get_measurement(self, true_J2000_R_ST):
        """Simulate a Star tracker measurement as a random perturbation of turth attitude

        Args:
            J2000_R_ST (_type_): Rotation matrix. "ST" means Star Tracker camera frame
        """

        # generate a random axis, v, for the rotaiton error
        v = np.random.uniform(-1, 1, 3)
        assert np.all(v != 0)
        v /= np.linalg.norm(v)

        # sample the magnitude of rotation error
        error_rad = np.random.normal(0, self.sigma_error_rad)

        # generate the rotation error
        perturbation = R.from_rotvec(v * error_rad).as_matrix()

        return true_J2000_R_ST @ perturbation
