import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
from .SunSensorAlgs import compute_body_ang_vel_from_sun_rays


def skew_symmetric(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


# TODO convert to sqrt form if fixed precision math ends up being an issue


class Attitude_EKF:
    def __init__(
        self,
        initial_ECI_R_b_estimate,  # [w, x, y, z]
        initial_gyro_bias_estimate,
        sigma_initial_attitude,  # [rad]
        sigma_gyro_white,  # [rad/sqrt(s)]
        sigma_gyro_bias_deriv,  # [(rad/s)/sqrt(s))]
        sigma_sunsensor,  # [rad]
        NOMINAL_GYRO_DT,
        time_of_initial_attitude,  # [s]
    ):
        self.state_vector = np.zeros(7)
        self.set_ECI_R_b(initial_ECI_R_b_estimate)
        self.set_gyro_bias(initial_gyro_bias_estimate)
        self.time_of_initial_attitude = time_of_initial_attitude

        self.P = None
        self.sigma_initial_attitude = sigma_initial_attitude
        self.sigma_gyro_white = sigma_gyro_white
        self.sigma_gyro_bias_deriv = sigma_gyro_bias_deriv

        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= sigma_gyro_white**2
        self.Q[3:6, 3:6] *= sigma_gyro_bias_deriv**2
        self.NOMINAL_GYRO_DT = NOMINAL_GYRO_DT

        self.last_gyro_measurement_time = None

        self.sigma_sunsensor = sigma_sunsensor

        # State variable pertaining to initialization
        self.initialized = False
        self.init_gyro_measurements = []
        self.init_sun_rays = []
        self.min_degress_changed_to_init_from_sun_rays = 30  # [deg]  # TODO put me in param file

    def set_ECI_R_b(self, ECI_R_b):
        q = ECI_R_b.as_quat()  # [x, y, z, w]
        self.state_vector[0:4] = [q[3], *q[0:3]]  # [w, x, y, z]

    def set_gyro_bias(self, gyro_bias):
        self.state_vector[4:7] = gyro_bias

    def get_ECI_R_b(self):
        q = self.state_vector[0:4]  # [w, x, y, z]
        return R.from_quat([*q[1:4], q[0]])  # from_quat takes in [x, y, z, w]

    def get_quat_ECI_R_b(self):
        return self.state_vector[0:4]  # [w, x, y, z]

    def get_gyro_bias(self):
        return self.state_vector[4:7]

    def get_F(self):
        F = np.zeros((6, 6))
        F[0:3, 3:6] = -self.get_ECI_R_b().as_matrix()
        return F

    def get_G(self):
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -self.get_ECI_R_b().as_matrix()
        G[3:6, 3:6] = np.eye(3)
        return G

    def get_uncertainty_sigma(self):
        return np.sqrt(np.diag(self.P))

    def gyro_update(self, gyro_measurement, t):
        if not self.initialized:
            self.init_gyro_measurements.append((t, gyro_measurement))
            return
        if self.last_gyro_measurement_time is None:  # first time
            dt = self.NOMINAL_GYRO_DT
        else:
            dt = t - self.last_gyro_measurement_time

        unbiased_gyro_measurement = gyro_measurement - self.get_gyro_bias()
        rotvec = unbiased_gyro_measurement * dt
        delta_rotation = R.from_rotvec(rotvec)
        ECI_R_b_prev = self.get_ECI_R_b()
        ECI_R_b_curr = ECI_R_b_prev * delta_rotation
        self.set_ECI_R_b(ECI_R_b_curr)

        # TODO propogate error state covariance
        F = self.get_F()
        G = self.get_G()

        """
            Fill in the A matrix according to the Van Loan Algorithm (for computing
            integrals involving matrix exponential):
            https://www.cs.cornell.edu/cv/ResearchPDF/computing.integrals.involving.Matrix.Exp.pdf
            Inspired by Farrell, pg.143 (equations 4.113, 4.114)
        """
        A = np.zeros((12, 12))
        A[0:6, 0:6] = -F
        A[0:6, 6:12] = G @ self.Q @ G.transpose()
        A[6:12, 6:12] = F.transpose()
        A = A * dt

        # Compute Matrix exponential and extract values for Phi and Qdk*/
        # TODO does this matrix exponential work with circuit python?
        # if not - consider taylor series, horner's method?
        AExp = expm(A)
        Phi = AExp[6:12, 6:12].T
        Qdk = Phi @ AExp[0:6, 6:12]
        np.set_printoptions(suppress=True, linewidth=999)
        # Propogate covariance
        self.P = Phi @ self.P @ Phi.T + Qdk
        self.last_gyro_measurement_time = t

    def get_sun_sensor_measurement_jacobian(self, true_sun_ray_ECI):
        H = np.zeros((3, 6))
        H[:3, :3] = -skew_symmetric(true_sun_ray_ECI)
        return H

    def sun_sensor_update(self, measured_sun_ray_in_body, true_sun_ray_ECI, t):
        true_sun_ray_ECI /= np.linalg.norm(true_sun_ray_ECI)
        if not self.initialized:
            self.init_sun_rays.append((t, measured_sun_ray_in_body))
            self.attempt_to_initialize(t)
            return
        predicted_sun_ray_ECI = self.get_ECI_R_b().as_matrix() @ measured_sun_ray_in_body
        innovation = true_sun_ray_ECI - predicted_sun_ray_ECI

        s_cross = skew_symmetric(true_sun_ray_ECI)
        self.R_sunsensor = self.sigma_sunsensor**2 * s_cross @ np.eye(3) @ s_cross.T

        H = self.get_sun_sensor_measurement_jacobian(true_sun_ray_ECI)
        S = H @ self.P @ H.T + self.R_sunsensor
        K = self.P @ H.T @ np.linalg.pinv(S, 1e-4)  # TODO tuneme
        dx = K @ innovation

        attitude_correction = R.from_rotvec(dx[:3])
        self.set_ECI_R_b(attitude_correction * self.get_ECI_R_b())

        gyro_bias_correction = dx[3:]
        self.set_gyro_bias(self.get_gyro_bias() + gyro_bias_correction)

        # Symmetric Joseph update
        Identity = np.eye(6)
        self.P = (Identity - K @ H) @ self.P @ (Identity - K @ H).T + K @ self.R_sunsensor @ K.T

    def attempt_to_initialize(self, t):
        if len(self.init_sun_rays) < 10:
            return

        average_gyro_measurement = np.average([gyro for (t, gyro) in self.init_gyro_measurements], axis=0)
        # print("average_gyro_measurement [deg/s]: ", np.rad2deg(average_gyro_measurement))

        result = compute_body_ang_vel_from_sun_rays(self.init_sun_rays)
        if result is not None:
            (estimated_omega_in_body_frame, estimated_omega_in_body_frame_cov) = result
            print("estimated_omega_in_body_frame [deg/s]: ", np.rad2deg(estimated_omega_in_body_frame))
            print(
                "estimated_omega_in_body_frame_cov [deg/s]:\n",
                np.rad2deg(np.sqrt(np.diag(estimated_omega_in_body_frame_cov))),
            )

            # Initialize the attitude
            time_to_initialize = t - self.time_of_initial_attitude
            # rotation since getting initial attitude:
            b0_R_b1 = R.from_rotvec(estimated_omega_in_body_frame * time_to_initialize)
            inital_ECI_R_b = self.get_ECI_R_b()
            current_ECI_R_b = inital_ECI_R_b * b0_R_b1
            self.set_ECI_R_b(current_ECI_R_b)

            # Initialize the Gyro Bias
            gyro_bias_estimate = average_gyro_measurement - estimated_omega_in_body_frame
            self.set_gyro_bias(gyro_bias_estimate)

            # ======================== Initialize the Covariance ========================
            self.P = np.zeros((6, 6))

            # TODO Initialize the Attitude Covariance
            self.P[0:3, 0:3] = np.eye(3) * self.sigma_initial_attitude**2

            # TODO Initialize the Gyro Bias Covariance
            uncertainty_increase_in_true_gyro_bias_while_waiting_to_initialize = (
                np.eye(3) * time_to_initialize * self.sigma_gyro_bias_deriv**2
            )
            averaging_time = time_to_initialize
            average_gyro_measurement_cov = np.eye(3) * self.sigma_gyro_white**2 / averaging_time

            initial_gyro_bias_covariance = (
                uncertainty_increase_in_true_gyro_bias_while_waiting_to_initialize
                + average_gyro_measurement_cov
                + estimated_omega_in_body_frame_cov
            )
            self.P[3:6, 3:6] = initial_gyro_bias_covariance

            self.initialized = True
