import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
from .SunSensorAlgs import compute_body_ang_vel_from_sun_rays
import collections


def skew_symmetric(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


# TODO convert to sqrt form if fixed precision math ends up being an issue

# STATES: WAIT_FOR_CONSTANT_OMEGA, INITIALIZING, INITIALIZED


class Attitude_EKF:
    def __init__(
        self,
        sigma_initial_attitude,  # [rad]
        sigma_gyro_white,  # [rad/sqrt(s)]
        sigma_gyro_bias_deriv,  # [(rad/s)/sqrt(s))]
    ):
        self.state_vector = np.zeros(7)

        self.P = np.zeros((6, 6))
        self.sigma_initial_attitude = sigma_initial_attitude
        self.sigma_gyro_white = sigma_gyro_white
        self.sigma_gyro_bias_deriv = sigma_gyro_bias_deriv

        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= sigma_gyro_white**2
        self.Q[3:6, 3:6] *= sigma_gyro_bias_deriv**2

        self.last_gyro_measurement_time = None

        self.gyro_ringbuff = collections.deque(maxlen=100)
        self.sun_ray_ringbuff = collections.deque(maxlen=40)
        self.Bfield_ringbuff = collections.deque(maxlen=40)

        # State variable pertaining to initialization
        self.initialized = False
        self.attitude_initialized = False
        self.gyro_bias_initialized = False
        self.spinning_at_roughly_constant_rate = False
        self.min_degress_changed_to_init_from_sun_rays = 30  # [deg]  # TODO put me in param file

    def set_ECI_R_b(self, ECI_R_b):
        q = ECI_R_b.as_quat()  # [x, y, z, w]
        self.state_vector[0:4] = [q[3], *q[0:3]]  # [w, x, y, z]

    def set_gyro_bias(self, gyro_bias):
        self.state_vector[4:7] = gyro_bias

    def get_ECI_R_b(self):
        q = self.state_vector[0:4]  # [w, x, y, z]
        return R.from_quat([*q[1:4], q[0]])  # from_quat takes in [x, y, z, w]

    def get_state(self):
        return self.state_vector

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

    def sun_buff_full(self):
        return len(self.sun_ray_ringbuff) == self.sun_ray_ringbuff.maxlen

    def gyro_buff_full(self):
        return len(self.gyro_ringbuff) == self.gyro_ringbuff.maxlen

    def Bfield_buff_full(self):
        return len(self.Bfield_ringbuff) == self.Bfield_ringbuff.maxlen

    def gyro_update(self, gyro_measurement, t):
        # self.gyro_ringbuff.append((t, gyro_measurement))
        # if self.gyro_buff_full():
        #     norms = [np.linalg.norm(v) for (t, v) in self.gyro_ringbuff]
        #     std_deviation = np.std(norms)
        #     self.spinning_at_roughly_constant_rate = std_deviation < np.deg2rad(0.75)

        # if not self.initialized:
        #     if self.spinning_at_roughly_constant_rate:
        #         self.attempt_to_initialize(t)
        #     return
        if self.last_gyro_measurement_time is None:  # first time
            dt = 0
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

    def EKF_update(self, H, innovation, R_noise):
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.pinv(S, 1e-4)  # TODO tuneme
        dx = K @ innovation

        attitude_correction = R.from_rotvec(dx[:3])
        self.set_ECI_R_b(attitude_correction * self.get_ECI_R_b())

        gyro_bias_correction = dx[3:]
        self.set_gyro_bias(self.get_gyro_bias() + gyro_bias_correction)

        # Symmetric Joseph update
        Identity = np.eye(6)
        self.P = (Identity - K @ H) @ self.P @ (Identity - K @ H).T + K @ R_noise @ K.T

    def get_sun_sensor_measurement_jacobian(self, true_sun_ray_ECI):
        H = np.zeros((3, 6))
        H[:3, :3] = -skew_symmetric(true_sun_ray_ECI)
        return H

    def get_Bfield_measurement_jacobian(self, true_B_field_ECI):
        H = np.zeros((3, 6))
        H[:3, :3] = -skew_symmetric(true_B_field_ECI)
        return H

    def sun_sensor_update(self, measured_sun_ray_in_body, true_sun_ray_ECI, t, sigma_sunsensor):
        # sigma_sunsensor [rad]
        measured_sun_ray_in_body /= np.linalg.norm(measured_sun_ray_in_body)
        true_sun_ray_ECI /= np.linalg.norm(true_sun_ray_ECI)
        if not self.initialized:
            if self.spinning_at_roughly_constant_rate:
                self.sun_ray_ringbuff.append((t, measured_sun_ray_in_body, true_sun_ray_ECI))
            return
        print("sun_sensor_update update")
        predicted_sun_ray_ECI = self.get_ECI_R_b().as_matrix() @ measured_sun_ray_in_body
        innovation = true_sun_ray_ECI - predicted_sun_ray_ECI

        s_cross = skew_symmetric(true_sun_ray_ECI)
        Cov_sunsensor = sigma_sunsensor**2 * s_cross @ np.eye(3) @ s_cross.T

        H = self.get_sun_sensor_measurement_jacobian(true_sun_ray_ECI)
        self.EKF_update(H, innovation, Cov_sunsensor)

    def Bfield_update(self, measured_Bfield_in_body, true_Bfield_ECI, t, sigma_Bfield):
        # sigma_Bfield [rad]
        measured_Bfield_in_body /= np.linalg.norm(measured_Bfield_in_body)
        true_Bfield_ECI /= np.linalg.norm(true_Bfield_ECI)
        if not self.initialized:
            if self.spinning_at_roughly_constant_rate:
                self.Bfield_ringbuff.append((t, measured_Bfield_in_body, true_Bfield_ECI))
            return
        print("Bfield update")
        predicted_Bfield_ECI = self.get_ECI_R_b().as_matrix() @ measured_Bfield_in_body
        innovation = true_Bfield_ECI - predicted_Bfield_ECI

        s_cross = skew_symmetric(true_Bfield_ECI)
        Cov_Bfield = sigma_Bfield**2 * s_cross @ np.eye(3) @ s_cross.T

        H = self.get_Bfield_measurement_jacobian(true_Bfield_ECI)
        self.EKF_update(H, innovation, Cov_Bfield)

    def triad(self, sun_ECI, sun_body, B_ECI, B_body):
        cross_ECI = np.cross(sun_ECI, B_ECI)
        cross_body = np.cross(sun_body, B_body)

        cross_ECI /= np.linalg.norm(cross_ECI)
        cross_body /= np.linalg.norm(cross_body)

        ECI_Vectors = np.array([sun_ECI, B_ECI, cross_ECI]).T
        Body_Vectors = np.array([sun_body, B_body, cross_body]).T
        U, sigma, Vt = np.linalg.svd(ECI_Vectors @ Body_Vectors.T)
        ECI_R_body = U @ Vt
        return ECI_R_body
