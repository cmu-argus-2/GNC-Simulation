import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
from .PlaneFitting import fit_plane_RANSAC


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

    def compute_body_ang_vel_from_sun_rays(self):
        # Find the satellite rotaiton axis as the unit vector with a constant dot product with all measured sun vectors.
        # This assumes:
        #     the satellite has a constant angular velocity
        #     the sat has rotated less than 180 degrees between every other sun ray measurement
        sun_rays = np.array([sun_ray for (t, sun_ray) in self.init_sun_rays])
        plane, inlier_idxs = fit_plane_RANSAC(sun_rays, tolerance=0.01)  # TODO tune tolerance

        # TODO ensure we don't div-by-0
        plane_normal = plane[0:3] / np.linalg.norm(plane[0:3])

        # TODO dont accept the plane if the sun rays are bunched up (the differences between measurements will get swamped out by noise)
        rotation_axis = plane_normal

        # Choose sign for rotation axis that is consistent with the sun ray measurement history
        N = len(self.init_sun_rays)
        N_consistent_with_axis = 0
        N_consistent_with_opposite_axis = 0
        for i in range(1, N - 1):
            _, sun_ray0 = self.init_sun_rays[i - 1]
            _, sun_ray1 = self.init_sun_rays[i]
            _, sun_ray2 = self.init_sun_rays[i + 1]

            delta_ray10 = sun_ray1 - sun_ray0
            delta_ray21 = sun_ray2 - sun_ray1

            # negative cross product because sat's rotation is opposite the apparent motion of the sun relative to the sat
            rough_axis_estimate = -np.cross(delta_ray10, delta_ray21)
            rough_axis_estimate /= np.linalg.norm(rough_axis_estimate)
            similarity = np.dot(rough_axis_estimate, rotation_axis)

            if similarity > 0.9:
                N_consistent_with_axis += 1
            elif similarity < -0.9:
                N_consistent_with_opposite_axis += 1

        min_consistent = 0.7 * N
        N_consistent = N_consistent_with_axis
        if N_consistent_with_opposite_axis >= min_consistent:
            rotation_axis *= -1
            N_consistent = N_consistent_with_opposite_axis
        elif N_consistent_with_axis < min_consistent:
            print("Couldn't find enough consistent data")
            return None  # couldn't find enough consistent data

        # Compute the angular velocity magnitude
        omega_norm_estimates = []
        for i in range(N - 1):
            t0, sun_ray0 = self.init_sun_rays[i]
            t1, sun_ray1 = self.init_sun_rays[i + 1]

            # get componenets perpendicular to the rotation axis
            sun_ray0_perp = sun_ray0 - rotation_axis * np.dot(sun_ray0, rotation_axis)
            sun_ray1_perp = sun_ray1 - rotation_axis * np.dot(sun_ray1, rotation_axis)

            sun_ray0_perp_unit = sun_ray0_perp / np.linalg.norm(sun_ray0_perp)
            sun_ray1_perp_unit = sun_ray1_perp / np.linalg.norm(sun_ray1_perp)

            dot_prod = np.dot(sun_ray0_perp_unit, sun_ray1_perp_unit)
            dot_prod = np.clip(dot_prod, -1, 1)
            delta_theta = np.arccos(dot_prod)
            delta_time = t1 - t0

            omega_norm_estimates.append(delta_theta / delta_time)

        omega_norm_estimate = np.average(omega_norm_estimates)

        cov = np.eye(3) * self.sigma_sunsensor**2 / N_consistent
        return rotation_axis * omega_norm_estimate, cov

    def attempt_to_initialize(self, t):
        if len(self.init_sun_rays) < 10:
            return

        average_gyro_measurement = np.average([gyro for (t, gyro) in self.init_gyro_measurements], axis=0)
        print("average_gyro_measurement [deg/s]: ", np.rad2deg(average_gyro_measurement))

        estimated_omega_in_body_frame, estimated_omega_in_body_frame_cov = self.compute_body_ang_vel_from_sun_rays()
        if estimated_omega_in_body_frame is not None:
            print("estimated_omega_in_body_frame [deg/s]: ", np.rad2deg(estimated_omega_in_body_frame))

            gyro_bias_estimate = average_gyro_measurement - estimated_omega_in_body_frame
            inital_ECI_R_b = self.get_ECI_R_b()

            # Set the attitude
            # rotation since getting initial attitude
            time_to_initialize = t - self.time_of_initial_attitude
            b0_R_b1 = R.from_rotvec(estimated_omega_in_body_frame * time_to_initialize)
            current_ECI_R_b = inital_ECI_R_b * b0_R_b1
            self.set_ECI_R_b(current_ECI_R_b)

            # Set the Gyro Bias
            self.set_gyro_bias(gyro_bias_estimate)

            # Set the Initial Covariance
            self.P = np.zeros((6, 6))

            # TODO Deriving the Initial Attitude Covariance
            self.P[0:3, 0:3] = np.eye(3) * self.sigma_initial_attitude**2

            # TODO Deriving the initial gyro bias covaraince
            random_walk_in_true_gyro_bias_while_waiting_to_initialize = (
                np.eye(3) * time_to_initialize * self.sigma_gyro_bias_deriv**2
            )
            averaging_time = time_to_initialize
            average_gyro_measurement_cov = np.eye(3) * self.sigma_gyro_white**2 / (averaging_time)
            initial_gyro_bias_covaraince = (
                random_walk_in_true_gyro_bias_while_waiting_to_initialize
                + average_gyro_measurement_cov
                + estimated_omega_in_body_frame_cov
            )
            self.P[3:6, 3:6] = initial_gyro_bias_covaraince

            self.initialized = True

        # if delta_degrees
        # min_degress_changed_to_init_from_sun_rays
