import numpy as np
from scipy.spatial.transform import Rotation

from brahe.constants import R_EARTH


class Camera:
    def __init__(self, fov: float, R_body_to_camera: np.ndarray, t_body_to_camera: np.ndarray):
        """
        Initialize the Camera object.

        :param fov: The field of view of the camera in radians.
        :param R_body_to_camera: The rotation matrix from the body frame to the camera frame, as a numpy array of shape (3, 3).
        :param t_body_to_camera: The translation vector from the body frame to the camera frame, as a numpy array of shape (3,).
        """
        self.fov = np.deg2rad(fov)
        self.cos_fov = np.cos(self.fov)
        self.R_body_to_camera = R_body_to_camera
        self.t_body_to_camera = t_body_to_camera

    @staticmethod
    def get_ray_and_earth_intersections(ray_dirs: np.ndarray, ray_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of rays, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related as follows:
        M == np.sum(valid_intersections) <= N.

        :param ray_dirs: A numpy array of shape (N, 3) containing the direction vectors of the rays in ECI coordinates.
                         Note that the direction vectors must be normalized.
        :param ray_start: A numpy array of shape (3,) containing the starting point of the rays in ECI coordinates.
        :return: A tuple containing a boolean array  of shape (N,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points in ECI coordinates.
        """
        assert np.allclose(np.linalg.norm(ray_dirs, axis=1), 1), "ray_dirs must be normalized"

        # As = np.sum(ray_dirs ** 2, axis=1)  # this is always 1 since the rays are normalized
        Bs = 2 * ray_dirs @ ray_start
        C = np.sum(ray_start ** 2) - R_EARTH ** 2
        assert C > 0, "The ray start location is inside the Earth!"

        discriminants = Bs ** 2 - 4 * C

        """
        Since C > 0 and np.all(As > 0), if the roots are real they must have the same sign.
        Bs < 0 implies that the slope at x = 0 is negative, so the roots are positive.
        Intuitively, this check is equivalent to np.dot(ray_dir, offset) < 0 which checks if ray_dir is in
        the half-space that is pointing towards the Earth. 
        """
        valid_intersections = (discriminants >= 0) & (Bs < 0)

        # pick the smaller of the two positive roots from the quadratic formula, since it is closer to the camera
        ts = (-Bs[valid_intersections] - np.sqrt(discriminants[valid_intersections])) / 2
        intersection_points = ray_start + ts[:, np.newaxis] * ray_dirs[valid_intersections, :]

        assert intersection_points.shape[0] == np.sum(valid_intersections)
        return valid_intersections, intersection_points

    def sample_bearing_unit_vectors(self, N: int) -> np.ndarray:
        """
        Sample N random bearing unit vectors in the body frame.

        :param N: The number of bearing unit vectors to sample.
        :return: A numpy array of shape (N, 3) containing the sampled bearing unit vectors.
        """
        phi = 2 * np.pi * np.random.random(N)
        # uniformly sample cos(theta) to get a uniform distribution on the unit sphere
        theta = np.arccos(np.random.uniform(self.cos_fov, 1))
        return Rotation.from_euler("ZX", np.column_stack((phi, theta))).apply(np.array([0, 0, 1]))

    def get_earth_intersections(self,
                                bearing_unit_vectors: np.ndarray,
                                cubesat_position_eci: np.ndarray,
                                R_body_to_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of pixel coordinates, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related as follows:
        M == np.sum(valid_intersections) <= N.

        :param bearing_unit_vectors: A numpy array of shape (N, 3) containing the bearing unit vectors in the body frame.
        :param cubesat_position_eci: A numpy array of shape (3,) containing the ECI coordinates of the satellite's
                                     body frame origin.
        :param R_body_to_eci: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to ECI.
        :return: A tuple containing a boolean array  of shape (M,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points in ECI coordinates.
        """
        bearing_unit_vectors_eci = (R_body_to_eci @ bearing_unit_vectors.T).T

        camera_position_eci = cubesat_position_eci + self.t_body_to_camera
        return self.get_ray_and_earth_intersections(bearing_unit_vectors_eci, camera_position_eci)
