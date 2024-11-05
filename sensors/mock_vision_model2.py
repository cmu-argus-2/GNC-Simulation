import numpy as np
from scipy.spatial.transform import Rotation

from brahe.constants import R_EARTH


class Camera:
    def __init__(self, image_dimensions: np.ndarray, K: np.ndarray,
               R_body_to_camera: np.ndarray, t_body_to_camera: np.ndarray):
        """
        Initialize the Camera object.

        :param image_dimensions: The width and height of the image in pixels, as a numpy array of shape (2,).
        :param K: The camera matrix, as a numpy array of shape (3, 3).
        :param R_body_to_camera: The rotation matrix from the body frame to the camera frame, as a numpy array of shape (3, 3).
        :param t_body_to_camera: The translation vector from the body frame to the camera frame, as a numpy array of shape (3,).
        """
        self.image_dimensions = image_dimensions
        self.K = K
        self.K_inv = np.linalg.inv(K)
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

    def sample_pixel_coordinates(self, N: int) -> np.ndarray:
        """
        Sample n random pixel coordinates.

        :param N: The number of pixel coordinates to sample.
        :return: A numpy array of shape (N, 2) containing the sampled pixel coordinates.
        """
        return np.random.rand(N, 2) * self.image_dimensions[np.newaxis, :]

    def pixel_coordinates_to_bearing_unit_vectors(self, pixel_coordinates: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to bearing unit vectors in the body frame.

        :param pixel_coordinates: A numpy array of shape (N, 2) containing the pixel coordinates.
                                  The origin is the top-left corner of the image, the x-axis points to the right,
                                  and the y-axis points down.
        :return: A numpy array of shape (N, 3) containing the bearing unit vectors in the body frame.
        """
        assert len(pixel_coordinates.shape) == 2, "pixel_coordinates must be a 2D array"
        assert pixel_coordinates.shape[1] == 2, "pixel_coordinates must have 2 columns"

        screen_coordinates = 1 - 2 * pixel_coordinates / self.image_dimensions

        screen_coordinates_homogeneous = np.column_stack((screen_coordinates, np.ones(screen_coordinates.shape[0])))
        bearing_vectors = (self.R_body_to_camera.T @ self.K_inv @ screen_coordinates_homogeneous.T).T

        bearing_unit_vectors = bearing_vectors / np.linalg.norm(bearing_vectors, axis=1, keepdims=True)
        return bearing_unit_vectors

    def get_earth_intersections(self,
                                pixel_coordinates: np.ndarray,
                                cubesat_position_eci: np.ndarray,
                                R_body_to_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of pixel coordinates, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related as follows:
        M == np.sum(valid_intersections) <= N.

        :param pixel_coordinates: A numpy array of shape (N, 2) containing the pixel coordinates.
        :param cubesat_position_eci: A numpy array of shape (3,) containing the ECI coordinates of the satellite's
                                     body frame origin.
        :param R_body_to_eci: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to ECI.
        :return: A tuple containing a boolean array  of shape (M,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points in ECI coordinates.
        """
        bearing_unit_vectors = self.pixel_coordinates_to_bearing_unit_vectors(pixel_coordinates)
        bearing_unit_vectors_eci = (R_body_to_eci @ bearing_unit_vectors.T).T

        camera_position_eci = cubesat_position_eci + self.t_body_to_camera
        return self.get_ray_and_earth_intersections(bearing_unit_vectors_eci, camera_position_eci)
