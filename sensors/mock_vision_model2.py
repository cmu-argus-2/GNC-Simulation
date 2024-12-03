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
        self.fov = fov
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
        theta = np.arccos(np.random.uniform(self.cos_fov, 1, N))
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


def test():
    import yaml
    from matplotlib import pyplot as plt
    from matplotlib import use

    use("TkAgg")

    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    camera_params = config["satellite"]["camera"]
    camera = Camera(
        image_dimensions=np.array([camera_params["image_width"], camera_params["image_height"]]),
        K=np.asarray(camera_params["intrinsics"]),
        R_body_to_camera=np.eye(3),
        t_body_to_camera=np.zeros(3)
    )

    pixel_coordinates = camera.sample_pixel_coordinates(1)
    bearing_unit_vectors = camera.pixel_coordinates_to_bearing_unit_vectors(pixel_coordinates)
    print(pixel_coordinates)
    print(bearing_unit_vectors)

    # cubesat_position = np.array([R_EARTH + 600e3, 0, 0])
    #
    # # define nadir cubesat attitude
    # y_axis = [0, 1, 0]  # along orbital angular momentum
    # z_axis = cubesat_position / np.linalg.norm(cubesat_position)  # along radial vector
    # x_axis = np.cross(y_axis, z_axis)
    # R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])
    #
    # valid_intersections, intersection_points = camera.get_earth_intersections(pixel_coordinates, cubesat_position,
    #                                                                           R_body_to_eci)
    # print(valid_intersections)
    # print(intersection_points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(states[:, 0], states[:, 1], states[:, 2], label="True orbit")
    # ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label="Estimated orbit")
    # altitude_normalized_landmarks = landmarks
    # ax.scatter(altitude_normalized_landmarks[:, 0],
    #            altitude_normalized_landmarks[:, 1],
    #            altitude_normalized_landmarks[:, 2],
    #            label="Landmarks")
    # bearing_unit_vectors = 100e3 * camera.pixel_coordinates_to_bearing_unit_vectors(pixel_coordinates)
    # bearing_unit_vectors = np.array([R_body_to_eci @ bearing_unit_vector
    #                                  for R_body_to_eci, bearing_unit_vector in
    #                                  zip(Rs_body_to_eci, bearing_unit_vectors)])
    # ax.quiver(states[times, 0], states[times, 1], states[times, 2],
    #           bearing_unit_vectors[:, 0], bearing_unit_vectors[:, 1], bearing_unit_vectors[:, 2],
    #           color="r", label="Bearing Unit Vectors")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend()
    # plt.show()


if __name__ == "__main__":
    np.random.seed(69420)
    test()
