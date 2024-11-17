import numpy as np
from scipy.spatial.transform import Rotation


def columnify(vec: np.ndarray):
    if type(vec) != np.ndarray and type(vec) != np.ma.masked_array:
        vec = np.array(vec)
    return vec.reshape((vec.shape[0], 1))


class RayCaster():
    def get_ray_and_sphere_intersections(
        self,
        ray_dirs: np.ndarray,
        ray_start: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with a sphere.
        The input number of rays, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related via the following relationship:
        M == np.sum(valid_intersections) <= N.

        :param ray_dirs: A numpy array of shape (N, 3) containing the direction vectors of the rays.
        :param ray_start: A numpy array of shape (3,) containing the starting point of the rays.
        :param sphere_center: A numpy array of shape (3,) containing the center of the sphere.
        :param sphere_radius: The radius of the sphere.
        :return: A tuple containing a boolean array  of shape (N,) indicating which rays intersected the sphere,
                 and a numpy array of shape (M, 3) containing the intersection points.
        """
        As = np.linalg.norm(ray_dirs, ord=2, axis=1)**2
        k = ray_start - sphere_center
        Ks = np.repeat([k], repeats=ray_dirs.shape[0], axis=0)
        Bs = 2 * np.sum(ray_dirs*Ks, axis=1)
        Cs = np.linalg.norm(Ks, ord=2, axis=1)**2 - sphere_radius**2
        Ts = self.solve_quadratic_equations(As, Bs, Cs)
        valid_intersections = self.get_valid_intersection_parameters(Ts)
        # valid_intersections = False
        ts = np.real(np.min(Ts[valid_intersections, :], axis=1))  # TODO: should real be called first?
        intersection_points = ray_start + ts[:, None] * ray_dirs[valid_intersections, :]
        assert intersection_points.shape[0] == np.sum(valid_intersections)
        return valid_intersections, intersection_points

    def get_valid_intersection_parameters(
        self,
        Ts: np.ndarray
    ) -> np.ndarray:
        is_complex_or_negative = np.logical_or(np.iscomplex(Ts), Ts < 0)
        return np.logical_not(np.any(is_complex_or_negative, axis=1))

    def solve_quadratic_equations(
        self,
        As: np.ndarray,
        Bs: np.ndarray,
        Cs: np.ndarray,
    ) -> np.ndarray:
        Ds = columnify(Bs**2 - 4.0*As*Cs)
        Bs = columnify(Bs)
        As = columnify(As)
        return ([-1,-1]*Bs + [1,-1]*np.emath.sqrt(Ds)) / (2*As)


class Camera():
    def __init__(
        self,
        image_width: int,
        image_height: int,
        focal_length: float,
        position_in_cubesat_frame: np.ndarray,
        orientation_in_cubesat_frame: np.ndarray,
    ) -> None:
        self.dims = np.array([image_width, image_height])
        self.f = focal_length
        self.R_sat_cam = self.get_rotation(
            orientation_in_cubesat_frame
        )
        self.T_sat_cam = self.get_homogeneous_transform(
            position_in_cubesat_frame, orientation_in_cubesat_frame
        )

    def convert_pixel_coordinates_to_ecef_ray_directions(
        self,
        coords: np.ndarray,
        cubesat_att_in_ecef: np.ndarray,
    ) -> np.ndarray:
        coords_ndc = (coords + 0.5) / self.dims
        screen_coordinates = [1, -1] * (2 * coords_ndc - 1)

        ray_dirs_xy = -screen_coordinates / self.f
        ray_dirs_z = self.f * np.ones((screen_coordinates.shape[0], 1))
        cam_ray_dirs = np.hstack((ray_dirs_xy, ray_dirs_z))

        R_ecef_sat = self.get_rotation(cubesat_att_in_ecef)
        return (R_ecef_sat @ self.R_sat_cam @ cam_ray_dirs.T).T

    def get_camera_position_in_ecef(
        self,
        cubesat_pos_in_ecef: np.ndarray,
        cubesat_att_in_ecef: np.ndarray,
    ) -> np.ndarray:
        T_ecef_sat = self.get_homogeneous_transform(
            cubesat_pos_in_ecef, cubesat_att_in_ecef,
        )
        cam_pos_in_cam = np.array([0.0, 0.0, 0.0, 1.0])
        cam_pos_in_ecef = T_ecef_sat @ self.T_sat_cam @ cam_pos_in_cam
        return cam_pos_in_ecef[:3]

    def get_homogeneous_transform(
        self,
        translation: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        R = self.get_rotation(orientation)
        t_vec = columnify(translation)
        return np.block([
            [R, t_vec],
            [np.zeros(3), 1.0],
        ])

    def get_rotation(
        self,
        orientation: np.ndarray,
    ) -> np.ndarray:
        return Rotation.from_quat(orientation, scalar_first=True).as_matrix()


class MockVisionModel:
    def __init__(
        self,
        camera: Camera,
        max_correspondences: int,
        earth_radius: float,
        pixel_noise_mean=np.zeros(2),
        pixel_noise_std_dev=np.zeros(2),
    ) -> None:
        self.caster = RayCaster()
        self.mean = pixel_noise_mean
        self.sd = pixel_noise_std_dev
        self.cam = camera
        self.N = max_correspondences
        self.R = earth_radius

    def get_measurement(
        self,
        cubesat_position_in_ecef: np.ndarray,
        cubesat_attitude_in_ecef: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        #TODO: convert eci to ecef

        # Sample N pixels
        pixel_coords, pixel_coords_trunc = self.sample_pixel_coordinates()

        # Add noise to sampled pixels
        noisy_coords = self.add_gaussian_noise(pixel_coords)

        # Get ray directions for each sampled pixel
        ray_dirs = self.cam.convert_pixel_coordinates_to_ecef_ray_directions(
            noisy_coords, cubesat_attitude_in_ecef
        )

        # Get ECEF camera position as a product of several rigid body transforms
        cam_pos = self.cam.get_camera_position_in_ecef(
            cubesat_position_in_ecef, cubesat_attitude_in_ecef
        )
        # Get ECEF surface coordinates of rays
        valid_intersections, ecef_coords = self.get_ecef_ray_and_earth_intersections(
            ray_dirs, cam_pos
        )
        # Filter out pixel coordinates that did not intersect the Earth, ecef_coords is already filtered
        return pixel_coords_trunc[valid_intersections, :], ecef_coords

    def add_gaussian_noise(
        self,
        coords: np.ndarray
    ) -> np.ndarray:
        noise = np.random.normal(loc=self.mean, scale=self.sd, size=coords.shape)
        return coords + noise

    def sample_pixel_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        coords = np.random.uniform(high=self.cam.dims, size=(self.N, 2))
        coords_trunc = np.int32(coords)
        return coords, coords_trunc

    def get_ecef_ray_and_earth_intersections(
        self,
        ray_dirs: np.ndarray,
        camera_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of rays, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related via the following relationship:
        M == np.sum(valid_intersections) <= N.

        :param ray_dirs: A numpy array of shape (N, 3) containing the direction vectors of the rays.
        :param camera_pos: A numpy array of shape (3,) containing the position of the camera in ECEF coordinates.
        :return: A tuple containing a boolean array of shape (N,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points.
        """
        earth_pos = np.zeros(3)
        return self.caster.get_ray_and_sphere_intersections(
            ray_dirs=ray_dirs, ray_start=camera_pos,
            sphere_center=earth_pos, sphere_radius=self.R
        )
