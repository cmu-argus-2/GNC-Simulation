from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

import brahe
from brahe.epoch import Epoch
from brahe.constants import R_EARTH


@dataclass
class Surface:
    """
    Assumes that the surface is a rectangle.
    Defines a plane by normal.T @ (x - center) = 0.
    The four corners of the surface are defined by the following equations.
    corner = center ± (width / 2) * x_dir ± (height / 2) * y_dir
    """
    is_solar_panel: bool  # True if the surface is a solar panel, False otherwise.
    center: np.ndarray  # The center of the surface in the body frame.
    normal: np.ndarray  # The unit normal vector of the surface in the body frame.
    x_dir: np.ndarray  # A unit vector in the direction parallel to the width of the rectangular surface.
    y_dir: np.ndarray  # A unit vector in the direction parallel to the height of the rectangular surface.
    width: float  # The width of the surface in meters.
    height: float  # The height of the surface in meters.

    def transform(self, transformation_matrix: np.ndarray) -> "Surface":
        """
        Transforms this Surface by applying the given transformation matrix to all of its constituent vectors.

        :param transformation_matrix: The 3x3 transformation matrix to apply.
        :return: The resulting Surface.
        """
        return Surface(is_solar_panel=self.is_solar_panel,
                       center=transformation_matrix @ self.center,
                       normal=transformation_matrix @ self.normal,
                       x_dir=transformation_matrix @ self.x_dir,
                       y_dir=transformation_matrix @ self.y_dir,
                       width=self.width,
                       height=self.height)

    def flip_normal(self) -> "Surface":
        """
        Flips the normal vector of the surface.

        :return: The resulting Surface.
        """
        return Surface(is_solar_panel=self.is_solar_panel,
                       center=self.center,
                       normal=-self.normal,
                       x_dir=self.x_dir,
                       y_dir=self.y_dir,
                       width=self.width,
                       height=self.height)


class SolarGeneration:
    """
    TODO: validate that occlusions from the Moon are negligible, since they are not accounted for in this model.
    """
    SOLAR_FLUX = 1373  # The solar power flux at the Earth's orbital radius in W/m^2.
    PANEL_EFFICIENCY = 0.15  # The efficiency of the solar panels, as a fraction in [0, 1].

    def __init__(self, deployables_dir: np.ndarray, deployables_tilt_angle: float) -> None:
        self.surfaces = SolarGeneration.get_solar_config(deployables_dir, deployables_tilt_angle)
        self.sample_resolution = 0.05  # The resolution of the occlusion sampling grid, in meters.

    @staticmethod
    def get_solar_config(deployables_dir: np.ndarray, deployables_tilt_angle: float) -> list[Surface]:
        """
        Returns a list of Surface objects representing the solar panels on the satellite.

        :param deployables_dir: A 3-element numpy array representing the direction of the deployable solar panels.
                                Must be a unit vector with exactly one non-zero element.
        :param deployables_tilt_angle: The angle in radians by which the deployable solar panels are tilted.
                                       See the diagram below for the definition of the tilt angle.
        :return: A list of Surface objects representing the configuration of the satellite.
        """
        assert deployables_dir.shape == (3,), "deployables_dir must be a 3-element numpy array."
        assert np.sum(np.abs(deployables_dir) == 1) == 1 and np.sum(deployables_dir == 0) == 2, \
            "deployables_dir must be a unit vector with exactly one non-zero element."

        R = 0.05  # Half the side length of the cubesat in meters

        cube_surfaces = [Surface(is_solar_panel=True,
                                 center=np.array([R, 0, 0]),
                                 normal=np.array([1, 0, 0]),
                                 x_dir=np.array([0, 1, 0]),
                                 y_dir=np.array([0, 0, 1]),
                                 width=0.1,
                                 height=0.1)]  # +x face
        cube_surfaces.append(cube_surfaces[0].transform(np.array([[0, 1, 0],
                                                                  [1, 0, 0],
                                                                  [0, 0, 1]])))  # +y face
        cube_surfaces.append(cube_surfaces[0].transform(np.array([[0, 0, 1],
                                                                  [0, 1, 0],
                                                                  [1, 0, 0]])))  # +z face
        cube_surfaces.append(cube_surfaces[0].transform(np.array([[-1, 0, 0],
                                                                  [0, 1, 0],
                                                                  [0, 0, 1]])))  # -x face
        cube_surfaces.append(cube_surfaces[1].transform(np.array([[1, 0, 0],
                                                                  [0, -1, 0],
                                                                  [0, 0, 1]])))  # -y face
        cube_surfaces.append(cube_surfaces[2].transform(np.array([[1, 0, 0],
                                                                  [0, 1, 0],
                                                                  [0, 0, -1]])))  # -z face

        """
        deployables_tilt_angle
                |  /    deployables_dir
                | /            ^
        ________|/             |
        |       |              |
        |       |   --> deployables_offset_dir
        |_______|
        deployables_tilt_dir is into the page
        """
        deployables_tilt_dir = np.roll(deployables_dir, 1)  # doesn't matter which of the 4 perpendiculars we choose
        deployables_offset_dir = np.cross(deployables_tilt_dir, deployables_dir)
        deployables_tilt = Rotation.from_rotvec(deployables_tilt_angle * deployables_tilt_dir).as_matrix()

        deployable_surfaces = [Surface(is_solar_panel=True,
                                       center=(np.eye(3) + deployables_tilt) * R * deployables_dir + R * deployables_offset_dir,
                                       normal=deployables_tilt @ deployables_offset_dir,
                                       x_dir=deployables_tilt_dir,
                                       y_dir=deployables_tilt @ deployables_dir,
                                       width=0.1,
                                       height=0.1)]

        # add the 3 other rotated copies of the deployable surfaces
        rot_90 = Rotation.from_rotvec((np.pi / 2) * deployables_dir).as_matrix()
        for i in range(0, 4):
            deployable_surfaces.append(deployable_surfaces[0].transform(rot_90 ** i))

        # add deployable surfaces with flipped normals
        for deployable_surface in deployable_surfaces:
            deployable_surfaces.append(deployable_surface.flip_normal())

        return cube_surfaces + deployable_surfaces

    @staticmethod
    def get_intersections(surface: Surface, ray_starts: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
        """
        Check if a set of rays intersect a Surface.

        :param surface: A Surface object representing the surface.
        :param ray_starts: A numpy array of shape (n, 3) representing the starting points of the rays in the body frame.
        :param ray_dir: A 3-element array representing the direction of the rays in the body frame.
        :return: A boolean array of shape (n,) indicating whether the corresponding rays intersect the surface.
        """
        cos_theta = np.dot(surface.normal, ray_dir)
        if np.abs(cos_theta) < 1e-3:
            # The rays are parallel to the surface.
            return np.zeros(ray_starts.shape[0], dtype=bool)

        ts = (surface.center - ray_starts) @ surface.normal / cos_theta
        intersection_points = ray_starts + np.outer(ts, ray_dir)
        intersection_point_offsets = intersection_points - surface.center
        xs = intersection_point_offsets @ surface.x_dir
        ys = intersection_point_offsets @ surface.y_dir

        return (ts > 0) & (np.abs(xs) <= surface.width / 2) & (np.abs(ys) <= surface.height / 2)

    @staticmethod
    def in_earths_shadow(position_eci: np.ndarray, sun_dir_eci: np.ndarray) -> bool:
        """
        Check if the satellite is in the Earth's shadow.

        :param position_eci: The position of the satellite in ECI.
        :param sun_dir_eci: A unit vector pointing from the Earth to the Sun in ECI.
        :return: True if the satellite is in the Earth's shadow, False otherwise.
        """
        """
        We want to find if there exists a non-negative t such that ||position_eci + t * sun_vector_eci||^2 <= R_EARTH^2.
        This simplifies to finding a non-negative value of t such that f(t) = a * t^2 + b * t + c <= 0, where:
        a = np.dot(sun_vector_eci, sun_vector_eci) = 1, since sun_vector_eci is a unit vector.
        b = 2 * np.dot(position_eci, sun_vector_eci)
        c = np.dot(position_eci, position_eci) - R_EARTH^2 > 0, since the satellite is outside the Earth.
        For f(t) <= 0, we need the discriminant to be non-negative (i.e., b^2 - 4 * a * c >= 0).
        Moreover, we need f'(0) = b < 0, so that the minimum of the parabola is in the t >= 0 region.
        The condition on b also has a geometric interpretation: b = 2 * np.dot(position_eci, sun_vector_eci) >= 0 means
        that the satellite is on the side of the Earth facing the Sun, so it cannot be in the Earth's shadow. 
        """
        b = 2 * np.dot(position_eci, sun_dir_eci)
        if b >= 0:
            return False
        c = np.dot(position_eci, position_eci) - R_EARTH ** 2
        return b ** 2 - 4 * c >= 0

    def get_power_output(self, epoch: Epoch, position_eci: np.ndarray, R_body_to_eci: np.ndarray) -> float:
        """
        Get the power output of the solar panels given the time, position, and attitude of the satellite.

        :param epoch: The current epoch as an instance of brahe's Epoch class.
        :param position_eci: The position of the satellite in ECI.
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame.
        :return: The power output of the solar panels, in Watts.
        """
        sun_dir_eci = brahe.ephemerides.sun_position(epoch)
        sun_dir_eci /= np.linalg.norm(sun_dir_eci)
        if SolarGeneration.in_earths_shadow(position_eci, sun_dir_eci):
            return 0

        sun_dir_body = R_body_to_eci.T @ sun_dir_eci
        power_output = 0
        for i, surface in enumerate(self.surfaces):
            if not surface.is_solar_panel:
                continue

            cos_theta = np.dot(surface.normal, sun_dir_body)
            if cos_theta < 0:
                continue  # The surface is facing away from the sun.

            # Sample the surface
            x_samples = np.linspace(-surface.width / 2, surface.width / 2,
                                    int(np.ceil(surface.width / self.sample_resolution)))
            y_samples = np.linspace(-surface.height / 2, surface.height / 2,
                                    int(np.ceil(surface.height / self.sample_resolution)))
            x_samples, y_samples = np.meshgrid(x_samples, y_samples)
            x_samples, y_samples = x_samples.flatten(), y_samples.flatten()
            ray_starts = surface.center + np.outer(x_samples, surface.x_dir) + np.outer(y_samples, surface.y_dir)

            # Check how many of the sampled rays are exposed to the sun
            occluded_rays = np.zeros(ray_starts.shape[0], dtype=bool)
            for j, other_surface in enumerate(self.surfaces):
                if i == j:
                    continue
                # TODO: there are probably some simple heuristics we can use to skip some surfaces

                occluded_rays |= SolarGeneration.get_intersections(other_surface, ray_starts, sun_dir_body)

            # Calculate the power output
            exposed_area = np.mean(~occluded_rays) * surface.width * surface.height
            power_output += cos_theta * exposed_area * self.SOLAR_FLUX * self.PANEL_EFFICIENCY

        return power_output
