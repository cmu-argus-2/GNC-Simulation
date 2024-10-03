from dataclasses import dataclass
import numpy as np

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


class SolarGeneration:
    """
    TODO: validate that occlusions from the Moon are negligible, since they are not accounted for in this model.
    """
    SOLAR_FLUX = 1373  # The solar power flux at the Earth's orbital radius in W/m^2.
    PANEL_EFFICIENCY = 0.2  # The efficiency of the solar panels, as a fraction in [0, 1].

    def __init__(self, config: dict) -> None:
        self.surfaces = self.parse_config(config)
        self.sample_resolution = 0.05  # The resolution of the occlusion sampling grid, in meters.

    def parse_config(self, config: dict) -> list[Surface]:
        pass

    @staticmethod
    def has_intersection(surface: Surface, ray_start: np.ndarray, ray_dir: np.ndarray) -> bool:
        """
        Check if a ray intersects a Surface.

        :param surface: A Surface object representing the surface.
        :param ray_start: The starting point of the ray in the body frame.
        :param ray_dir: The direction of the ray in the body frame.
        :return: True if the ray intersects the surface, False otherwise.
        """
        cos_theta = np.dot(surface.normal, ray_dir)
        if cos_theta < 1e-3:
            # The ray is parallel to the surface.
            return False

        t = np.dot(surface.normal, surface.center - ray_start) / cos_theta
        if t <= 0:
            # The intersection point is behind the ray start.
            return False

        intersection_point = ray_start + t * ray_dir
        intersection_point_offset = intersection_point - surface.center
        x = np.dot(intersection_point_offset, surface.x_dir)
        y = np.dot(intersection_point_offset, surface.y_dir)

        return -surface.width / 2 <= x <= surface.width / 2 and \
            -surface.height / 2 <= y <= surface.height / 2

    @staticmethod
    def in_earths_shadow(position_eci: np.ndarray, sun_vector_eci: np.ndarray) -> bool:
        """
        Check if the satellite is in the Earth's shadow.

        :param position_eci: The position of the satellite in ECI.
        :param sun_vector_eci: The sun vector in the ECI frame.
        :return: True if the satellite is in the Earth's shadow, False otherwise.
        """
        a = np.dot(sun_vector_eci, sun_vector_eci)
        b = 2 * np.dot(position_eci, sun_vector_eci)
        c = np.dot(position_eci, position_eci) - R_EARTH ** 2
        """
        We want to find if there exists a t such that ||position_eci + t * sun_vector_eci||^2 <= R_EARTH^2.
        This simplifies to a * t^2 + b * t + c <= 0, which is true if and only if the discriminant is non-negative.
        """
        return b ** 2 - 4 * a * c >= 0

    def get_power_output(self, epoch: Epoch, position_eci: np.ndarray, R_body_to_eci: np.ndarray) -> float:
        """
        Get the power output of the solar panels given the time, position, and attitude of the satellite.

        :param epoch: The current epoch as an instance of brahe's Epoch class.
        :param position_eci: The position of the satellite in ECI.
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame.
        :return: The power output of the solar panels, in Watts.
        """
        sun_vector_eci = brahe.ephemerides.sun_position(epoch)
        if SolarGeneration.in_earths_shadow(position_eci, sun_vector_eci):
            return 0

        sun_vector_body = R_body_to_eci.T @ sun_vector_eci
        power_output = 0
        for i, surface in enumerate(self.surfaces):
            if not surface.is_solar_panel:
                continue

            cos_theta = np.dot(surface.normal, sun_vector_body)
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
            exposed_count = 0
            for ray_start in ray_starts:
                for j, other_surface in enumerate(self.surfaces):
                    if i == j:
                        continue

                    if self.has_intersection(other_surface, ray_start, sun_vector_body):
                        break
                else:
                    exposed_count += 1

            # Calculate the power output
            exposed_area = (exposed_count / ray_starts.shape[0]) * surface.width * surface.height
            power_output += cos_theta * exposed_area * self.SOLAR_FLUX * self.PANEL_EFFICIENCY

        return power_output
