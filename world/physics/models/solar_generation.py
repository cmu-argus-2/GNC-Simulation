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

    @staticmethod
    def parse_config(config: dict) -> list[Surface]:
        pass

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
        if cos_theta < 1e-3:
            # The rays are parallel to the surface.
            return np.zeros(ray_starts.shape[0], dtype=bool)

        ts = (surface.center - ray_starts) @ surface.normal / cos_theta
        intersection_points = ray_starts + np.outer(ts, ray_dir)
        intersection_point_offsets = intersection_points - surface.center
        xs = intersection_point_offsets @ surface.x_dir
        ys = intersection_point_offsets @ surface.y_dir

        return ts > 0 & (np.abs(xs) <= surface.width / 2) & (np.abs(ys) <= surface.height / 2)

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
            occluded_rays = np.zeros(ray_starts.shape[0], dtype=bool)
            for j, other_surface in enumerate(self.surfaces):
                if i == j:
                    continue
                # TODO: there are probably some simple heuristics we can use to skip some surfaces

                occluded_rays |= SolarGeneration.get_intersections(other_surface, ray_starts, sun_vector_body)

            # Calculate the power output
            exposed_area = np.mean(~occluded_rays) * surface.width * surface.height
            power_output += cos_theta * exposed_area * self.SOLAR_FLUX * self.PANEL_EFFICIENCY

        return power_output
