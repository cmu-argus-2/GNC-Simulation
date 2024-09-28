from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation


def columnify(vec: np.ndarray):
    return vec.reshape((vec.shape[0], 1))


class PixelEcefCorrespondence():
    def __init__(
        self,
        pixel_coord: np.ndarray,
        ecef_coord: np.ndarray,
    ) -> None:
        self.pixel_coordinate = pixel_coord
        self.ecef_coordinate = ecef_coord


class RayCaster():
    def get_ray_and_sphere_intersections(
        self,
        ray_dirs: np.ndarray,
        ray_start: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ) -> np.ndarray:
        As = np.linalg.norm(ray_dirs, ord=2, axis=1)**2
        k = ray_start - sphere_center
        Ks = np.repeat([k], repeats=ray_dirs.shape[0], axis=0)
        Bs = 2 * np.sum(ray_dirs*Ks, axis=1)
        Cs = np.linalg.norm(Ks, ord=2, axis=1)**2 - sphere_radius**2
        Ts = self.solve_quadratic_equations(As, Bs, Cs)
        ts = self.get_valid_intersection_parameters(Ts)
        return ray_start + [1,1,1]*columnify(ts) * ray_dirs

    def get_valid_intersection_parameters(
        self,
        Ts: np.ndarray
    ) -> np.ndarray:
        is_complex_or_negative = np.logical_or(np.iscomplex(Ts), Ts < 0)
        Ts_val = np.ma.masked_array(Ts, mask=is_complex_or_negative)
        return np.real(np.min(Ts_val, axis=1))

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
        cam_ray_dirs = self.convert_pixel_coordinates_to_camera_ray_directions(coords)
        return self.convert_camera_directions_to_ecef_directions(
            cam_ray_dirs, cubesat_att_in_ecef
        )

    def convert_camera_directions_to_ecef_directions(
        self,
        dirs_in_camera_frame: np.ndarray,
        cubesat_att_in_ecef: np.ndarray,
    ) -> np.ndarray:
        R_ecef_sat = self.get_rotation(cubesat_att_in_ecef)
        return (R_ecef_sat @ self.R_sat_cam @ dirs_in_camera_frame.T).T

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

    def convert_screen_coordinates_to_camera_ray_directions(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        ray_dirs_xy = -coords / self.f
        ray_dirs_z = self.f * np.ones((coords.shape[0], 1))
        return np.hstack((ray_dirs_xy, ray_dirs_z))

    def convert_pixel_coordinates_to_screen_coordinates(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        coords_ndc = (coords + 0.5) / self.dims
        return [1,-1] * (2*coords_ndc - 1)

    def convert_pixel_coordinates_to_camera_ray_directions(
            self,
            coords: np.ndarray
    ) -> np.ndarray:
        return self.convert_screen_coordinates_to_camera_ray_directions(
            self.convert_pixel_coordinates_to_screen_coordinates(coords)
        )


class MockVisionModel:
    def __init__(
        self,
        camera: Camera,
        max_correspondences: int,
        earth_radius: float,
    ) -> None:
        self.caster = RayCaster()
        self.cam = camera
        self.N = max_correspondences
        self.R = earth_radius

    def get_measurement(
        self,
        cubesat_position_in_ecef: np.ndarray,
        cubesat_attitude_in_ecef: np.ndarray
    ) -> List[PixelEcefCorrespondence]:
        #TODO: convert eci to ecef
        # Sample N pixels
        pixel_coords, pixel_coords_trunc = self.sample_pixel_coordinates()

        #TODO: add noise to pixel coordinates

        # Get ray directions for each sampled pixel
        ray_dirs = self.cam.convert_pixel_coordinates_to_ecef_ray_directions(
            pixel_coords, cubesat_attitude_in_ecef
        )

        # Get ECEF camera position as a product of several rigid body transforms
        cam_pos = self.cam.get_camera_position_in_ecef(
            cubesat_position_in_ecef, cubesat_attitude_in_ecef
        )

        # Get ECEF surface coordinates of rays
        ecef_coords = self.get_ecef_ray_and_earth_intersections(
            ray_dirs, cam_pos
        )

        # Pack valid ECEF surface coordinates into correspondences
        correspondences = []
        for i in range(self.N):
            if np.ma.is_masked(ecef_coords[i,:]):
                continue
            else:
                correspondences += [PixelEcefCorrespondence(
                    pixel_coord=pixel_coords_trunc[i,:],
                    ecef_coord=ecef_coords[i,:]
                )]
        return correspondences

    def sample_pixel_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.random.uniform(high=self.cam.dims, size=(self.N, 2))
        coords_trunc = np.int32(coords)
        return coords, coords_trunc

    def get_ecef_ray_and_earth_intersections(
        self,
        ray_dirs: np.ndarray,
        camera_pos: np.ndarray,
    ) -> np.ndarray:
        earth_pos = np.zeros(3)
        return self.caster.get_ray_and_sphere_intersections(
            ray_dirs=ray_dirs, ray_start=camera_pos,
            sphere_center=earth_pos, sphere_radius=self.R
        )

    #TODO: noise method
