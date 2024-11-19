from .PlaneFitting import fit_plane_RANSAC
import numpy as np


def _compute_body_ang_vel_from_sun_rays(init_sun_rays):
    # Find the satellite rotaiton axis as the unit vector with a constant dot product with all measured sun vectors.
    # This assumes:
    #     the satellite has a constant angular velocity
    #     the sat has rotated less than 180 degrees between every other sun ray measurement
    sun_rays = np.array([sun_ray for (t, sun_ray) in init_sun_rays])
    plane, inlier_idxs = fit_plane_RANSAC(sun_rays, tolerance=3e-2)  # TODO tune tolerance
    # print(sum(inlier_idxs != 0), len(init_sun_rays))

    # TODO ensure we don't div-by-0
    plane_normal = plane[0:3] / np.linalg.norm(plane[0:3])

    # TODO dont accept the plane if the sun rays are bunched up (the differences between measurements will get swamped out by noise)
    rotation_axis = plane_normal
    # print(f"{len(sun_rays)} plane_normal: ", plane_normal * np.sign(plane_normal[2]))

    # Choose sign for rotation axis that is consistent with the sun ray measurement history
    N = len(init_sun_rays)
    N_consistent_with_axis = 0
    N_consistent_with_opposite_axis = 0
    for i in range(1, N - 1):
        _, sun_ray0 = init_sun_rays[i - 1]
        _, sun_ray1 = init_sun_rays[i]
        _, sun_ray2 = init_sun_rays[i + 1]

        delta_ray10 = sun_ray1 - sun_ray0
        delta_ray21 = sun_ray2 - sun_ray1

        # negative cross product because sat's rotation is opposite the apparent motion of the sun relative to the sat
        rough_axis_estimate = -np.cross(delta_ray10, delta_ray21)
        rough_axis_estimate /= np.linalg.norm(rough_axis_estimate)
        similarity = np.dot(rough_axis_estimate, rotation_axis)

        # print(similarity)
        if similarity > 0.8:
            N_consistent_with_axis += 1
        elif similarity < -0.8:
            N_consistent_with_opposite_axis += 1

    min_consistent = 0.5 * N
    if N_consistent_with_opposite_axis >= min_consistent:
        rotation_axis *= -1
    elif N_consistent_with_axis < min_consistent:
        print(f"Couldn't find enough consistent data {N_consistent_with_axis} {N_consistent_with_opposite_axis}")
        print(sun_rays)
        return None, None  # couldn't find enough consistent data
    # print("rotation_axis:", rotation_axis)

    # Compute the angular velocity magnitude
    omega_norm_estimates = []
    for i in range(N - 1):
        t0, sun_ray0 = init_sun_rays[i]
        t1, sun_ray1 = init_sun_rays[i + 1]

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

    return rotation_axis * omega_norm_estimate, inlier_idxs


def compute_body_ang_vel_from_sun_rays(init_sun_rays):
    # Find the satellite rotaiton axis as the unit vector with a constant dot product with all measured sun vectors.
    # This assumes:
    #     the satellite has a constant angular velocity
    #     the sat has rotated less than 180 degrees between every other sun ray measurement

    # Estimate the body angular velocity
    estimated_omega_in_body_frame, inlier_rays_idxs = _compute_body_ang_vel_from_sun_rays(init_sun_rays)

    # Estimate the covariance of the estimated body angular velocity via jacknife resamping
    if estimated_omega_in_body_frame is not None:
        inlier_sunrays = [init_sun_rays[idx] for idx in inlier_rays_idxs]

        estimates = []
        for i in range(len(inlier_sunrays)):
            all_but_one_sunray = [ray for (idx, ray) in enumerate(inlier_sunrays) if idx != i]
            estimate, _ = _compute_body_ang_vel_from_sun_rays(all_but_one_sunray)
            if estimate is not None:
                estimates.append(estimate)

        if len(estimates) > 1:
            estimates = np.array(estimates)
            np.set_printoptions(linewidth=1000, suppress=True)
            # print("estimates [deg/s]:\n", np.rad2deg(estimates))

            bessel_correction = len(estimates) / (len(estimates) - 1)
            cov = bessel_correction * np.cov(estimates.T)
            return estimated_omega_in_body_frame, cov

    return None
