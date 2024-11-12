import numpy as np


def distances_to_plane(points, plane_coeffs):
    n, d = plane_coeffs[:3], plane_coeffs[3]
    distances = np.abs(points @ n + d) / np.linalg.norm(n)
    return distances


def fit_plane(points):
    # points: Nx3 matrix of (x, y, z) triples
    # Returns: ([a b c d] representing ax + by + cz + d = 0, inlier_idxs)
        
    N, M = points.shape
    assert N >= 3  # need at least 3 points to fit a plane in R^3
    assert M == 3

    N = len(points)
    A = np.hstack((points, np.ones((N, 1))))

    U, s, Vt = np.linalg.svd(A)

    return Vt[-1, :]  # last row of Vt yields the minimum norm solution


def fit_plane_RANSAC(
    points,
    tolerance=0.1,
    ITERS=100,
):
    # points: Nx3 numpy array of (x, y, z) points
    # Returns: [a b c d] that describe the plane ax + by + cz + d = 0

    N, M = points.shape
    assert N >= 3  # need at least 3 points to fit a plane in R^3
    assert M == 3

    sample_size = 3  # only need 3 points to fit a plane

    max_inliers = 0
    largest_inlier_idxs_set = None
    for i in range(ITERS):
        sampled_idxs = np.random.choice(range(N), sample_size)
        plane_coeffs = fit_plane(points[sampled_idxs])

        distance_to_plane = distances_to_plane(points, plane_coeffs)
        inlier_idxs = np.where(distance_to_plane < tolerance)[0]

        N_inliers = len(inlier_idxs)
        if N_inliers > max_inliers:
            max_inliers = N_inliers
            largest_inlier_idxs_set = inlier_idxs

    final_coeffs = fit_plane(points[largest_inlier_idxs_set])

    distance_to_plane = distances_to_plane(points, final_coeffs)
    assert len(distance_to_plane) == N
    final_inlier_idxs = np.where(distance_to_plane < tolerance)[0]
    return final_coeffs, final_inlier_idxs
