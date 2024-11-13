import numpy as np

"""
    FUNCTION HAMILTONPRODUCT
    Computes the hamilton product q⊗v
    INPUTS:
        q - quaternion 1
        v - quaternion 2
    
    OUTPUTS:
        w - q⊗v
    NOTE : quaternion multiplication is not commutative
"""


def hamiltonproduct(q: np.ndarray, v: np.ndarray):
    w = np.zeros_like(q)
    w[0] = q[0] * v[0] - np.dot(q[1:], v[1:])
    w[1:] = q[0] * v[1:] + v[0] * q[1:] + np.cross(q[1:], v[1:])

    return w


"""
    FUNCTION CROSSPRODUCT
    Returns the skew-symmetric matrix representing the cross-product of a vector (a x b = Ab)
    INPUTS:
        1. v - vector to be transformed into a skew-symmetric matrix
    
    OUTPUTS;
        1. V - skew-symmetric form of V
"""


def crossproduct(v: np.ndarray):
    V = np.array([[0, -v[2], v[1]], 
                  [v[2], 0, -v[0]], 
                  [-v[1], v[0], 0]])
    return V


def rotmat2quat(R):
    q = np.zeros((4,))
    q[0] = np.sqrt(1 + np.trace(R)) / 2
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])

    return q

def q_inv(q):
    q_inv = np.zeros_like(q)
    q_inv[0] = q[0]
    q_inv[1:] = -q[1:]
    return q_inv

def Left(q):
    Q = np.zeros((4, 4))
    s = q[0]
    v = q[1:]
    Q[0, 0] = s
    Q[0, 1:] = -v
    Q[1:, 0] = v
    Q[1:, 1:] = np.eye(3) * s + crossproduct(v)
    return Q

"""
    FUNCTION QUATROTATION
    Quaternion to 3D rotation matrix
    INPUTS:
        1. Quaternion q
    
    OUTPUTS:
        1. R - rotation matrix represented by q
"""


def quatrotation(q: np.ndarray):
    T = np.diag([1, -1, -1, -1])
    H = np.vstack((np.zeros((1, 3)), np.eye(3)))
    return H.T @ T @ Left(q) @ T @ Left(q) @ H
    """
    R = np.zeros((3, 3))
    R[0, 0] = 2 * (q[0] ** 2 + q[1] ** 2) - 1
    R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[1, 1] = 2 * (q[0] ** 2 + q[2] ** 2) - 1
    R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[2, 2] = 2 * (q[0] ** 2 + q[3] ** 2) - 1
    
    return R
    """

def Gquat(q):
    H = np.vstack((np.zeros((1, 3)), np.eye(3)))
    return Left(q) @ H


def quat_from_two_vectors(v1: np.ndarray, v2: np.ndarray):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    z = v1
    x = np.cross(v1, v2)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    R = np.vstack((x, y, z)).T
    return rotmat2quat(R)


def quatconj(q: np.ndarray):
    q_conj = np.zeros_like(q)
    q_conj[0] = q[0]
    q_conj[1:] = -q[1:]
    return q_conj

def quat_to_axis_angle(q: np.ndarray):
    if q[0] > 1:  # normalize if needed
        q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0]**2)
    if s < 1e-8:  # to avoid division by zero
        axis = q[1:]
    else:
        axis = q[1:] / s
    return axis, angle