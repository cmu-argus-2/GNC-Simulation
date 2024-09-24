import numpy as np

"""
    FUNCTION QUATROTATION
    Quaternion to 3D rotation matrix

    INPUTS:
        1. Quaternion q
    
    OUTPUTS:
        1. R - rotation matrix represented by q
"""


def quatrotation(q: np.ndarray):
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
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return V
