import numpy as np
from brahe.constants import GM_EARTH
from world.math.integrators import RK4, RK4_jac


"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
"""


def state_derivative(x: np.ndarray, u: None) -> np.ndarray:
    """
    The state derivative function x_dot = f(x, u) for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param u: The current control input (not used, but needed for RK4).
    :return: A numpy array of shape (6,) containing the state derivative.
    """
    r = x[:3]
    v = x[3:]
    a = -r * GM_EARTH / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


def state_derivative_jac(x: np.ndarray, u: None) -> np.ndarray:
    """
    The state derivative Jacobian function df/dx, where x_dot = f(x, u) for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param u: The current control input (not used, but needed for RK4).
    :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
    """
    r = x[:3]
    r_norm = np.linalg.norm(r)
    da_dr = (-GM_EARTH / r_norm ** 3) * np.eye(3) + (3 * GM_EARTH / r_norm ** 5) * np.outer(r, r)
    return np.block([[np.zeros((3, 3)), np.eye(3)],
                     [da_dr, np.zeros((3, 3))]])


def f(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The state transition function x_{t+1} = f_d(x_t).

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6,) containing the next state (position and velocity).
    """
    return RK4(x, None, state_derivative, dt)


def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The state transition Jacobian function df_d/dx.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
    """
    return RK4_jac(x, None, state_derivative, state_derivative_jac, dt)
