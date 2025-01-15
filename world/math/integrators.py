import numpy as np


def RK4(x, u, f, solver_timestep):
    """
    Runge-Kutta 4th Order integration
    Accepts the state transition function and current state to compute the next state

    INPUTS:
        1. x - current state vector
        2. u - current input
        3. f - state transition function xdot = f(x,u)
        4. sim_params - instance of class SimSetup and contains timestepping parameters
    """
    h = solver_timestep

    # Deviations
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)

    # x_(n+1)
    x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def RK4_jac(x, u, f, f_jac, dt):
    """
    Computes the Jacobian of the state transition function using RK4 integration.

    :param x: The current state vector.
    :param u: The current input vector.
    :param f: The state transition function. Takes in the current state and input and returns the state derivative.
    :param f_jac: The Jacobian of the state transition function. Takes in the current state and input and returns the
                  Jacobian of the state derivative.
    :param dt: The timestep.
    :return: The Jacobian of the state transition function.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)

    k1_jac = f_jac(x, u)
    k2_jac = f_jac(x + 0.5 * dt * k1, u) @ (np.eye(6) + 0.5 * dt * k1_jac)
    k3_jac = f_jac(x + 0.5 * dt * k2, u) @ (np.eye(6) + 0.5 * dt * k2_jac)
    k4_jac = f_jac(x + dt * k3, u) @ (np.eye(6) + dt * k3_jac)

    return np.eye(6) + (dt / 6) * (k1_jac + 2 * k2_jac + 2 * k3_jac + k4_jac)
