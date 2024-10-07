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
