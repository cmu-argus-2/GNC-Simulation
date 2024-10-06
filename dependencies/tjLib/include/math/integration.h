#ifndef _TJLIB_INTEGRATION_
#define _TJLIB_INTEGRATION_

#include <functional>

template <typename Tx, typename Tu>
Tx rk4(Tx x, Tu u, std::function<Tx(Tx, Tu)> f, double dt) {
    /*Fourth order Runge-Kutta integration.
    Keyword arguments:
    x -- states
    u -- inputs (constant for dt)
    dt -- time over which to integrate
    */
    double half_dt = dt * 0.5;
    Tx k1          = f(x, u);
    Tx k2          = f(x + half_dt * k1, u);
    Tx k3          = f(x + half_dt * k2, u);
    Tx k4          = f(x + dt * k3, u);
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

#endif