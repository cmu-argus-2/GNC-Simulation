#include <unistd.h>

#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>

using namespace Eigen;

/* Interesting Initial Parameters:
        x_init = 0 y_init = 0 theta_init = 45 vel_init = 11.0875e3 dt = 10 //eliptic orbit around earth and moon
        x_init = 0 y_init = 0 theta_init = 47 vel_init = 11.08136e3 dt = 10 //leave earth, go extremely close to the
   moon, then come back close to earth
*/

double G = 6.67430e-11;   // Gravitational constant

double R_e     = 6.3781e6;    // Earth's radius [m]
double x_earth = 0.0;         // x position of the earth's center [m]
double y_earth = -R_e;        // y position of the earth's center [m]
double m_earth = 5.9722e24;   // mass of the earth [kg]

double R_m    = 1.7371e6;   // Moon's radius [m]
double x_moon = 384.4e6;    // x position of the moon's center [m]
double y_moon = -R_e;       // y position of the moon's center [m]
double m_moon = 7.348e22;   // mass of the moon [kg]

double x_init     = 0;            // initial x position of the satellite [m]
double y_init     = 0;            // initial y position of the satellite [m]
double theta_init = 47;           // initial angle of the satellite's velocity vector relative to the x-axis [deg]
double vel_init   = 11.08136e3;   // magnitude of the satellite's initial velocity [m/s]
double x_vel_init = vel_init * cos(theta_init * M_PI / 180);   // x component of the satellite's initial velocity [m/s]
double y_vel_init = vel_init * sin(theta_init * M_PI / 180);   // y component of the satellite's initial velocity [m/s]

std::vector<double> x_pos_arr;   // array that holds the satellite's x position at various time-steps [m]
std::vector<double> y_pos_arr;   // array that holds the satellite's y position at various time-steps [m]

// calculating the time-derivative of the satellite's state vector, x: [x_position, x_velocity, y_position, y_velocity],
// given its current state vector and control inputs, u
Vector4d f(VectorXd x, double u) {
    Vector2d earth_pos;
    earth_pos << x_earth, y_earth;
    Vector2d moon_pos;
    moon_pos << x_moon, y_moon;
    Vector2d sat_pos;
    sat_pos << x(0), x(2);

    Vector2d pos_sat_to_earth = earth_pos - sat_pos;
    Vector2d pos_sat_to_moon  = moon_pos - sat_pos;

    Vector2d a = G * (((m_earth / pos_sat_to_earth.squaredNorm()) * pos_sat_to_earth.normalized()) +
                      ((m_moon / pos_sat_to_moon.squaredNorm()) *
                       pos_sat_to_moon.normalized()));   // satellite's acceleration vector due to gravity [m/s^2]

    Vector4d xdot;   // time derivative of the state vector, x
    xdot << x(1), a(0), x(3), a(1);
    return xdot;
}

// 4th order runge-kutta numerical integration

VectorXd rk4(const VectorXd& x, double u, double dt) {
    double half_dt = dt * 0.5;
    Vector4d k1    = f(x, u);
    Vector4d k2    = f(x + half_dt * k1, u);
    Vector4d k3    = f(x + half_dt * k2, u);
    Vector4d k4    = f(x + dt * k3, u);
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

double dt       = 10;      // simulation time-step [s]
double currTime = 0.0;     // simulation's start time [s]
double maxTime  = 750e3;   // maximum time to simulate [s]

int main() {
    Vector4d x;
    x << x_init, x_vel_init, y_init, y_vel_init;   // initializing the satellite's state vector

    auto start = std::chrono::steady_clock::now();
    while (currTime <= maxTime &&
           (x(0) - x_earth) * (x(0) - x_earth) + (x(2) - y_earth) * (x(2) - y_earth) >= R_e * R_e) {
        x_pos_arr.push_back(x(0));
        y_pos_arr.push_back(x(2));
        x = rk4(x, 0, dt);
        currTime += dt;
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time to Simulate: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    std::ofstream logFile;
    logFile.open("SatelliteAroundEarthAndMoon.txt");
    auto N = x_pos_arr.size();
    for (long i = 0; i < N; i++) {
        logFile << x_pos_arr[i] << "\t" << y_pos_arr[i] << "\n";
    }
    logFile.close();
    return 0;
}