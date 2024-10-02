#include <unistd.h>

#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>

using namespace Eigen;

double G = 6.67430e-11;   // Gravitational constant

double R_e     = 6.3781e6;    // Earth's radius [m]
double x_earth = 0.0;         // x position of the earth's center [m]
double y_earth = -R_e;        // y position of the earth's center [m]
double m_earth = 5.9722e24;   // mass of the earth [kg]

double x_init     = 0;         // initial x position of the satellite [m]
double y_init     = 0;         // initial y position of the satellite [m]
double theta_init = 45;        // initial angle of the satellite's velocity vector relative to the x-axis [deg]
double vel_init   = 11.15e3;   // magnitude of the satellite's initial velocity [m/s]
double x_vel_init = vel_init * cos(theta_init * M_PI / 180);   // x component of the satellite's initial velocity [m/s]
double y_vel_init = vel_init * sin(theta_init * M_PI / 180);   // y component of the satellite's initial velocity [m/s]

std::vector<double> x_pos_arr;   // array that holds the satellite's x position at various time-steps [m]
std::vector<double> y_pos_arr;   // array that holds the satellite's y position at various time-steps [m]

// calculating the time-derivative of the satellite's state vector, x: [x_position, x_velocity, y_position, y_velocity],
// given its current state vector and control inputs, u
Vector4d f(VectorXd x) {
    double r_x = x_earth - x(0);   // x component of the position vector from the satellite TO the earth's center [m]
    double r_y = y_earth - x(2);   // y component of the position vector from the satellite TO the earth's center [m]

    // magnitude of the satellite's acceleration due to gravity [m/s^2]
    double a = G * m_earth / (r_x * r_x + r_y * r_y);

    // magnitude of the position vector from the satellite to the earth [m]
    double r_norm = sqrt(r_x * r_x + r_y * r_y);

    Vector4d xdot;   // time derivative of the state vector, x
    xdot << x(1), a * r_x / r_norm, x(3), a * r_y / r_norm;
    return xdot;
}

// 4th order runge-kutta numerical integration
Vector4d rk4(const VectorXd& x, double dt) {
    double half_dt = dt * 0.5;
    Vector4d k1    = f(x);
    Vector4d k2    = f(x + half_dt * k1);
    Vector4d k3    = f(x + half_dt * k2);
    Vector4d k4    = f(x + dt * k3);
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

double dt       = 10;    // simulation time-step [s]
double currTime = 0.0;   // simulation's start time [s]
double maxTime  = 1e7;   // maximum time to simulate [s]

int main() {
    Vector4d x;
    x << x_init, x_vel_init, y_init, y_vel_init;   // initializing the satellite's state vector

    auto start = std::chrono::steady_clock::now();
    while (currTime <= maxTime &&
           (x(0) - x_earth) * (x(0) - x_earth) + (x(2) - y_earth) * (x(2) - y_earth) >= R_e * R_e) {
        x = rk4(x, dt);
        x_pos_arr.push_back(x(0));
        y_pos_arr.push_back(x(2));
        currTime += dt;
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time to Simulate: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    std::ofstream logFile;
    logFile.open("SatelliteAroundEarth.txt");
    auto N = x_pos_arr.size();
    for (uint64_t i = 0; i < N; i++) {
        logFile << x_pos_arr[i] << "\t" << y_pos_arr[i] << "\n";
    }
    logFile.close();
    return 0;
}