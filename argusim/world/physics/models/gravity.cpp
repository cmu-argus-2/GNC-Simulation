#include "gravity.h"

#include <iostream>
#include "math/EigenWrapper.h"
#include <math.h>


Vector3 gravitational_acceleration(const Vector3 r) 
{
    return spherical_acceleration(r) + J2_perturbation(r);
}

Vector3 gravity_gradient_torque(const Vector3& r, const Matrix_3x3& I_sat) {
    Vector3 r_hat = r.normalized();
    Vector3 I_r_hat = I_sat * r_hat;
    Vector3 tau_gg = 3 * mu / pow(r.norm(), 3) * r_hat.cross(I_r_hat);
    return tau_gg;
}

Vector3 spherical_acceleration(const Vector3 r)
{
    Vector3 acceleration = - (mu/pow(r.norm(), 3.0))*r;

    return acceleration;
}

Vector3 J2_perturbation(const Vector3 r)
{
    Vector3 acceleration;

    double J2_factor = 3*J2*mu*pow(R_earth, 2)/(2*pow(r.norm(), 5));

    acceleration(0) = (5*pow(r(2),2)/pow(r.norm(),2) - 1)*r(0)*J2_factor;
    acceleration(1) = (5*pow(r(2),2)/pow(r.norm(),2) - 1)*r(1)*J2_factor;
    acceleration(2) = (5*pow(r(2), 2)/pow(r.norm(), 2) - 3)*r(2)*J2_factor;

    return acceleration;
}

Vector3 sun_gravity(const Vector3 r, const Vector3 sun_position) 
{
    // Calculate the distance from the spacecraft to the Sun
    Vector3 r_sun = sun_position - r;
    double r_sun_norm = r_sun.norm();
    double sun_pos_norm = sun_position.norm();

    // Calculate the gravitational acceleration due to the Sun
    Vector3 acceleration = (mu_sun / pow(r_sun_norm, 3.0)) * r_sun;
    acceleration = acceleration - (mu_sun / pow(sun_pos_norm, 3.0)) * sun_position;

    return acceleration;
}

Vector3 moon_gravity(const Vector3 r, double t_J2000) 
{
    // Get the Moon's position at the given time
    Vector3 moon_position = moon_position_eci(t_J2000);
    // Calculate the distance from the spacecraft to the Moon
    Vector3 r_moon = moon_position - r;
    double r_moon_norm = r_moon.norm();
    double moon_pos_norm = moon_position.norm();

    // Calculate the gravitational acceleration due to the Moon
    Vector3 acceleration = (mu_moon / pow(r_moon_norm, 3.0)) * r_moon;
    acceleration = acceleration - (mu_moon / pow(moon_pos_norm, 3.0)) * moon_position;

    return acceleration;
}


Vector3 moon_position_eci(double t_J2000) {

    //Load all kernels
    loadAllKernels();

    SpiceDouble state[3];
    SpiceDouble lt;

    spkpos_c("moon", t_J2000, "J2000", "NONE", "earth", state, &lt);
    Vector3 moon_pos(1000.0 * state[0], 1000.0 * state[1],
                    1000.0 * state[2]);   // convert km to m and cast SpiceDouble into Vector3
    return moon_pos;
}