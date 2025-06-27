#include "gravity.h"

#include <iostream>
#include "SpiceUsr.h"
#include "math/EigenWrapper.h"
#include <math.h>
#include "utils_and_transforms.h"
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/GravityModel.hpp>

using namespace std;
using namespace GeographicLib;

Vector3 gravitational_acceleration(const Vector3 r, double t_J2000, int Nmax, int Mmax) 
{
    static GravityModel grav("egm96");
    static bool initialized = false;

    if (!initialized) {
        grav = GravityModel("egm96", "", Nmax, Mmax); 
        initialized = true;
    }
    
    // Get GEOD location
    Matrix_3x3 ECI2ECEF_R = ECI2ECEF(t_J2000);
    Vector3 r_ecef = ECI2ECEF_R*r;
    double gx, gy, gz;
    /*
    static Geocentric earth(Constants::WGS84_a(), Constants::WGS84_f());
    double lat, lon, h;
    vector<double> M(9, 0.0);
    // Vector3 r_geod = ECEF2GEOC(r_ecef);
    // lat = RAD_2_DEG(r_geod(1));
    // lon = RAD_2_DEG(r_geod(0));
    // h = r_geod(2); // height above the Earth's surface

    earth.Reverse(r_ecef[0], r_ecef[1], r_ecef[2], lat, lon, h, M);
    Eigen::Matrix3d M_enu2ecef;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M_enu2ecef(i, j) = M[3 * i + j];

    // eci to lat, lon, h
    grav.Gravity(lat,lon, h, gx, gy, gz);
    // gx = easterly, gy = northerly, gz = upward 
    // Vector3 grav_sez(-gy, gx, gz); 
    // Vector3 grav_ecef = SEZ2ECEF(grav_sez, DEG_2_RAD(lat), DEG_2_RAD(lon));
    // Vector3 grav_enu(gx, gy, gz); 
    // Vector3 grav_ecef = M_enu2ecef * grav_enu;
    */
    grav.V(r_ecef[0], r_ecef[1], r_ecef[2], gx, gy, gz);
    Vector3 grav_ecef(gx, gy, gz); 
    Vector3 grav_eci  = ECI2ECEF_R.transpose() * grav_ecef;
    /*
    static double prev_compute_time = -100;
    if ((t_J2000 - prev_compute_time) < 1000.00) {
        Vector3 simple_grav_eci = spherical_acceleration(r) + ECI2ECEF_R.transpose() * J2_perturbation(ECI2ECEF_R*r);
        cout << "Norm of grav_eci - spherical_acceleration: " << (grav_eci - simple_grav_eci).norm() << endl;
        cout << "r: " << r.transpose() << endl;
        cout << "grav_eci: " << grav_eci.transpose() << endl;
        cout << "spherical_acceleration: " << simple_grav_eci.transpose() << endl;
    }
    prev_compute_time = t_J2000;
    */
    // return spherical_acceleration(r) + ECI2ECEF_R.transpose() * J2_perturbation(ECI2ECEF_R*r);
    return grav_eci;
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

    double r_norm = r.norm();

    double J2_factor = 3*J2*mu*pow(R_earth, 2)/(2*pow(r_norm, 5));

    acceleration(0) = (5*pow(r(2),2)/pow(r_norm,2) - 1)*r(0)*J2_factor;
    acceleration(1) = (5*pow(r(2),2)/pow(r_norm,2) - 1)*r(1)*J2_factor;
    acceleration(2) = (5*pow(r(2), 2)/pow(r_norm, 2) - 3)*r(2)*J2_factor;

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