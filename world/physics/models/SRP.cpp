#include "SRP.h"
#include <iostream>

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"


const double R_EARTH = 6371000.0; // Earth radius in meters
const double R_SUN = 696340000.0; // Sun radius in meters

double partial_illumination_rel(const Vector3& r_earth, const Vector3& r_sun) {
    double r_mag = r_earth.norm();
    double R_sun = R_SUN;
    double R_earth = R_EARTH;
    double dmag = r_sun.norm();
    double sd = r_earth.dot(r_sun);
    double a = asin(R_sun / dmag);
    if (R_earth > r_mag) {
        std::cerr << "Error! Collision detected with Earth." << std::endl;
        return 0.0;
    }
    double b = asin(R_earth / r_mag);
    double c = acos(sd / (r_mag * dmag));
    if ((a + b) <= c) { 
        return 1.0;
    } else if (c < (b - a)) { 
        return 0.0;
    } else {
        double x = (c * c + a * a - b * b) / (2 * c);
        double y = sqrt(a * a - x * x);
        double A = a * a * acos(x / a) + b * b * acos((c - x) / b) - c * y;
        double nu = 1 - A / (M_PI * a * a);
        return nu;
    }
}

double partial_illumination(const Vector3& r, const Vector3& r_Sun) {
    Vector3 d = r - r_Sun;
    return partial_illumination_rel(r, d);
}


// TODO : Compute shadow function
Vector3 SRP_acceleration(const Quaternion q, const Vector3 r, double t_J2000, double CR, double A, double m)
{
    // Constants
    double solar_constant = 1367; // W/m^2
    double c = 299792458; // light speed m/s

    // Get sun position
    Vector3 r_sun = sun_position_eci(t_J2000);

    // Frontal Area
    double A_f = A*FrontalAreaFactor(q, r_sun);

    // Shadow factor
    double shadow_factor = partial_illumination_rel(r, r_sun);

    //Drag acceleration
    Vector3 acceleration;
    acceleration = (shadow_factor*CR*(solar_constant/c)*A_f/(r_sun.norm()*m))*r_sun;

    return acceleration;
}

double FrontalAreaFactor(const Quaternion q, const Vector3 r)
{
    Matrix_3x3 R_q = q.normalized().toRotationMatrix();
    double projection_factor = (r.transpose()*R_q).sum()/r.norm();

    return projection_factor;
}

Vector3 sun_position_eci(double t_J2000) {

    //Load all kernels
    loadAllKernels();

    SpiceDouble state[3];
    SpiceDouble lt;

    spkpos_c("sun", t_J2000, "J2000", "NONE", "earth", state, &lt);
    Vector3 sun_pos(1000.0 * state[0], 1000.0 * state[1],
                    1000.0 * state[2]);   // convert km to m and cast SpiceDouble into Vector3
    return sun_pos;
}