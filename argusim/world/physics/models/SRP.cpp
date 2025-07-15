#include "SRP.h"
#include <iostream>

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"

const double R_EARTH = 6378000.0; // Earth radius in meters

double shadow_factor(const Vector3 r_sat, const Vector3 r_sun) 
{
    double r_mag = r_sat.norm();
    if (R_EARTH > r_mag) {
        std::cerr << "Error! Collision detected with Earth." << std::endl;
        return 0.0;
    }
    double sd = r_sat.dot(r_sun);

    if (sd >= 0.0) { 
        return 1.0;
    }
    
    double dmag = r_sun.norm();

    Vector3 r_sat_parallel = r_sun * sd / (dmag * dmag);
    Vector3 r_sat_perpendicular = r_sat - r_sat_parallel;

    double r_sat_perpendicular_mag = r_sat_perpendicular.norm();
    if (r_sat_perpendicular_mag < R_EARTH) { 
        return 0.0;
    }
    return 1.0;
}

Vector3 SRP_acceleration(const Vector3 r, const Quaternion q, double t_J2000, double CR, double A, double m)
{
    // Constants
    double solar_constant = 1367; // W/m^2
    double c = 299792458; // light speed m/s

    // Get sun position
    Vector3 r_sun = sun_position_eci(t_J2000);

    // Shadow Factor
    double shadow = shadow_factor(r, r_sun);

    // Frontal Area
    double A_f = A*FrontalAreaFactor(q, r_sun);

    //Drag acceleration
    Vector3 acceleration;
    acceleration = shadow*(CR*(solar_constant/c)*A_f/(r_sun.norm()*m))*r_sun;

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