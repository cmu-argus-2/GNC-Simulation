#include "SRP.h"

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"

// TODO : Compute shadow function
Vector3 SRP_acceleration(const Quaternion q, double t_J2000, double CR, double A, double m)
{
    // Constants
    double solar_constant = 1367; // W/m^2
    double c = 299792458; // light speed m/s

    // Get sun position
    Vector3 r_sun = sun_position_eci(t_J2000);

    // Frontal Area
    double A_f = A*FrontalAreaFactor(q, r_sun);

    //Drag acceleration
    Vector3 acceleration;
    acceleration = (CR*(solar_constant/c)*A_f/(r_sun.norm()*m))*r_sun;

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