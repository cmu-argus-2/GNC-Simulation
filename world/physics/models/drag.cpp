#include "drag.h"

#include <math.h>
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "SRP.h"


Vector3 drag_acceleration(const Vector3 r, const Vector3 v, const Quaternion q, double t_J2000, double Cd, double A, double m)
{
    // Get density
    double rho = density(r, t_J2000);

    // Frontal Area
    double A_f = A*FrontalAreaFactor(q, v);

    //Drag acceleration
    Vector3 acceleration;
    acceleration = (0.5*Cd*rho*A_f*v.norm()/m)*v;

    return acceleration;
}

double density(const Vector3 r, double t_J2000) {
    // Load all kernels
    loadAllKernels();

    /* Harris Priester data*/
    double HP_alt_MIN = 100.0; // Lower height limit [km]
    double HP_alt_MAX = 1000.0; // Upper height limit [km]
    double RA_lag = 0.523599; // Right ascension lag [rad]
    double HP_prm = 3; // Harris-Priester parameter

    Eigen::Matrix<double, 50, 1> Altitudes {100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,     
                210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,     
                320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,     
                520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,     
                720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0,1000.0}; // in km

    Eigen::Matrix<double, 50, 1> rho_MIN {4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,         
                8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,         
                9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,         
                2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,         
                2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,         
                2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,         
                4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,         
                1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,         
                1.560e-03, 1.150e-03}; // Minimum density [gm/km^3]

    Eigen::Matrix<double, 50, 1> rho_MAX {4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,         
                8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,         
                1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,         
                4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,         
                7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,         
                1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,         
                4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,         
                1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,         
                2.360e-02, 1.810e-02}; // Maximum density [gm/km^3]
    /* END Harris proester data*/

    // Get sun position 
    Vector3 r_sun = sun_position_eci(t_J2000);

    // Convert ECI positions to GOED
    Vector3 r_ecef = ECI2ECEF(t_J2000)*r;
    Vector3 r_geod = ECEF2GEOD(r_ecef);

    // Search for density values
    double alt = r_geod(2)/1.0e3; // altitude in km

    if (alt < HP_alt_MIN || alt > HP_alt_MAX) { return 0.0;} // Outside Harris Priester range

    int ih = 0; // Section index reset
    for (int i=0; i < 50 - 1; i++) // Loop over 50 altitude regimes
    {
        if( alt >= Altitudes(i) && alt < Altitudes(i+1) ) 
        {
            ih = i; // ih identifies altitude section
            break;
        }
    }
    double alt_MIN = ( Altitudes(ih) - Altitudes(ih+1) )/log( rho_MIN(ih+1)/rho_MIN(ih) );
    double alt_MAX = ( Altitudes(ih) - Altitudes(ih+1) )/log( rho_MAX(ih+1)/rho_MAX(ih) );

    double d_MIN = rho_MIN(ih)*exp( (Altitudes(ih) - alt)/alt_MIN );
    double d_MAX = rho_MAX(ih)*exp( (Altitudes(ih) - alt)/alt_MAX );


    // Compute Solar flux related modification
    // Compute Right ascension and declination of the sun
    double RA = atan2(r_sun(1), r_sun(0));
    if (RA < 0) {RA += 2*M_PI;} // normalize RA to [0, 2pi] range
    double DEC = asin(r_sun[2]/r_sun.norm());

    double c_dec = cos(DEC);
    Vector3 bulge_u (c_dec*cos(RA + RA_lag), c_dec*sin(RA + RA_lag), sin(DEC));
    double c_psi2 = 0.5 + 0.5*r.dot(bulge_u)/r.norm();

    // Density computation
    double rho = ( d_MIN + (d_MAX - d_MIN)*pow(c_psi2,HP_prm) )*1.0e-12; // [kg/m3]

    return rho;
}