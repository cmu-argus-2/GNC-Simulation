//igrf14syn (isv,date,itype,alt,colat,elong,x,y,z,f)

#include "MagneticField.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "SpiceUsr.h"
#include <iostream>
#include <string>

Vector3 MagneticField(const Vector3 r, double t_J2000)
{

    static Vector3 B_eci = Vector3::Zero();
    static double prev_compute_time = -100;
    
    if (abs(t_J2000 - prev_compute_time) < 1.00) {
        return B_eci;
    } 

    // get year from seconds past J2000
    Vector5 UTCdoy = TJ2000toUTC(t_J2000);
    double year = UTCdoy(0) + UTCdoy(1)/365.25; // convert doy to fractional years

    // Get spherical location
    Vector3 r_ecef = ECI2ECEF(t_J2000)*r;
    // Vector3 r_geod = ECEF2GEOD(r_ecef);
    // double longitude = RAD_2_DEG(r_geod(0));
    // double latitude  = RAD_2_DEG(r_geod(1));
    // Vector3 r_sph = ECEF2SPH(r_ecef);
    // double longitude = r_sph(2);
    // double latitude  = M_PI/2.0 - r_sph(1);
    Vector3 r_lat = ECEF2LAT(r_ecef);
    double longitude = r_lat(1);
    double latitude  = r_lat(2);

    // Convert SEZ frame to ECEF to ECI
    Vector3 B_sez = MagneticFieldSEZ(r_lat, year);
    Vector3 B_ecef = SEZ2ECEF(B_sez, latitude, longitude);
    B_eci = ECEF2ECI(t_J2000)*B_ecef*1e-9;

    prev_compute_time = t_J2000;

    return B_eci; // account for magnetic field in (nT)   

}
// functions to be validated:
// ECI2ECEF, ECEF2GEOD, SEZ2ECEF, ECEF2ECI, MagneticFieldSEZ
// computation with geocentric coordinates is easier 
Vector3 MagneticFieldSEZ(const Vector3 r_geod, double year)
{
    // if input is in geodetic coordinates:
    // double colat = 90.0 - RAD_2_DEG(r_geod(1)); //co-latitude
    // double elong = RAD_2_DEG(r_geod(0));
    // if (elong < 0) {elong = elong + 360.0;}
    // double alt = r_geod(2)/1000.0;
    // if input is in latitudinal coordinates:
    double colat = 90.0 - RAD_2_DEG(r_geod(2)); //co-latitude
    double elong = RAD_2_DEG(r_geod(1));
    if (elong < 0) {elong = elong + 360.0;}
    double alt = r_geod(0)/1000.0;

    // get magnetic field
    int isv = 0;
    // int itype = 1; // geodetic input
    int itype = 2; // geocentric (should be quicker. also matches sez2ecef)
    double Bn, Be, Bd, Bt;
    igrf14syn_(&isv, &year, &itype, &alt, &colat, &elong, &Bn, &Be, &Bd, &Bt);

    Vector3 B_sez (-Bn, Be, -Bd);
    return B_sez;
}