//igrf13syn (isv,date,itype,alt,colat,elong,x,y,z,f)

#include "MagneticField.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "SpiceUsr.h"
#include <iostream>
#include <string>

Vector3 MagneticField(Vector3 r, double t_J2000)
{
    // get year from seconds past J2000
    Vector5 UTCdoy = TJ2000toUTC(t_J2000);
    double year = UTCdoy(0) + UTCdoy(1)/365.25; // convert doy to fractional years

    // Get GEOD location
    Vector3 r_geod = ECEF2GEOD(ECI2ECEF(t_J2000)*r);
    
    // Convert SEZ frame to ECEF to ECI
    Vector3 B_sez = MagneticFieldSEZ(r_geod, year);
    
    Vector3 B_ecef = SEZ2ECEF(B_sez, r_geod(1), r_geod(0));

    Vector3 B_eci = ECEF2ECI(t_J2000)*B_ecef;

    return B_eci*1e-9; // account for magnetic field in (nT)   

}

Vector3 MagneticFieldSEZ(Vector3 r_geod, double year)
{
    double colat = 90.0 - r_geod(1)*180.0/M_PI; //co-latitude
    double elong = r_geod(0)*180.0/M_PI;
    if (elong < 0) {elong = elong + 360.0;}

    // get magnetic field
    int isv = 0;
    int itype = 1;
    double Bn, Be, Bd, Bt;
    double alt = r_geod(2)/1000.0; 
    igrf13syn_(&isv, &year, &itype, &alt, &colat, &elong, &Bn, &Be, &Bd, &Bt);
    
    Vector3 B_sez (-Bn, Be, -Bd);
    return B_sez;
}