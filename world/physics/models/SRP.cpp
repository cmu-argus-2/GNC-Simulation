#include "SRP.h"

#include "math/EigenWrapper.h"
#include "SpiceUsr.h"

// provide absolute path to ephemeris data file located at PROJECT_ROOT/data/de440.bsp
SRP::SRP(ConstSpiceChar *ephemeris_data) {
    furnsh_c(ephemeris_data);
}

Vector3 SRP::sun_position_eci(double MJD) {
    double ephem_date = (MJD + 2400000.5 - 2451545.000000)/(60.0*60.0*24.0); // SPICE takes epoch as seconds past J2000 i.e., seconds past noon Jan 1st 2000

    SpiceDouble state[6] ;
    SpiceDouble lt;

    spkezr_c("sun", ephem_date, "J2000", "NONE", "earth", state, &lt);
    Vector3 sun_pos (1000.0*state[0], 1000.0*state[1], 1000.0*state[2]); // convert km to m and cast SpiceDouble into Vector3
    return sun_pos;
}
