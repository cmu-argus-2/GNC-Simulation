#ifndef C___SRP_H
#define C___SRP_H

#include "math/EigenWrapper.h"
#include "SpiceUsr.h"

class SRP {
   public:
    /**
     * @brief Construct an SRP Model
     * 
     * @param ephemeris_data - Absolute path to binary spice ephemeris data
     */
    SRP(ConstSpiceChar *ephemeris_data);

    /**
     * @brief Compute sun position in J2000 ECI frame
     * 
     * @param mjd : Modified Julian date
     * @return Sun position vector in ECI frame
     */
    Vector3 sun_position_eci(double mjd);
};

#endif   // C___SRP_H