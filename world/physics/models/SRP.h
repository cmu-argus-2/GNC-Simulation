#ifndef C___SRP_H
#define C___SRP_H

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"

class SRP {
   public:
    /**
     * @brief Construct an SRP Model
     *
     * @param ephemeris_data_filepath - Absolute path to binary spice ephemeris data
     */
    SRP(std::string ephemeris_data_filepath);

    /**
     * @brief Compute sun position in J2000 ECI frame
     *
     * @param t_J2000 : seconds since the J2000 epoch
     * @return Sun position vector in ECI frame
     */
    static Vector3 sun_position_eci(double t_J2000);
};

#endif   // C___SRP_H