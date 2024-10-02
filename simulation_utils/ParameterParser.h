#ifndef _SIMULATOR_PARAMETER_PARSER_
#define _SIMULATOR_PARAMETER_PARSER_

#include <string>

#include "math/EigenWrapper.h"

class Simulation_Parameters {
   public:
    void getParamsFromFileAndSample(const std::string& filename);

    void dumpSampledParametersToYAML(const std::string& absolute_filename);

    // ========================================================================
    // ================== MAIN PARAMETERS (parsed from file) ==================
    // ========================================================================
    double MAX_TIME;                  // [s]
    double dt;                        // [s]
    double RAAN;                      // [rad]
    double init_altitute;             // [m]
    double orbital_velocity;          // [m/s]
    double orbital_incliniation;      // [rad]
    double satellite_mass;            // [kg]
    Matrix_3x3 InertiaTensor;         // [kg*m^2]
    double earliest_sim_start_unix;   // [s]
    double latest_sim_start_unix;     // [s]
    double EARTH_RADIUS;              // [m]

    Vector3 orbital_plane_normal;
};
#endif