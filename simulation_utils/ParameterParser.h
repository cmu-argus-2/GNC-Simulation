#ifndef _SIMULATOR_PARAMETER_PARSER_
#define _SIMULATOR_PARAMETER_PARSER_

#include <string>

#include "math/EigenWrapper.h"
#include "Magnetorquer.h"

class Simulation_Parameters {
   public:
    Simulation_Parameters(std::string filename);
    //void getParamsFromFileAndSample(std::string filename);
    VectorXd initializeSatellite(double epoch);

    // void dumpSampledParametersToYAML(std::string absolute_filename); TODO

    // ========================================================================
    // ================== MAIN PARAMETERS (parsed from file) ==================
    // ========================================================================    

    /* Satellite Physical Properties */
    // physical properties
    double mass; // [Kg]
    Matrix_3x3 I_sat; // [kg.m^2]
    double A; // [m^2] face area of each facet

    // Drag and SRP properties
    double Cd; // [unitless]
    double CR; // [unitless]

    // Reaction Wheels
    int num_RWs;
    MatrixXd G_rw_b; // Matrix whose columns are the axes of each RW in the body frame
    double I_rw; // [Kg.m^2] inertia of each RW about its rotary axis

    // Magnetorquers
    int num_MTBs;
    MatrixXd G_mtb_b; // Matrix whose columns are the field axes of each MTB in the body frame
    double max_voltage;
    double coils_per_layer;
    double layers;
    double trace_width;
    double gap_width;
    double trace_thickness;
    double max_current_rating;
    double max_power;
    Magnetorquer MTB; // Magnetorquer class object
    
    /* Simulation Settings */ 
    double MAX_TIME;                   // [s]
    double dt;                         // [s]
    double earliest_sim_start_unix;    // [s]
    double latest_sim_start_unix;      // [s]
    bool useDrag; // set to False to deactivate drag calcs
    bool useSRP; // set to False to deactivate SRP calcs

    /* Satellite Initialization */
    double semimajor_axis; // [m]
    double eccentricity; // [unitless]
    double inclination; // [deg]
    double RAAN; // [deg]
    double AOP; // [deg]
    double true_anomaly; // [deg]
    Vector4 initial_attitude; 
    Vector3 initial_angular_rate; // [rad/s]
    VectorXd initial_state;

    private:
    Magnetorquer load_MTB(std::string filename);
};

#endif