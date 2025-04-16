#ifndef _SIMULATOR_PARAMETER_PARSER_
#define _SIMULATOR_PARAMETER_PARSER_

#include <string>
#include <random>

#include "math/EigenWrapper.h"
#include "Magnetorquer.h"
#include "yaml-cpp/yaml.h"

class Simulation_Parameters {
   public:
    Simulation_Parameters(std::string filename, int trial_number, std::string results_folder);
    VectorXd initializeSatellite(double epoch);
    void dumpSampledParametersToYAML(std::string results_folder);

    std::mt19937 dev;
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
    VectorXd resistances;
    MatrixXd G_mtb_b; // Matrix whose columns are the field axes of each MTB in the body frame
    double max_voltage;
    double max_current_rating;
    double max_power;
    Magnetorquer MTB; // Magnetorquer class object

    /* Sensors */
    // GPS
    double gps_pos_std;
    double gps_vel_std;

    // Sun Sensors
    int num_photodiodes;
    MatrixXd G_pd_b; // orientation matrix for photodiodes
    double photodiode_std;

    // Magnetometer
    double magnetometer_noise_std;

    // Gyroscope
    double gyro_sigma_w;
    double gyro_sigma_v;
    double gyro_correlation_time;
    double gyro_scale_factor_err;

    // Solar Panels
    int num_panels;
    MatrixXd G_sp_b;
    double solar_panel_efficiency;
    double solar_panel_area;

    // Static power consumption
    double mb_power;
    double jetson_power;

    // Batteries
    double battery_capacity;
    double battery_initial_soc;
    double battery_internal_resistance;
    double battery_thermal_mass;
    double battery_radiative_loss;
    double battery_initial_temp;
    double max_pack_voltage;
    double solar_heat_factor;
    
    /* Simulation Settings */ 
    double MAX_TIME;                   // [s]
    double dt;                         // [s]
    double sim_start_time;             // [s] measured relative to J2000
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

    // Satellite Parameetr Dispersion distributions

    // Physical
    std::normal_distribution<double> mass_dist;
    std::normal_distribution<double> area_dist;
    std::normal_distribution<double> Ixx_dist;
    std::normal_distribution<double> Iyy_dist;
    std::normal_distribution<double> Izz_dist;

    // Actuators
    std::normal_distribution<double> rw_orientation_dist;
    std::normal_distribution<double> I_rw_dist;
    std::normal_distribution<double> mtb_orientation_dist;
    std::normal_distribution<double> mtb_resistance_dist;

    // Sensors
    std::normal_distribution<double> gps_pos_dist;
    std::normal_distribution<double> gps_vel_dist;
    std::normal_distribution<double> photodiode_orientation_dist;
    std::normal_distribution<double> photodiode_dist;
    std::normal_distribution<double> magnetometer_dist;
    std::normal_distribution<double> gyro_bias_dist;
    std::normal_distribution<double> gyro_white_noise_dist;
    std::normal_distribution<double> solar_panel_orientation_dist;

    // Initialization
    std::normal_distribution<double> sma_dist;
    std::normal_distribution<double> eccentricity_dist;
    std::normal_distribution<double> inclination_dist;
    std::normal_distribution<double> RAAN_dist;
    std::normal_distribution<double> AOP_dist;
    std::uniform_real_distribution<double> true_anomaly_dist;
    std::uniform_real_distribution<double> initial_attitude_dist;
    std::uniform_real_distribution<double> initial_angular_rate_dist;
    std::uniform_real_distribution<double> sim_start_time_dist;

    private:
    Magnetorquer load_MTB(std::string filename, std::mt19937 gen);
    void defineDistributions(std::string filename);
    std::mt19937 loadSeed(int trial_number);
};

#endif