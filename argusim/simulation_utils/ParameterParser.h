#ifndef _SIMULATOR_PARAMETER_PARSER_
#define _SIMULATOR_PARAMETER_PARSER_

#include <string>
#include <random>

#include "math/EigenWrapper.h"
#include "Magnetorquer.h"
#include "yaml-cpp/yaml.h"
#include "utils_and_transforms.h"


class Simulation_Parameters {
   public:
    Simulation_Parameters(std::string filename, int trial_number, std::string results_folder, std::string data_filename);
    Vector3 spinStabilizedRate(double tgt_ss_ang_vel);
    Vector4 nadirPointingAttitude(VectorXd State, std::mt19937 gen);
    Vector4 sunPointingAttitude(VectorXd State, std::mt19937 gen);
    
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

    // Center of Pressure/Mass arm
    Vector3 CoPM; // [m,m,m]

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
    VectorXd inductances;
    VectorXd Ahdt;
    VectorXd Bhdt;
    VectorXd Adt;
    VectorXd Bdt;
    MatrixXd mag_mtb_sens;
    MatrixXd G_mtb_b; // Matrix whose columns are the field axes of each MTB in the body frame
    double max_voltage;
    double max_current_rating;
    double max_power;
    Magnetorquer MTB; // Magnetorquer class object

    /* Sensors */
    // GPS
    double gps_dt; // [s]
    double gps_pos_std;
    double gps_vel_std;

    // Sun Sensors
    int num_photodiodes;
    MatrixXd G_pd_b; // orientation matrix for photodiodes
    double photodiode_std;
    // double sigma_sunsensor;
    double photodiode_dt; // sampling period of the photodiodes

    // Magnetometer
    double magnetometer_noise_std;
    double magnetometer_dt; // sampling period of the magnetometer
    double magnetometer_range_z;
    double magnetometer_range_xy; 
    double magnetometer_resolution;

    // Gyroscope
    double gyro_sigma_w;
    double gyro_sigma_v;
    double gyro_correlation_time;
    double gyro_scale_factor_err;
    double gyro_bias_stab;
    double gyro_range;
    int gyro_nbits;
    double gyro_resolution;
    //double gyro_bias_std; 
    Vector3 initial_gyro_bias;

    // Star Tracker
    int num_stk;
    bool include_star_tracker;
    double star_tracker_std; 

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
    bool useSRP;  // set to False to deactivate SRP calcs
    bool useSun;  // set to False to deactivate Sun calcs
    bool useMoon; // set to False to deactivate Moon calcs
    bool useDT;   // set to False to deactivate Drag Torque calcs
    bool useGG;   // set to False to deactivate Gravity Gradient calcs
    bool useLUTs; // set to False to deactivate LUTs

    /* Satellite Initialization */
    double semimajor_axis; // [m]
    double eccentricity; // [unitless]
    double inclination; // [deg]
    double RAAN; // [deg]
    double AOP; // [deg]
    double true_anomaly; // [deg]
    double LTDN; // [hours]
    Vector4 initial_attitude; 
    Vector3 initial_angular_rate; // [rad/s]
    VectorXd initial_state;

    // Index maps
    // using Seq = decltype(Eigen::seqN(0, 0)); // type alias for Eigen::seqN
    std::unordered_map<std::string, SliceDef> x_idx_map; // State Vector index map
    std::unordered_map<std::string, SliceDef> u_idx_map; // Control Vector index map
    std::unordered_map<std::string, SliceDef> y_idx_map; // Measurement Vector index map

    // Lookup tables
    int NElev;
    int NAzim;
    MatrixXd sc_area_LUT;
    MatrixXd sp_area_LUT;
    MatrixXd ss_visib_sum_LUT;
    //std::vector<MatrixXd> ss_visib_LUT;
    std::vector<MatrixXd> aero_torque_fac_LUT;
    std::vector<MatrixXd> aero_force_fac_LUT;

    // Satellite Parameetr Dispersion distributions

    // Physical
    std::normal_distribution<double> mass_dist;
    std::normal_distribution<double> area_dist;
    std::normal_distribution<double> CoPM_dist;
    std::normal_distribution<double> Ixx_dist;
    std::normal_distribution<double> Iyy_dist;
    std::normal_distribution<double> Izz_dist;


    // Actuators
    std::normal_distribution<double> rw_orientation_dist;
    std::normal_distribution<double> I_rw_dist;
    std::normal_distribution<double> mtb_orientation_dist;
    std::normal_distribution<double> mtb_resistance_dist;
    std::normal_distribution<double> mtb_inductance_dist;

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
    std::uniform_real_distribution<double> eccentricity_dist;
    std::normal_distribution<double> inclination_dist;
    std::normal_distribution<double> RAAN_dist;
    //std::normal_distribution<double> AOP_dist;
    std::normal_distribution<double> initial_angular_rate_dist;
    std::uniform_real_distribution<double> AOP_dist;
    std::uniform_real_distribution<double> LTDN_dist;
    std::uniform_real_distribution<double> true_anomaly_dist;
    std::uniform_real_distribution<double> initial_attitude_dist;
    //std::uniform_real_distribution<double> initial_angular_rate_dist;
    std::uniform_real_distribution<double> sim_start_time_dist;

    private:
    Magnetorquer load_MTB(std::string filename, std::mt19937 gen);
    void defineDistributions(std::string filename);
    void defineLUTs(std::string data_folder);
    std::mt19937 loadSeed(int trial_number);
};

#endif