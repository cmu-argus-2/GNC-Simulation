#include "ParameterParser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <random>

#include "ExpressionEvaluation.h"
#include "StringUtils/StringUtils.h"
#include "colored_output.h"
#include "misc.h"
#include "yaml-cpp/yaml.h"
#include "utils_and_transforms.h"
#include "ReactionWheel.h"
#include "Magnetorquer.h"
#include "MagneticField.h"
#include "SRP.h"

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic pop
#endif

// FSW includes

// ==========================================================================
// ============================ Helper Functions ============================
// ==========================================================================
// ==========================================================================

Simulation_Parameters::Simulation_Parameters(std::string filename, int trial_number, std::string results_folder) : 
                                    dev(loadSeed(trial_number)), MTB(load_MTB(filename, dev))
{    
    std::cout << "skdfskdjfbsdkjf";
    /* Parse parameters */
    YAML::Node params = YAML::LoadFile(filename);

    results_folder = results_folder; // PLACEHOLDER
    std::cout << "wsfyhjfsdjfh";
    defineDistributions(params);

    std::cout << "Here";

    // Physical Parameters
    mass = mass_dist(dev);
    I_sat = Eigen::Map<Matrix_3x3, Eigen::RowMajor>(params["inertia"].as<std::vector<double>>().data());
    A = area_dist(dev);

    // Drag & SRP properties
    Cd = params["Cd"].as<double>();
    CR = params["CR"].as<double>();
    useDrag = params["useDrag"].as<bool>();
    useSRP = params["useSRP"].as<bool>();

    // Reaction Wheel
    num_RWs = params["N_rw"].as<int>();
    G_rw_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["rw_orientation"].as<std::vector<double>>().data(), 3, num_RWs);
    for (int i=0; i<num_RWs; i++) {
        G_rw_b.col(i) = random_SO3_rotation(rw_orientation_dist, dev)*G_rw_b.col(i);
    }
    I_rw = I_rw_dist(dev);

    // GPS
    gps_pos_std = gps_pos_dist(dev);
    gps_vel_std = gps_vel_dist(dev);

    // Photodiodes
    num_photodiodes = params["num_photodiodes"].as<int>();
    G_pd_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["photodiode_normals"].as<std::vector<double>>().data(), 3, num_photodiodes);
    for (int i=0; i<num_photodiodes; i++) {
        G_pd_b.col(i) = random_SO3_rotation(photodiode_orientation_dist, dev)*G_pd_b.col(i);
    }
    photodiode_std = photodiode_dist(dev);

    // Magnetometer
    magnetometer_noise_std =magnetometer_dist(dev);

    // Gyroscope
    gyro_sigma_w = gyro_bias_dist(dev);
    gyro_sigma_v = gyro_white_noise_dist(dev);
    gyro_correlation_time = params["gyro_correlation_time"].as<double>(); 
    gyro_scale_factor_err = params["gyro_scale_factor_err"].as<double>();

    // Sim Settings
    MAX_TIME = params["MAX_TIME"].as<double>();
    dt = params["dt"].as<double>();
    controller_dt = params["controller_dt"].as<double>();
    estimator_dt  = params["estimator_dt"].as<double>();
    
    // Satellite Orbit Initialization
    semimajor_axis = sma_dist(dev);
    eccentricity = eccentricity_dist(dev);
    inclination = inclination_dist(dev);
    RAAN = RAAN_dist(dev);
    AOP = AOP_dist(dev);
    true_anomaly = true_anomaly_dist(dev);
    
    // Satellite Attitude Initialization
    initial_attitude = Vector4::NullaryExpr([&](){return initial_attitude_dist(dev);});
    initial_attitude = initial_attitude/initial_attitude.norm();

    initial_angular_rate = Vector3::NullaryExpr([&](){return initial_angular_rate_dist(dev);});
    sim_start_time = sim_start_time_dist(dev);
    
    // Populate State Vector
    initial_state = initializeSatellite(sim_start_time);

    // Dump Dispersed Parameters to YAML
    //dumpSampledParametersToYAML(results_folder);
}

Magnetorquer Simulation_Parameters::load_MTB(std::string filename, std::mt19937 gen)
{
    YAML::Node params = YAML::LoadFile(filename);
    std::cout << "HELLO";
    num_MTBs = params["N_mtb"].as<int>();
    std::cout << "sdfhsdfjsdhfb";
    printf("sfskjdfskdjf");
    
    resistances = VectorXd::NullaryExpr(num_MTBs,[&](){return mtb_resistance_dist(gen);});

    G_mtb_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["mtb_orientation"].as<std::vector<double>>().data(), 3, num_MTBs);
    for (int i=0; i<num_MTBs; i++) {
        G_mtb_b.col(i) = random_SO3_rotation(mtb_orientation_dist, gen)*G_mtb_b.col(i);
    }
    
    Magnetorquer magnetorquer = Magnetorquer(
                                    params["N_mtb"].as<int>(),
                                    resistances,
                                    params["A_cross"].as<double>(),
                                    params["N_turns"].as<double>(),
                                    params["max_voltage"].as<double>(),
                                    params["max_current_rating"].as<double>(),
                                    params["max_power"].as<double>(),
                                    G_mtb_b
                                    );

    return magnetorquer;
}

VectorXd Simulation_Parameters::initializeSatellite(double epoch)
{    
    VectorXd State(19+num_RWs);

    Vector6 KOE {semimajor_axis, eccentricity, inclination, RAAN, AOP, true_anomaly};
    Vector6 CartesianState = KOE2ECI(KOE, epoch);
    State(Eigen::seqN(0,6)) = CartesianState;
    State(Eigen::seqN(6,4)) = initial_attitude;
    State(Eigen::seqN(10,3)) = initial_angular_rate;
    State(Eigen::seqN(13, 3)) = sun_position_eci(epoch);
    State(Eigen::seqN(16, 3)) = MagneticField(State(Eigen::seqN(0, 3)), epoch);
    State(Eigen::seqN(19, num_RWs)).setZero();

    return State;

}

void Simulation_Parameters::defineDistributions(YAML::Node params) 
{
    // Mass
    double mass_nominal = params["mass"].as<double>();
    double mass_std = mass_nominal*(params["mass_dev"].as<double>()/100);
    mass_dist = std::normal_distribution<double>(mass_nominal, mass_std);

    // Area
    double area_nominal = params["area"].as<double>();
    double area_std = area_nominal*(params["area_dev"].as<double>()/100);
    area_dist = std::normal_distribution<double>(area_nominal, area_std);

    // Inertia


    // Reaction Wheels
    rw_orientation_dist = std::normal_distribution<double>(0, params["rw_orientation_dev"].as<double>());
    double I_rw_nominal = params["I_rw"].as<double>();
    double I_rw_std = I_rw_nominal*(params["I_rw_dev"].as<double>()/100);
    I_rw_dist = std::normal_distribution<double>(I_rw_nominal, I_rw_std);

    // Magnetorquers
    mtb_orientation_dist = std::normal_distribution<double>(0, params["mtb_orientation_dev"].as<double>());
    double mtb_resistance_nominal = params["mtb_resistance"].as<double>();
    double mtb_resistance_std = mtb_resistance_nominal*(params["mtb_resistance_dev"].as<double>()/100);
    mtb_resistance_dist = std::normal_distribution<double>(mtb_resistance_nominal, mtb_resistance_std);

    // GPS
    double gps_pos_std_nominal = params["gps_pos_std"].as<double>();
    double gps_pos_std_std = gps_pos_std_nominal*(params["gps_pos_std_dev"].as<double>()/100);
    gps_pos_dist = std::normal_distribution<double>(gps_pos_std_nominal, gps_pos_std_std);
    
    double gps_vel_std_nominal = params["gps_vel_std"].as<double>();
    double gps_vel_std_std = gps_vel_std_nominal*(params["gps_vel_std_dev"].as<double>()/100);
    gps_vel_dist = std::normal_distribution<double>(gps_vel_std_nominal, gps_vel_std_std);

    // Photodiodes
    photodiode_orientation_dist = std::normal_distribution<double>(0, params["photodiode_orientation_dev"].as<double>());
    double photodiode_std_nominal = params["photodiode_std"].as<double>();
    double photodiode_std_std = photodiode_std_nominal*(params["photodiode_std_dev"].as<double>()/100);
    photodiode_dist = std::normal_distribution<double>(photodiode_std_nominal, photodiode_std_std);

    // Magnetometer
    double magnetometer_noise_std_nominal = params["magnetometer_noise_std"].as<double>();
    double magnetometer_noise_std_std = magnetometer_noise_std_nominal*(params["magnetometer_std_dev"].as<double>()/100);
    magnetometer_dist = std::normal_distribution<double>(magnetometer_noise_std_nominal, magnetometer_noise_std_std);

    // Gyroscope
    double gyro_sigma_w_nominal = params["gyro_sigma_w"].as<double>();
    double gyro_sigma_w_std = gyro_sigma_w_nominal*(params["gyro_sigma_w_dev"].as<double>()/100);
    gyro_bias_dist = std::normal_distribution<double>(gyro_sigma_w_nominal, gyro_sigma_w_std);

    double gyro_sigma_v_nominal = params["gyro_sigma_v"].as<double>();
    double gyro_sigma_v_std = gyro_sigma_v_nominal*(params["gyro_sigma_v_dev"].as<double>()/100);
    gyro_white_noise_dist = std::normal_distribution<double>(gyro_sigma_v_nominal, gyro_sigma_v_std);

    // Initialization
    double sma_nominal = params["semimajor_axis"].as<double>();
    double sma_std = sma_nominal*(params["semimajor_axis_dev"].as<double>()/100);
    sma_dist = std::normal_distribution<double>(sma_nominal, sma_std);

    double ecc_nominal = params["eccentricity"].as<double>();
    double ecc_std = sma_nominal*(params["eccentricity_dev"].as<double>()/100);
    eccentricity_dist = std::normal_distribution<double>(ecc_nominal, ecc_std);

    double incl_nominal = params["inclination"].as<double>();
    double incl_std = sma_nominal*(params["inclination_dev"].as<double>()/100);
    sma_dist = std::normal_distribution<double>(incl_nominal, incl_std);

    double RAAN_nominal = params["RAAN"].as<double>();
    double RAAN_std = RAAN_nominal*(params["RAAN_dev"].as<double>()/100);
    RAAN_dist = std::normal_distribution<double>(RAAN_nominal, RAAN_std);

    double AOP_nominal = params["AOP"].as<double>();
    double AOP_std = AOP_nominal*(params["AOP_dev"].as<double>()/100);
    AOP_dist = std::normal_distribution<double>(AOP_nominal, AOP_std);

    double true_anomaly_nominal = params["true_anomaly"].as<double>();
    double true_anomaly_std = true_anomaly_nominal*(params["true_anomaly_dev"].as<double>()/100);
    true_anomaly_dist = std::normal_distribution<double>(true_anomaly_nominal, true_anomaly_std);

    initial_attitude_dist = std::uniform_real_distribution<double>(-1,1);

    double angular_rate_bound = params["initial_angular_rate_bound"].as<double>();
    initial_angular_rate_dist = std::uniform_real_distribution<double>(-angular_rate_bound, angular_rate_bound);

    double earliest_sim_start_J2000 = UTCStringtoTJ2000(params["earliest_sim_start_time_UTC"].as<std::string>());
    double latest_sim_start_J2000 = UTCStringtoTJ2000(params["latest_sim_start_time_UTC"].as<std::string>());
    sim_start_time_dist = std::uniform_real_distribution<double>(earliest_sim_start_J2000, latest_sim_start_J2000);

}

std::mt19937 Simulation_Parameters::loadSeed(int trial_number)
{
    std::mt19937 gen(trial_number);

    return gen;
}

/*void Simulation_Parameters::dumpSampledParametersToYAML(std::string results_folder)
{

}*/

#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysim_utils, m) {
    pybind11::class_<Simulation_Parameters>(m, "Simulation_Parameters")
        .def(pybind11::init<std::string, int, std::string>())
        //.def("getParamsFromFileAndSample", &Simulation_Parameters::getParamsFromFileAndSample)
        //.def("dumpSampledParametersToYAML", &Simulation_Parameters::dumpSampledParametersToYAML)
        .def_readonly("mass", &Simulation_Parameters::mass)
        .def_readonly("inertia", &Simulation_Parameters::I_sat)
        .def_readonly("facet_area", &Simulation_Parameters::A)
        //
        .def_readonly("num_RWs", &Simulation_Parameters::num_RWs)
        .def_readonly("G_rw_b", &Simulation_Parameters::G_rw_b)
        .def_readonly("mass", &Simulation_Parameters::mass)
        .def_readonly("inertia_RW", &Simulation_Parameters::I_rw)
        //
        .def_readonly("num_MTBs", &Simulation_Parameters::num_MTBs)
        .def_readonly("G_mtb_b", &Simulation_Parameters::G_mtb_b)
        //
        .def_readonly("num_photodiodes", &Simulation_Parameters::num_photodiodes)
        //
        .def_readonly("MAX_TIME", &Simulation_Parameters::MAX_TIME)
        .def_readonly("dt", &Simulation_Parameters::dt)
        .def_readonly("sim_start_time", &Simulation_Parameters::sim_start_time)
        .def_readonly("useDrag", &Simulation_Parameters::useDrag)
        .def_readonly("useSRP", &Simulation_Parameters::useSRP)
        //
        .def_readonly("initial_state", &Simulation_Parameters::initial_state);
}

#endif