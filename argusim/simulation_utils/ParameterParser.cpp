#include "ParameterParser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <random>
#include <math.h>
#include "ExpressionEvaluation.h"
#include "StringUtils/StringUtils.h"
#include "colored_output.h"
#include "math/EigenWrapper.h"
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

Simulation_Parameters::Simulation_Parameters(std::string filename, int trial_number, std::string results_folder, std::string data_filename) : 
                                    dev(loadSeed(trial_number)), MTB((defineDistributions(filename), load_MTB(filename, dev)))
{    
    /* Parse parameters */
    YAML::Node params = YAML::LoadFile(filename);
    defineDistributions(filename);
    useLUTs = params["useLUTs"].as<bool>();
    defineLUTs(data_filename);

    results_folder = results_folder; // PLACEHOLDER

    // Physical Parameters
    mass = mass_dist(dev);
    mass = std::min(params["mass"]["max_mass"].as<double>(), std::max(mass, params["mass"]["min_mass"].as<double>()));

    A = area_dist(dev);
    I_sat = Eigen::Map<Matrix_3x3, Eigen::RowMajor>(params["inertia"]["nominal_inertia"].as<std::vector<double>>().data());
    I_sat(0,0) = Ixx_dist(dev);
    I_sat(1,1) = Iyy_dist(dev);
    I_sat(2,2) = Izz_dist(dev);

    // Center of Pressure/Mass arm
    CoPM = Vector3::NullaryExpr([&](){return CoPM_dist(dev);});

    // Drag & SRP properties
    Cd = params["Cd"].as<double>();
    CR = params["CR"].as<double>();
    useDrag = params["useDrag"].as<bool>();
    useSRP = params["useSRP"].as<bool>();

    // Attitude perturbation properties
    useDT = params["useDragTorque"].as<bool>();
    useGG = params["useGravityGradient"].as<bool>();

    // Reaction Wheel
    num_RWs = params["reaction_wheels"]["N_rw"].as<int>();
    G_rw_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["reaction_wheels"]["rw_orientation"].as<std::vector<double>>().data(), 3, num_RWs);
    for (int i=0; i<num_RWs; i++) {
        G_rw_b.col(i) = random_SO3_rotation(rw_orientation_dist, dev)*G_rw_b.col(i);
    }
    I_rw = I_rw_dist(dev);

    // GPS
    gps_pos_std = params["gps"]["gps_pos_std"].as<double>(); // gps_pos_dist(dev);
    gps_vel_std = params["gps"]["gps_vel_std"].as<double>(); // gps_vel_dist(dev);

    // Photodiodes
    num_photodiodes = params["photodiodes"]["num_photodiodes"].as<int>();
    G_pd_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["photodiodes"]["photodiode_normals"].as<std::vector<double>>().data(), 3, num_photodiodes);
    for (int i=0; i<num_photodiodes; i++) {
        G_pd_b.col(i) = random_SO3_rotation(photodiode_orientation_dist, dev)*G_pd_b.col(i);
    }
    photodiode_std = photodiode_dist(dev);
    sigma_sunsensor = sigma_sunsensor_dist(dev);
    photodiode_dt = params["photodiodes"]["photodiodes_dt"].as<double>();


    // Magnetometer
    sigma_magnetometer = sigma_magnetometer_dist(dev);
    magnetometer_dt = params["magnetometer"]["magnetometer_dt"].as<double>();
    gyro_dt = params["gyroscope"]["gyro_dt"].as<double>();
    
    // Sim Settings
    MAX_TIME = params["MAX_TIME"].as<double>();
    dt = params["dt"].as<double>();
    controller_dt = params["controller_dt"].as<double>();
    estimator_dt  = params["estimator_dt"].as<double>();
    
    // Satellite Orbit Initialization
    semimajor_axis = sma_dist(dev);
    eccentricity = eccentricity_dist(dev);
    inclination = inclination_dist(dev);
    // RAAN = RAAN_dist(dev);
    AOP = AOP_dist(dev);
    bool disperse_ltdn = params["initialization"]["disperse_LTDN"].as<bool>();
    if (disperse_ltdn) {
        LTDN = LTDN_dist(dev);
    } else {
        LTDN = UTCStringtoHours(params["initialization"]["LTDN"].as<std::string>());
    }
    std::cout << "LTDN: " << LTDN << std::endl;

    bool disperse_true_anomaly = params["initialization"]["disperse_true_anomaly"].as<bool>();
    if (disperse_true_anomaly) {
        true_anomaly = true_anomaly_dist(dev);
    } else {
        true_anomaly = params["initialization"]["true_anomaly"].as<double>();
    }
    
    // Satellite Attitude Initialization
    bool disperse_initial_attitude = params["initialization"]["disperse_initial_attitude"].as<bool>();
    if (disperse_initial_attitude) {
        initial_attitude = Vector4::NullaryExpr([&](){return initial_attitude_dist(dev);});
        initial_attitude = initial_attitude/initial_attitude.norm();
    } else {
        initial_attitude = Eigen::Map<Vector4>(params["initialization"]["initial_attitude"].as<std::vector<double>>().data());
    }

    bool disperse_initial_angular_rate = params["initialization"]["disperse_initial_angular_rate"].as<bool>();
    if (disperse_initial_angular_rate) {
        initial_angular_rate = Vector3::NullaryExpr([&](){return initial_angular_rate_dist(dev);});
    } else {
        initial_angular_rate = Eigen::Map<Vector3>(params["initialization"]["initial_angular_rate"].as<std::vector<double>>().data());
    }

    // Sim Start Time
    sim_start_time = sim_start_time_dist(dev);

    RAAN = LTDN_to_RAAN(LTDN, sim_start_time);
    
    // Populate State Vector
    initial_true_state = initializeSatellite(sim_start_time);
    
    bool start_spin_stabilized = params["initialization"]["start_spin_stabilized"].as<bool>();
    bool start_ss_pointed = params["initialization"]["start_ss_pointed"].as<bool>();
    auto start_ss_pointing = params["initialization"]["start_ss_pointing"].as<std::string>(); //"Nadir" or "Sun"
    
    // adjust initial attitude and angular rate if to begin spin-stabilized/pointed
    if (start_spin_stabilized) {
        auto tgt_ss_ang_vel = params["tgt_ss_ang_vel"].as<double>();
        initial_angular_rate = spinStabilizedRate(tgt_ss_ang_vel);
        initial_true_state(Eigen::seqN(10,3)) = initial_angular_rate;
    }

    if (start_ss_pointed) {
        if (start_ss_pointing == "Nadir") {
            initial_attitude = nadirPointingAttitude(initial_true_state, dev);
        } else if (start_ss_pointing == "Sun") {
            initial_attitude = sunPointingAttitude(initial_true_state, dev);
        } else {
            throw std::invalid_argument("Invalid initial pointing direction. Must be 'Nadir' or 'Sun'.");
        }
        initial_true_state(Eigen::seqN(6,4)) = initial_attitude;
    }
    
    // Dump Dispersed Parameters to YAML
    dumpSampledParametersToYAML(results_folder);
}

Magnetorquer Simulation_Parameters::load_MTB(std::string filename, std::mt19937 gen)
{
    YAML::Node params = YAML::LoadFile(filename);
    num_MTBs = params["magnetorquers"]["N_mtb"].as<int>();
    
    resistances = VectorXd::NullaryExpr(num_MTBs,[&](){return mtb_resistance_dist(gen);});
    double resistance_bound = params["magnetorquers"]["mtb_resistance_lub"].as<double>();
    resistances = resistances.cwiseMax((1-resistance_bound)*resistances).cwiseMin((1+resistance_bound)*resistances);  

    G_mtb_b = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(params["magnetorquers"]["mtb_orientation"].as<std::vector<double>>().data(), 3, num_MTBs);
    for (int i=0; i<num_MTBs; i++) {
        G_mtb_b.col(i) = random_SO3_rotation(mtb_orientation_dist, gen)*G_mtb_b.col(i);
    }
    
    Magnetorquer magnetorquer = Magnetorquer(
                                    params["magnetorquers"]["N_mtb"].as<int>(),
                                    resistances,
                                    params["magnetorquers"]["A_cross"].as<double>(),
                                    params["magnetorquers"]["N_turns"].as<double>(),
                                    params["magnetorquers"]["max_voltage"].as<double>(),
                                    params["magnetorquers"]["max_current_rating"].as<double>(),
                                    params["magnetorquers"]["max_power"].as<double>(),
                                    G_mtb_b
                                    );

    return magnetorquer;
}

Vector3 Simulation_Parameters::spinStabilizedRate(double tgt_ss_ang_vel)
{
    // get target angular momentum
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver;
    eigensolver.compute(I_sat);
    Eigen::VectorXd eigen_values = eigensolver.eigenvalues().real();
    Eigen::MatrixXd eigen_vectors = eigensolver.eigenvectors().real();
    double maxeigenval = 0;
    Eigen::Vector3d I_max_dir = Eigen::Vector3d::Zero();
    for(int i=0; i<eigen_values.size(); i++){
        if (eigen_values[i] > maxeigenval) {
            maxeigenval = eigen_values[i];
            I_max_dir = eigen_vectors.col(i);
        }
    }
    // largest element should be positive
    if (std::abs(I_max_dir.cwiseAbs().maxCoeff() + I_max_dir.maxCoeff()) < 1e-9) {
        I_max_dir = -I_max_dir;
    }
    
    // Eigen::Vector3d h_tgt = maxeigenval*tgt_ss_ang_vel*I_max_dir*M_PI/180.0;
    // double h_tgt_norm = h_tgt.norm();

    initial_angular_rate = tgt_ss_ang_vel*I_max_dir*M_PI/180.0;
    // I_sat.inverse() * h_tgt;

    // [TODO]: disperse around tolerated values
    // spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        
    return initial_angular_rate;
}

Vector4 Simulation_Parameters::nadirPointingAttitude(VectorXd State, std::mt19937 gen)
{
    // angular momentum direction in body frame
    Eigen::Vector3d h = I_sat * State(Eigen::seqN(10,3));
    std::uniform_real_distribution<> dis(-1, 1);
    auto uni = [&](){ return dis(gen); };
    Eigen::Vector3d v1 = Eigen::Vector3d::NullaryExpr(3,uni);
    Eigen::Vector3d h_normalized = h.normalized();
    Eigen::Vector3d v1_proj_h = v1.dot(h_normalized) * h_normalized;
    v1 -= v1_proj_h;
    v1.normalize();
    Eigen::Vector3d v2 = h_normalized.cross(v1);
    v2.normalize();
    Eigen::Matrix3d Rb; 
    Rb << h_normalized, v1, v2;

    // sun direction in inertial frame 
    //Eigen::Vector3d s = State(Eigen::seqN(13, 3));
    Eigen::Vector3d init_pos = State(Eigen::seqN(0, 3));
    Eigen::Vector3d init_vel = State(Eigen::seqN(3, 3));
    Eigen::Vector3d s = init_pos.cross(init_vel);
    std::uniform_real_distribution<> dis2(-1, 1);
    auto uni2 = [&](){ return dis2(gen); };
    Eigen::Vector3d v3 = Eigen::Vector3d::NullaryExpr(3,uni2);
    Eigen::Vector3d s_normalized = s.normalized();
    Eigen::Vector3d v3_proj_h = v3.dot(s_normalized) * s_normalized;
    v3 -= v3_proj_h;
    v3.normalize();
    Eigen::Vector3d v4 = s_normalized.cross(v3);
    v4.normalize();
    Eigen::Matrix3d Ri; 
    Ri << s_normalized, v3, v4;

    Eigen::Matrix3d Rb2i = Ri * Rb.inverse();
    Eigen::Quaterniond q(Rb2i);
    Eigen::Matrix<double, 4, 1> init_att(q.w(), q.x(), q.y(), q.z());
    //initial_attitude = q;
    // sun_pointing = (np.linalg.norm(sun_vector-(h/h_norm))<= np.deg2rad(10))

    return init_att;
}

Vector4 Simulation_Parameters::sunPointingAttitude(VectorXd State, std::mt19937 gen)
{
    // angular momentum direction in body frame
    Eigen::Vector3d h = I_sat * State(Eigen::seqN(10,3));
    std::uniform_real_distribution<> dis(-1, 1);
    auto uni = [&](){ return dis(gen); };
    Eigen::Vector3d v1 = Eigen::Vector3d::NullaryExpr(3,uni);
    Eigen::Vector3d h_normalized = h.normalized();
    Eigen::Vector3d v1_proj_h = v1.dot(h_normalized) * h_normalized;
    v1 -= v1_proj_h;
    v1.normalize();
    Eigen::Vector3d v2 = h_normalized.cross(v1);
    v2.normalize();
    Eigen::Matrix3d Rb; 
    Rb << h_normalized, v1, v2;

    // sun direction in inertial frame 
    Eigen::Vector3d s = State(Eigen::seqN(13, 3));
    std::uniform_real_distribution<> dis2(-1, 1);
    auto uni2 = [&](){ return dis2(gen); };
    Eigen::Vector3d v3 = Eigen::Vector3d::NullaryExpr(3,uni2);
    Eigen::Vector3d s_normalized = s.normalized();
    Eigen::Vector3d v3_proj_h = v3.dot(s_normalized) * s_normalized;
    v3 -= v3_proj_h;
    v3.normalize();
    Eigen::Vector3d v4 = s_normalized.cross(v3);
    v4.normalize();
    Eigen::Matrix3d Ri; 
    Ri << s_normalized, v3, v4;

    Eigen::Matrix3d Rb2i = Ri * Rb.inverse();
    Eigen::Quaterniond q(Rb2i);
    Eigen::Matrix<double, 4, 1> init_att(q.w(), q.x(), q.y(), q.z());
    //initial_attitude = q;
    // sun_pointing = (np.linalg.norm(sun_vector-(h/h_norm))<= np.deg2rad(10))

    return init_att;
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

void Simulation_Parameters::defineDistributions(std::string filename) 
{
    YAML::Node params = YAML::LoadFile(filename);
    
    // Mass
    double mass_nominal = params["mass"]["nominal_mass"].as<double>();
    double mass_std = mass_nominal*(params["mass"]["mass_dev"].as<double>()/100);
    mass_dist = std::normal_distribution<double>(mass_nominal, mass_std);

    // Area
    double area_nominal = params["area"]["nominal_area"].as<double>();
    double area_std = area_nominal*(params["area"]["area_dev"].as<double>()/100);
    area_dist = std::normal_distribution<double>(area_nominal, area_std);

    // Center of Pressure/Mass arm
    double CoPM_std = params["CoPM_dev"].as<double>();
    CoPM_dist = std::normal_distribution<double>(0, CoPM_std);

    // Inertia
    Vector3 inertia_dev = Eigen::Map<Vector3>(params["inertia"]["principal_axis_dev"].as<std::vector<double>>().data());
    MatrixXd Isat = Eigen::Map<Matrix_3x3, Eigen::RowMajor>(params["inertia"]["nominal_inertia"].as<std::vector<double>>().data());
    
    double Ixx_std = Isat(0,0)*inertia_dev(0)/100;
    Ixx_dist = std::normal_distribution<double>(Isat(0,0), Ixx_std);

    double Iyy_std = Isat(1,1)*inertia_dev(1)/100;
    Iyy_dist = std::normal_distribution<double>(Isat(1,1), Iyy_std);

    double Izz_std = Isat(2,2)*inertia_dev(2)/100;
    Izz_dist = std::normal_distribution<double>(Isat(2,2), Izz_std);

    // Reaction Wheels
    rw_orientation_dist = std::normal_distribution<double>(0, params["reaction_wheels"]["rw_orientation_dev"].as<double>());
    double I_rw_nominal = params["reaction_wheels"]["I_rw"].as<double>();
    double I_rw_std = I_rw_nominal*(params["reaction_wheels"]["I_rw_dev"].as<double>()/100);
    I_rw_dist = std::normal_distribution<double>(I_rw_nominal, I_rw_std);

    // Magnetorquers
    mtb_orientation_dist = std::normal_distribution<double>(0, params["magnetorquers"]["mtb_orientation_dev"].as<double>());
    double mtb_resistance_nominal = params["magnetorquers"]["mtb_resistance"].as<double>();
    double mtb_resistance_std = mtb_resistance_nominal*(params["magnetorquers"]["mtb_resistance_dev"].as<double>())/100;
    mtb_resistance_dist = std::normal_distribution<double>(mtb_resistance_nominal, mtb_resistance_std);

    // GPS
    double gps_pos_std_nominal = params["gps"]["gps_pos_std"].as<double>();
    double gps_pos_std_std = gps_pos_std_nominal*(params["gps"]["gps_pos_std_dev"].as<double>()/100);
    std::normal_distribution<double>(gps_pos_std_nominal, gps_pos_std_std);
    
    double gps_vel_std_nominal = params["gps"]["gps_vel_std"].as<double>();
    double gps_vel_std_std = gps_vel_std_nominal*(params["gps"]["gps_vel_std_dev"].as<double>()/100);
    gps_vel_dist = std::normal_distribution<double>(gps_vel_std_nominal, gps_vel_std_std);

    // Photodiodes
    photodiode_orientation_dist = std::normal_distribution<double>(0, params["photodiodes"]["photodiode_orientation_dev"].as<double>());
    double photodiode_std_nominal = params["photodiodes"]["photodiode_std"].as<double>();
    double photodiode_std_std = photodiode_std_nominal*(params["photodiodes"]["photodiode_std_dev"].as<double>()/100);
    double min_sigma_sunsensor = params["photodiodes"]["min_sigma_sunsensor"].as<double>();
    double max_sigma_sunsensor = params["photodiodes"]["max_sigma_sunsensor"].as<double>();
    sigma_sunsensor_dist = std::uniform_real_distribution<>(min_sigma_sunsensor, max_sigma_sunsensor);
    photodiode_dist = std::normal_distribution<double>(photodiode_std_nominal, photodiode_std_std);

    // Magnetometer
    double min_sigma_magnetometer = params["magnetometer"]["min_sigma_magnetometer"].as<double>();
    double max_sigma_magnetometer = params["magnetometer"]["max_sigma_magnetometer"].as<double>();
    sigma_magnetometer_dist = std::uniform_real_distribution<>(min_sigma_magnetometer, max_sigma_magnetometer);

    // Initialization
    double sma_nominal = params["initialization"]["semimajor_axis"].as<double>();
    double sma_std = params["initialization"]["semimajor_axis_dev"].as<double>(); //0.01*sma_nominal*
    sma_dist = std::normal_distribution<double>(sma_nominal, sma_std);

    double ecc_nominal = params["initialization"]["eccentricity"].as<double>();
    double ecc_std = ecc_nominal*(params["initialization"]["eccentricity_dev"].as<double>()/100);
    eccentricity_dist = std::normal_distribution<double>(ecc_nominal, ecc_std);

    double incl_nominal = params["initialization"]["inclination"].as<double>();
    double incl_std = incl_nominal*(params["initialization"]["inclination_dev"].as<double>()/100);
    inclination_dist = std::normal_distribution<double>(incl_nominal, incl_std);

    double LTDN_min = UTCStringtoHours(params["initialization"]["LTDN_min"].as<std::string>());
    double LTDN_max = UTCStringtoHours(params["initialization"]["LTDN_max"].as<std::string>());
    LTDN_dist = std::uniform_real_distribution<double>(LTDN_min, LTDN_max);

    double RAAN_nominal = params["initialization"]["RAAN"].as<double>();
    double RAAN_std = RAAN_nominal*(params["initialization"]["RAAN_dev"].as<double>()/100);
    RAAN_dist = std::normal_distribution<double>(RAAN_nominal, RAAN_std);

    //double AOP_nominal = params["initialization"]["AOP"].as<double>();
    //double AOP_std = AOP_nominal*(params["initialization"]["AOP_dev"].as<double>()/100);
    // std::normal_distribution<double>(AOP_nominal, AOP_std);
    AOP_dist = std::uniform_real_distribution<double>(0, 360); // True anomaly uniformly distributed between 0 and 360

    true_anomaly_dist = std::uniform_real_distribution<double>(0, 360); // True anomaly uniformly distributed between 0 and 360

    initial_attitude_dist = std::uniform_real_distribution<double>(-1,1);

    // auto angular_rate_bound = params["initialization"]["initial_angular_rate_bound"].as<double>();
    // initial_angular_rate_dist = std::uniform_real_distribution<double>(-angular_rate_bound, angular_rate_bound);
    double angular_rate_std = params["initialization"]["initial_angular_rate_dev"].as<double>();
    initial_angular_rate_dist = std::normal_distribution<double>(0, angular_rate_std);
    
    //double sma_nominal = params["initialization"]["semimajor_axis"].as<double>();
    //double sma_std = 0.01*sma_nominal*params["initialization"]["semimajor_axis_dev"].as<double>();
    //sma_dist = std::normal_distribution<double>(sma_nominal, sma_std);

    double earliest_sim_start_J2000 = UTCStringtoTJ2000(params["earliest_sim_start_time_UTC"].as<std::string>());
    double latest_sim_start_J2000 = UTCStringtoTJ2000(params["latest_sim_start_time_UTC"].as<std::string>());
    sim_start_time_dist = std::uniform_real_distribution<double>(earliest_sim_start_J2000, latest_sim_start_J2000);
}

void Simulation_Parameters::defineLUTs(std::string data_folder)
{
    // Load LUTs
    if (data_folder.empty() || !useLUTs) {
        // Load dummy variables into LUTs
        NElev = 0;
        NAzim = 0;
        NSS   = 0;
        sc_area_LUT         = Eigen::MatrixXd::Zero(3, 3);
        sp_area_LUT         = Eigen::MatrixXd::Zero(3, 3);
        ss_visib_sum_LUT    = Eigen::MatrixXd::Zero(3, 3);
        //ss_visib_LUT        = std::vector<Eigen::MatrixXd>(3,   Eigen::MatrixXd::Zero(3, 3));
        aero_torque_fac_LUT = std::vector<Eigen::MatrixXd>(3,   Eigen::MatrixXd::Zero(3, 3));
        aero_force_fac_LUT  = std::vector<Eigen::MatrixXd>(3,   Eigen::MatrixXd::Zero(3, 3));

    } else {
        // Load actual LUTs from data_folder
        YAML::Node data_params = YAML::LoadFile(data_folder);

        NElev = data_params["NE"].as<int>();
        NAzim = data_params["NA"].as<int>();
        NSS   = data_params["NS"].as<int>();

        sc_area_LUT         = Eigen::MatrixXd::Zero(NElev, NAzim);
        sp_area_LUT         = Eigen::MatrixXd::Zero(NElev, NAzim);
        ss_visib_sum_LUT    = Eigen::MatrixXd::Zero(NElev, NAzim);
        // ss_visib_LUT        = std::vector<Eigen::MatrixXd>(NSS, Eigen::MatrixXd::Zero(NElev, NAzim));
        aero_torque_fac_LUT = std::vector<Eigen::MatrixXd>(3,   Eigen::MatrixXd::Zero(NElev, NAzim));
        aero_force_fac_LUT  = std::vector<Eigen::MatrixXd>(3,   Eigen::MatrixXd::Zero(NElev, NAzim));
        
        for (int i = 0; i < NElev; ++i) {
            for (int j = 0; j < NAzim; ++j) {
                sc_area_LUT(i, j) = data_params["effective_sc_area"][i][j].as<double>();
                sp_area_LUT(i, j) = data_params["effective_sp_area"][i][j].as<double>();
                ss_visib_sum_LUT(i, j) = data_params["visibility_sum"][i][j].as<double>();

                //for (int k = 0; k < NSS; ++k) {
                //    ss_visib_LUT[k](i, j) = data_params["visibility"][k][i][j].as<double>();
                //}
                for (int k = 0; k < 3; ++k) {
                    aero_torque_fac_LUT[k](i, j) = data_params["aero_torque_fac"][i][j][k].as<double>();
                    aero_force_fac_LUT[k](i, j)  = data_params["aero_force_fac"][i][j][k].as<double>();
                }
            }
        }
    }
}

std::mt19937 Simulation_Parameters::loadSeed(int trial_number)
{
    std::mt19937 gen;
    gen.seed(trial_number);

    return gen;
}

void Simulation_Parameters::dumpSampledParametersToYAML(std::string results_folder)
{
    YAML::Emitter out;
    std::string outpath = results_folder.append("/trial_params.yaml");
    
    out << YAML::BeginSeq;
    out << YAML::Key << "mass" << mass;
    out << YAML::Key << "area" << A;

    std::vector<double> vec;
    vec.assign(I_sat.data(), I_sat.data() + 9);
    out << YAML::Key << "inertia" << YAML::Flow << vec;

    vec.assign(G_rw_b.data(), G_rw_b.data() + G_rw_b.rows()*G_rw_b.cols());
    out << YAML::Key << "rw_orientation" << YAML::Flow << vec;
    out << YAML::Key << "I_rw" <<  I_rw;
    
    vec.assign(G_mtb_b.data(), G_mtb_b.data() + G_mtb_b.rows()*G_mtb_b.cols());
    out << YAML::Key << "mtb_orientation" << YAML::Flow << vec;

    vec.assign(resistances.data(), resistances.data() + resistances.size());
    out << YAML::Key << "mtb_resistances" << YAML::Flow << vec;

    out << YAML::Key << "gps_pos_std" << gps_pos_std;
    out << YAML::Key << "gps_vel_std" << gps_vel_std;

    vec.assign(G_pd_b.data(), G_pd_b.data() + G_pd_b.rows()*G_pd_b.cols());
    out<<YAML::Key << "photodiode_orientation" << YAML::Flow << vec;

    out<<YAML::Key << "magnetometer_std" << sigma_magnetometer;

    out<<YAML::Key << "gyro_sigma_w" <<gyro_sigma_w;
    out<< YAML::Key << "gyro_sigma_v" << gyro_sigma_v;

    out<<YAML::Key << "semimajor_axis" << semimajor_axis;
    out<<YAML::Key << "eccentricity" << eccentricity;
    out<<YAML::Key << "inclination" << inclination;
    out<<YAML::Key << "RAAN" << RAAN;
    out<<YAML::Key << "AOP" << AOP;
    out<<YAML::Key << "true_anomaly" << true_anomaly;

    vec.assign(initial_attitude.data(), initial_attitude.data() + initial_attitude.size());
    out<<YAML::Key << "initial_attitude" << YAML::Flow << vec;
    vec.assign(initial_angular_rate.data(), initial_angular_rate.data() + initial_angular_rate.size());
    out<<YAML::Key << "initial_angular_rate" << YAML::Flow << vec;

    out<<YAML::Key << "sim_start_time" << TJ2000toUTCString(sim_start_time);

    out<<YAML::EndSeq;

    std::ofstream fout(outpath);
    fout << out.c_str();
}

#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysim_utils, m) {
    pybind11::class_<Simulation_Parameters>(m, "Simulation_Parameters")
        .def(pybind11::init<std::string, int, std::string, std::string>())
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
        .def_readonly("num_photodiodes", &Simulation_Parameters::num_photodiodes)
        .def_readonly("photodiodes_dt", &Simulation_Parameters::photodiode_dt)
        .def_readonly("sigma_sunsensor", &Simulation_Parameters::sigma_sunsensor)
        //
        .def_readonly("num_MTBs", &Simulation_Parameters::num_MTBs)
        .def_readonly("G_mtb_b", &Simulation_Parameters::G_mtb_b)
        //
        .def_readonly("magnetometer_dt", &Simulation_Parameters::magnetometer_dt)
        .def_readonly("sigma_magnetometer", &Simulation_Parameters::sigma_magnetometer)
        //
        .def_readonly("gyro_dt", &Simulation_Parameters::gyro_dt)
        //
        .def_readonly("MAX_TIME", &Simulation_Parameters::MAX_TIME)
        .def_readonly("dt", &Simulation_Parameters::dt)
        .def_readonly("sim_start_time", &Simulation_Parameters::sim_start_time)
        .def_readonly("useDrag", &Simulation_Parameters::useDrag)
        .def_readonly("useSRP", &Simulation_Parameters::useSRP)
        //
        .def_readonly("sp_area_LUT", &Simulation_Parameters::sp_area_LUT)
        //
        .def_readonly("initial_true_state", &Simulation_Parameters::initial_true_state);
}

#endif