#include "ParameterParser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "ExpressionEvaluation.h"
#include "StringUtils/StringUtils.h"
#include "colored_output.h"
#include "misc.h"
#include "yaml-cpp/yaml.h"
#include "utils_and_transforms.h"
#include "ReactionWheel.h"
#include "Magnetorquer.h"

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

Simulation_Parameters::Simulation_Parameters(std::string filename) : MTB(load_MTB(filename))
{
    YAML::Node params = YAML::LoadFile(filename);
    
    // Parse parameters
    mass = params["mass"].as<double>();
    I_sat = Eigen::Map<Matrix_3x3, Eigen::RowMajor>(params["inertia"].as<std::vector<double>>().data());
    A = params["area"].as<double>();

    Cd = params["Cd"].as<double>();
    CR = params["CR"].as<double>();

    num_RWs = params["N_rw"].as<int>();
    G_rw_b = Eigen::Map<Eigen::MatrixXd>(params["rw_orientation"].as<std::vector<double>>().data(), 3, num_RWs);
    I_rw = params["I_rw"].as<double>();


    MAX_TIME = params["MAX_TIME"].as<double>();
    dt = params["dt"].as<double>();
    earliest_sim_start_unix = UTCStringtoTJ2000(params["earliest_sim_start_time_UTC"].as<std::string>());
    latest_sim_start_unix = UTCStringtoTJ2000(params["latest_sim_start_time_UTC"].as<std::string>());
    useDrag = params["useDrag"].as<bool>();
    useSRP = params["useSRP"].as<bool>();
    
    semimajor_axis = params["semimajor_axis"].as<double>();
    eccentricity = params["eccentricity"].as<double>();
    inclination = params["inclination"].as<double>();
    RAAN = params["RAAN"].as<double>();
    AOP = params["AOP"].as<double>();
    true_anomaly = params["true_anomaly"].as<double>();
    initial_attitude = Eigen::Map<Vector4>(params["initial_attitude"].as<std::vector<double>>().data());
    initial_angular_rate = Eigen::Map<Vector3>(params["initial_angular_rate"].as<std::vector<double>>().data());
    
    initial_state = initializeSatellite(earliest_sim_start_unix);
}

/*
void Simulation_Parameters::getParamsFromFileAndSample(std::string filename) {
    
    // TODO : Disperse params loaded from init for MC

}
*/

VectorXd Simulation_Parameters::initializeSatellite(double epoch)
{
    VectorXd State(13+num_RWs);

    Vector6 KOE {semimajor_axis, eccentricity, inclination, RAAN, AOP, true_anomaly};
    Vector6 CartesianState = KOE2ECI(KOE, epoch);
    State(Eigen::seqN(0,6)) = CartesianState;
    State(Eigen::seqN(6,4)) = initial_attitude;
    State(Eigen::seqN(10,3)) = initial_angular_rate;
    State(Eigen::seqN(13, num_RWs)).setZero();

    return State;

}

Magnetorquer Simulation_Parameters::load_MTB(std::string filename)
{
    YAML::Node params = YAML::LoadFile(filename);

    num_MTBs = params["N_mtb"].as<int>();
    G_mtb_b = Eigen::Map<Eigen::MatrixXd>(params["mtb_orientation"].as<std::vector<double>>().data(), 3, num_MTBs);

    Magnetorquer magnetorquer = Magnetorquer(
                                    params["N_mtb"].as<int>(),
                                    params["max_voltage"].as<double>(),
                                    params["coils_per_layer"].as<double>(),
                                    params["layers"].as<double>(),
                                    params["trace_width"].as<double>(),
                                    params["gap_width"].as<double>(),
                                    params["trace_thickness"].as<double>(),
                                    params["max_power"].as<double>(),
                                    params["max_current_rating"].as<double>(),
                                    G_mtb_b
                                    );
    
    return magnetorquer;
}

#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysim_utils, m) {
    pybind11::class_<Simulation_Parameters>(m, "Simulation_Parameters")
        .def(pybind11::init<std::string>())
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
        .def_readonly("MAX_TIME", &Simulation_Parameters::MAX_TIME)
        .def_readonly("dt", &Simulation_Parameters::dt)
        .def_readonly("earliest_sim_start_unix", &Simulation_Parameters::earliest_sim_start_unix)
        .def_readonly("latest_sim_start_unix", &Simulation_Parameters::latest_sim_start_unix)
        .def_readonly("useDrag", &Simulation_Parameters::useDrag)
        .def_readonly("useSRP", &Simulation_Parameters::useSRP)
        //
        .def_readonly("initial_state", &Simulation_Parameters::initial_state);
}

#endif