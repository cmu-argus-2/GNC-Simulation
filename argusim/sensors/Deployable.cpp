#include "Deployable.h"

#include "math/EigenWrapper.h"
#include <cmath>
#include <utility>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#pragma GCC diagnostic pop
#endif


// VL53L4CD Time-of-Flight Distance Sensor model
const double MAX_DISTANCE_SENSOR = 1200; // [mm]
const double MIN_DISTANCE_SENSOR = 1;    // [mm]


Deployable::Deployable() 
{
    num_deployables = 0;
    deployable_masses = VectorXd::Zero(0); // [Kg]
    deployable_inertia = MatrixXd::Zero(0,0); // [kg.m^2]
    deployable_com_stowed = MatrixXd::Zero(0,0); // [m]
    deployable_orient_stowed = MatrixXd::Zero(0,0); // [deg]
    deployable_com_deployed = MatrixXd::Zero(0,0); // [m]
    deployable_orient_deployed = MatrixXd::Zero(0,0); // [deg]
    deployable_status = std::vector<bool>();
    num_deploy_sensors = 0;
    sensed_deployable = std::vector<bool>(); // true if it has sensor, false if not
}

Deployable::Deployable(int N_deployables, VectorXd dep_masses, MatrixXd dep_inertia, MatrixXd dep_com_stowed, 
                        MatrixXd dep_orient_stowed, MatrixXd dep_com_deployed, MatrixXd dep_orient_deployed,
                        std::vector<bool> dep_status, int N_deploy_sensors, std::vector<bool> sensed_dep) 
{
    num_deployables = N_deployables;
    deployable_masses = dep_masses; // [Kg]
    deployable_inertia = std::move(dep_inertia); // [kg.m^2]
    deployable_com_stowed = std::move(dep_com_stowed); // [m]
    deployable_orient_stowed = std::move(dep_orient_stowed); // [deg]
    deployable_com_deployed = std::move(dep_com_deployed); // [m]
    deployable_orient_deployed = std::move(dep_orient_deployed); // [deg]
    deployable_status = std::move(dep_status);
    num_deploy_sensors = N_deploy_sensors;
    sensed_deployable = std::move(sensed_dep); // true if it has sensor, false if not
}

VectorXd Deployable::getDeploymentSensorReadings()
{
    VectorXd dep_sensor_readings = VectorXd::Zero(num_deploy_sensors);
    int sensor_idx = 0;
    for (int i = 0; i < num_deployables; ++i) {
        if (sensed_deployable[i]) {
            dep_sensor_readings(sensor_idx) = deployable_status[i] ? MAX_DISTANCE_SENSOR : MIN_DISTANCE_SENSOR;
            sensor_idx++;
        }
    }
    return dep_sensor_readings;
}


#ifdef USE_PYBIND_TO_COMPILE
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pydeployables, m) {
    py::class_<Deployable>(m, "Deployable")
        .def(py::init<int, VectorXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, std::vector<bool>, int, std::vector<bool>>())
        .def("getDeploymentSensorReadings", &Deployable::getDeploymentSensorReadings, "Return the deployment sensor reading.");
 }
#endif