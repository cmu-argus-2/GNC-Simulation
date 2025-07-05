#include "Magnetorquer.h"

#include "math/EigenWrapper.h"
#include <cmath>
#include <utility>
#include <iostream>

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif

Magnetorquer::Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double Across, double Nturns,
                           double maxVolt, double maxCurrentRating, double maxPower, VectorXd mtb_inductance,
                           MatrixXd mtb_orientation)  
{
    num_MTBs = N_MTBs;
    resistance = mtb_resistance;
    A_cross = Across;
    N_turns = Nturns;
    max_voltage = maxVolt;
    max_current_rating = maxCurrentRating;
    max_power = maxPower;
    inductance = mtb_inductance;
   
    G_mtb_b = std::move(mtb_orientation);
}


Vector3 Magnetorquer::getSingleDipoleMoment(int index, double current)
{
    // double current = voltage / resistance(index);
    Vector3 dipole_moment = N_turns * current * A_cross * G_mtb_b.col(index);
    return dipole_moment;
}

Vector3 Magnetorquer::getSingleTorque(int index, double current, Vector3 magnetic_field_b)
{
    Vector3 dipole_moment = getSingleDipoleMoment(index, current);
    Vector3 torque = dipole_moment.cross(magnetic_field_b);
    return torque;
}

Vector3 Magnetorquer::getTorque(VectorXd currents, Quaternion q, Vector3 magnetic_field)
{
    
    Vector3 magnetic_field_b = q.toRotationMatrix().transpose()*magnetic_field; // quaternion rotates vector in body frame to ECI. We need the inverse rotation

    Vector3 torque_net = getTorqueb(currents, magnetic_field_b);

    return torque_net;
}

Vector3 Magnetorquer::getTorqueb(VectorXd currents, Vector3 magnetic_field_b)
{
    
    //auto dipole_moments = N_per_face*A_cross*
    MatrixXd torque = MatrixXd::Zero(3, num_MTBs);
    for (int i = 0; i < num_MTBs; i++) {
        torque.col(i) = getSingleTorque(i, currents(i), magnetic_field_b);
    }
    Vector3 torque_net = torque.rowwise().sum();

    return torque_net;
}

VectorXd Magnetorquer::getdidt(VectorXd currents, VectorXd voltages)
{
    VectorXd didt = VectorXd::Zero(num_MTBs);
    const double epsilon = 1e-8; // threshold for "close to zero"
    for (int i = 0; i < num_MTBs; i++) {
        if (std::abs(inductance(i)) < epsilon) {
            didt(i) = 0.0;
        } else {
            didt(i) = (voltages(i) - currents(i) * resistance(i)) / inductance(i);
        }
    }
    return didt;
}

VectorXd Magnetorquer::getVoltageOrCurrent(VectorXd voltages, VectorXd currents) {
    VectorXd result = VectorXd::Zero(num_MTBs);
    const double epsilon = 1e-8; // threshold for "close to zero"
    for (int i = 0; i < num_MTBs; i++) {
        if (std::abs(inductance(i)) < epsilon) {
            result(i) = voltages(i) / resistance(i);
        } else {
            result(i) = currents(i);
        }
    }
    return result;
}

#ifdef USE_PYBIND_TO_COMPILE
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pymagnetorquers, m) {
    py::class_<Magnetorquer>(m, "Magnetorquer")
        .def(py::init<int, VectorXd, double, double, double, double, double, VectorXd, MatrixXd>())
        .def("getTorque", &Magnetorquer::getTorque, "Compute and return the net torque produced by the magnetorquers given input currents, attitude quaternion, and magnetic field vector.")
        .def("getTorqueb", &Magnetorquer::getTorqueb, "Compute and return the net torque produced by the magnetorquers given input currents, and magnetic field vector in the body frame.")
        .def("getSingleDipoleMoment", &Magnetorquer::getSingleDipoleMoment, "Compute the dipole moment for a single magnetorquer given its index and current.")
        .def("getSingleTorque", &Magnetorquer::getSingleTorque, "Compute the torque for a single magnetorquer given its index, current, attitude quaternion, and magnetic field vector.")
        .def("getdidt", &Magnetorquer::getdidt, "Compute the time derivative of the currents based on voltages and inductance.")
        .def("getVoltageOrCurrent", &Magnetorquer::getVoltageOrCurrent, "Get the voltage or current for each magnetorquer based on the input voltages and currents.");
}
#endif