#include "Magnetorquer.h"

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

Magnetorquer::Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double Across, double Nturns,
                           double maxVolt, double maxCurrentRating, double maxPower, VectorXd mtb_inductance,
                           MatrixXd mag_mtb_sens_mat, MatrixXd mtb_orientation, VectorXd mtb_Ahdt, 
                           VectorXd mtb_Adt, VectorXd mtb_Bhdt, VectorXd mtb_Bdt, std::vector<bool> status) 
{
    num_MTBs = N_MTBs;
    resistance = mtb_resistance;
    A_cross = Across;
    N_turns = Nturns;
    max_voltage = maxVolt;
    max_current_rating = maxCurrentRating;
    max_power = maxPower;
    inductance = mtb_inductance;
    mag_mtb_sens = std::move(mag_mtb_sens_mat);
    G_mtb_b = std::move(mtb_orientation);
    Ahdt = mtb_Ahdt;
    Bhdt = mtb_Bhdt;
    Adt = mtb_Adt;
    Bdt = mtb_Bdt;
    working_status = std::move(status);
}


Vector3 Magnetorquer::getSingleDipoleMoment(int index, double current)
{
    if (working_status[index] == false) {
        return Vector3::Zero(); // no dipole moment from failed magnetorquers
    }
    // double current = voltage / resistance(index);
    Vector3 dipole_moment = N_turns * current * A_cross * G_mtb_b.col(index);
    return dipole_moment;
}

Vector3 Magnetorquer::getDipoleMoment(VectorXd currents) {
    Vector3 dipole_moment = Vector3::Zero();
    for (int i = 0; i < num_MTBs; i++) {
        if (working_status[i] == false) {
            continue; // skip failed magnetorquers
        }
        dipole_moment += getSingleDipoleMoment(i, currents(i));
    }
    return dipole_moment;
}

Vector3 Magnetorquer::getSingleTorque(int index, double current, Vector3 magnetic_field_b)
{
    if (working_status[index] == false) {
        return Vector3::Zero(); // no torque from failed magnetorquers
    }
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
        if (working_status[i] == false) {
            continue; // skip failed magnetorquers
        }
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
        if (working_status[i] == false) {
            didt(i) = 0.0;
            continue; // skip failed magnetorquers
        }
        if (std::abs(inductance(i)) < epsilon) {
            didt(i) = 0.0;
        } else {
            didt(i) = (voltages(i) - currents(i) * resistance(i)) / inductance(i);
        }
    }
    return didt;
}

VectorXd Magnetorquer::getCurrent(VectorXd voltages, VectorXd currents, std::string mode) {
    VectorXd result = VectorXd::Zero(num_MTBs);
    const double epsilon = 1e-8; // threshold for "close to zero"
    for (int i = 0; i < num_MTBs; i++) {
        if (working_status[i] == false) {
            result(i) = 0.0;
            continue; // skip failed magnetorquers
        }
        if (std::abs(inductance(i)) < epsilon) {
            result(i) = voltages(i) / resistance(i);
        } else {
            if (mode == "first") {
                result(i) = currents(i);
            } else if (mode == "half") {
                result(i) = Ahdt(i) * currents(i) + Bhdt(i) * voltages(i);
            } else if (mode == "full") {
                result(i) = Adt(i) * currents(i) + Bdt(i) * voltages(i);
            } else {
                throw std::invalid_argument("Invalid mode specified. Use 'first', 'half', or 'full'.");
            }
        }
    }
    return result;
}

Vector3 Magnetorquer::getMagneticFieldAtMagnetometer(VectorXd currents) {
    // Get the total dipole moment
    Vector3 dipole_moment = getDipoleMoment(currents);
    Vector3 mtb_B_effect = mag_mtb_sens * dipole_moment;
    return mtb_B_effect;
}

#ifdef USE_PYBIND_TO_COMPILE
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pymagnetorquers, m) {
    py::class_<Magnetorquer>(m, "Magnetorquer")
        .def(py::init<int, VectorXd, double, double, double, double, double, VectorXd, MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, VectorXd, std::vector<bool>>())
        .def("getTorque", &Magnetorquer::getTorque, "Compute and return the net torque produced by the magnetorquers given input currents, attitude quaternion, and magnetic field vector.")
        .def("getTorqueb", &Magnetorquer::getTorqueb, "Compute and return the net torque produced by the magnetorquers given input currents, and magnetic field vector in the body frame.")
        .def("getSingleDipoleMoment", &Magnetorquer::getSingleDipoleMoment, "Compute the dipole moment for a single magnetorquer given its index and current.")
        .def("getDipoleMoment", &Magnetorquer::getDipoleMoment, "Compute the total dipole moment from all magnetorquers given their currents.")
        .def("getSingleTorque", &Magnetorquer::getSingleTorque, "Compute the torque for a single magnetorquer given its index, current, attitude quaternion, and magnetic field vector.")
        .def("getdidt", &Magnetorquer::getdidt, "Compute the time derivative of the currents based on voltages and inductance.")
        .def("getCurrent", &Magnetorquer::getCurrent, "Get the voltage or current for each magnetorquer based on the input voltages and currents.")
        .def("getMagneticFieldAtMagnetometer", &Magnetorquer::getMagneticFieldAtMagnetometer, "Compute the magnetic field at the magnetometer due to the magnetorquers' dipole moments.");
}
#endif