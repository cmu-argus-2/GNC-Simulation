#include "Magnetorquer.h"

#include "math/EigenWrapper.h"
#include <cmath>
#include <utility>
#include <iostream>

Magnetorquer::Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double Across, double Nturns,
                           double maxVolt, double maxCurrentRating, double maxPower, MatrixXd mtb_orientation)  
{
    num_MTBs = N_MTBs;
    resistance = mtb_resistance;
    A_cross = Across;
    N_turns = Nturns;
    max_voltage = maxVolt;
    max_current_rating = maxCurrentRating;
    max_power = maxPower;
   
    G_mtb_b = std::move(mtb_orientation);
}

Vector3 Magnetorquer::getTorque(VectorXd voltages, Quaternion q, Vector3 magnetic_field)
{
    
    Vector3 magnetic_field_b = q.toRotationMatrix().transpose()*magnetic_field; // quaternion rotates vector in body frame to ECI. We need the inverse rotation

    Eigen::VectorXd currents = voltages.cwiseProduct(resistance.cwiseInverse());
    Eigen::VectorXd power = voltages.cwiseProduct(currents);

    Eigen::VectorXd max_current = currents.cwiseAbs();
    Eigen::VectorXd max_applied_voltage = voltages.cwiseAbs();
    Eigen::VectorXd max_applied_power = power.cwiseAbs();

    // for (int i=0; i<num_MTBs; i++) {
    //     assert(max_current(i) <= max_current_rating);
    //     assert(max_applied_voltage(i) <= max_voltage);
    //     assert(max_applied_power(i) <= max_power);
    // }

    //auto dipole_moments = N_per_face*A_cross*
    Vector3 dipole_moment;
    MatrixXd torque = MatrixXd::Zero(3, num_MTBs);
    for (int i=0; i<num_MTBs; i++) {
        dipole_moment = N_turns*currents(i)*A_cross*G_mtb_b.col(i);
        torque.col(i) = dipole_moment.cross(magnetic_field_b);
    }
    Vector3 torque_net = torque.rowwise().sum();

    return torque_net;
}