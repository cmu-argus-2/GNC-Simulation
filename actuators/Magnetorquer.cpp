#include "Magnetorquer.h"

#include "math/EigenWrapper.h"
#include <cmath>
#include <utility>
#include <iostream>

Magnetorquer::Magnetorquer(int N_MTBs, VectorXd maxVolt, VectorXd coilsPerLayer, VectorXd layers, VectorXd traceThickness,
                           VectorXd pcbSideMax, VectorXd traceWidth, VectorXd gapWidth, VectorXd maxPower, VectorXd maxCurrentRating,
                           MatrixXd mtb_orientation) : num_MTBs(N_MTBs), max_voltage(std::move(maxVolt)), N(std::move(coilsPerLayer)), pcb_layers(std::move(layers)), 
                           trace_thickness(std::move(traceThickness)), pcb_side_max(std::move(pcbSideMax)),
                           trace_width(std::move(traceWidth)), gap_width(std::move(gapWidth)), max_power(std::move(maxPower)), 
                           max_current_rating(std::move(maxCurrentRating)) 
{
    
    coil_width = trace_width + gap_width;
    
    N_per_face = VectorXd::Zero(num_MTBs);
    A_cross = VectorXd::Zero(num_MTBs);
    coil_length = VectorXd::Zero(num_MTBs);
    resistance = VectorXd::Zero(num_MTBs);
    max_dipole_moment = VectorXd::Zero(num_MTBs);
    
    for (int i=0; i<num_MTBs; i++) {
        N_per_face(i) = N(i)*pcb_layers(i);
        max_current_rating(i) = std::min(max_power(i) / max_voltage(i), max_current_rating(i));
        A_cross(i) = pow(pcb_side_max(i) - N(i)*coil_width(i), 2.0);
            
        coil_length(i) = 4*(pcb_side_max(i) - N(i)*coil_width(i))*N(i)*pcb_layers(i);
        resistance(i) = COPPER_RESISTIVITY*coil_length(i)/(trace_width(i)*trace_thickness(i));
            
        max_current_rating(i) = std::min(max_current_rating(i), sqrt(max_power(i) / resistance(i)) );
        max_current_rating(i) = std::min(max_current_rating(i), max_voltage(i) / resistance(i));
        max_voltage(i) = resistance(i) * max_current_rating(i);
        max_power(i)   = resistance(i) * pow(max_current_rating(i), 2);
        max_dipole_moment(i) = N_per_face(i) * max_current_rating(i) * A_cross(i);
    }
   
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

    for (int i=0; i<num_MTBs; i++) {
        assert(max_current(i) <= max_current_rating(i));
        assert(max_applied_voltage(i) <= max_voltage(i));
        assert(max_applied_power(i) <= max_power(i));
    }

    //auto dipole_moments = N_per_face*A_cross*
    Vector3 dipole_moment;
    MatrixXd torque = MatrixXd::Zero(3, num_MTBs);
    for (int i=0; i<num_MTBs; i++) {
        dipole_moment = N_per_face(i)*currents(i)*A_cross(i)*G_mtb_b.col(i);
        torque.col(i) = dipole_moment.cross(magnetic_field_b);
    }

    return torque.rowwise().sum();
}