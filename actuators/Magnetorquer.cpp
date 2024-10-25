#include "Magnetorquer.h"

#include "math/EigenWrapper.h"
#include <cmath>


Magnetorquer::Magnetorquer(int N_MTBs, double maxVolt, double coilsPerLayer, double layers, double traceThickness,
                           double traceWidth, double gapWidth, double maxPower, double maxCurrentRating,
                           MatrixXd mtb_orientation) 
{
    num_MTBs = N_MTBs;
    max_voltage = maxVolt;
    N = coilsPerLayer;
    pcb_layers = layers;
    N_per_face = N*pcb_layers;
    trace_thickness = traceThickness;
    trace_width = traceWidth;
    gap_width = gapWidth;
    coil_width = trace_width + gap_width;
    max_power = maxPower;
    max_current_rating = maxCurrentRating;

    pcb_side_max = 0.1;
    A_cross = pow(pcb_side_max - N*coil_width, 2.0);
    
    double coil_length = 4*(pcb_side_max - N*coil_width)*N*pcb_layers;
    resistance = COPPER_RESISTIVITY*coil_length/(trace_width*trace_thickness);
    G_mtb_b = mtb_orientation;
}

Vector3 Magnetorquer::getTorque(VectorXd voltages, Quaternion q, Vector3 magnetic_field)
{
    
    Vector3 magnetic_field_b = q.toRotationMatrix().transpose()*magnetic_field; // quaternion rotates vector in body frame to ECI. We need the inverse rotation

    auto currents = voltages/resistance;
    auto power = voltages.cwiseProduct(currents);

    double max_current = currents.lpNorm<Eigen::Infinity>(); // infinity norm
    double max_applied_voltage = voltages.lpNorm<Eigen::Infinity>();
    double max_applied_power = power.lpNorm<Eigen::Infinity>();

    assert(max_current <= max_current_rating);
    assert(max_applied_voltage <= max_voltage);
    assert(max_applied_power <= max_power);

    //auto dipole_moments = N_per_face*A_cross*
    Vector3 dipole_moment;
    MatrixXd torque = MatrixXd::Zero(3, num_MTBs);
    for (int i=0; i<num_MTBs; i++) {
        dipole_moment = N_per_face*currents(i)*A_cross*G_mtb_b.col(i);
        torque.col(i) = dipole_moment.cross(magnetic_field_b);
    }

    return torque.rowwise().sum();
}