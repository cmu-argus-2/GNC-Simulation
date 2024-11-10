#ifndef C___magnetorquer_H
#define C___magnetorquer_H

#include "math/EigenWrapper.h"

double COPPER_RESISTIVITY = 1.724e-8;
class Magnetorquer {
    public:
        Magnetorquer(int N_MTBs, VectorXd maxVolt, VectorXd coilsPerLayer, VectorXd layers, VectorXd traceThickness,
                     VectorXd pcb_side_max, VectorXd traceWidth, VectorXd gapWidth, VectorXd maxPower, VectorXd maxCurrentRating,
                     MatrixXd mtb_orientation);

        /**
        * @brief Computes Torque on the body frame from input current 
        * 
        * @param voltages : voltages for each magnetorquer [UNITS: A]
        * @param q : satellite attitude quaternion representing a rotation from Body to ECI frames
        * @param magnetic_field : current magnetic field vector in ECI
        * @return torque due to the single magnetorquer on the satellite [UNITS: Nm]
        */
        Vector3 getTorque(VectorXd voltages, Quaternion q, Vector3 magnetic_field);

    private:
        int num_MTBs; 
        VectorXd max_voltage;
        VectorXd N; // coils per layer
        VectorXd pcb_layers; // Number of layers
        VectorXd N_per_face; 
        VectorXd trace_thickness;
        VectorXd pcb_side_max;
        VectorXd trace_width;
        VectorXd gap_width;
        VectorXd coil_width;
        VectorXd coil_length;
        VectorXd max_power;
        VectorXd max_current_rating;
        VectorXd A_cross;
        VectorXd resistance;
        VectorXd max_dipole_moment;
        MatrixXd G_mtb_b;
};


#endif   // C___magnetorquer_H