#ifndef C___magnetorquer_H
#define C___magnetorquer_H

#include "math/EigenWrapper.h"

double COPPER_RESISTIVITY = 1.724e-8;
class Magnetorquer {
    public:
        Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double A_cross, double N_turns,
                           double maxVolt, double maxCurrentRating, double maxPower, MatrixXd mtb_orientation);

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
        double A_cross;
        double N_turns;
        double max_voltage;
        double max_power;
        double max_current_rating;
        VectorXd resistance;
        MatrixXd G_mtb_b;
};


#endif   // C___magnetorquer_H