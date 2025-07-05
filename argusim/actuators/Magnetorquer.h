#ifndef C___magnetorquer_H
#define C___magnetorquer_H

#include "math/EigenWrapper.h"

double COPPER_RESISTIVITY = 1.724e-8;
class Magnetorquer {
    public:
        Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double A_cross, double N_turns,
                           double maxVolt, double maxCurrentRating, double maxPower, 
                           VectorXd mtb_inductance, MatrixXd mtb_orientation);

        Vector3 getSingleDipoleMoment(int index, double current);
        
        Vector3 getSingleTorque(int index, double current, Vector3 magnetic_field_b);

        /**
        * @brief Computes Torque on the body frame from input current 
        * 
        * @param voltages : voltages for each magnetorquer [UNITS: A]
        * @param q : satellite attitude quaternion representing a rotation from Body to ECI frames
        * @param magnetic_field : current magnetic field vector in ECI
        * @return torque due to the single magnetorquer on the satellite [UNITS: Nm]
        */
        Vector3 getTorque(VectorXd currents, Quaternion q, Vector3 magnetic_field);
        
        Vector3 getTorqueb(VectorXd currents, Vector3 magnetic_field_b);
        
        VectorXd getdidt(VectorXd currents, VectorXd voltages);

        VectorXd getVoltageOrCurrent(VectorXd voltages, VectorXd currents);

    private:
        int num_MTBs; 
        double A_cross;
        double N_turns;
        double max_voltage;
        double max_power;
        double max_current_rating;
        VectorXd resistance;
        VectorXd inductance;
        MatrixXd G_mtb_b;
};


#endif   // C___magnetorquer_H