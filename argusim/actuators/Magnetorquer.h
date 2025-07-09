#ifndef C___magnetorquer_H
#define C___magnetorquer_H

#include "math/EigenWrapper.h"
#include <string>

double COPPER_RESISTIVITY = 1.724e-8;
class Magnetorquer {
    public:
        Magnetorquer(int N_MTBs, VectorXd mtb_resistance, double A_cross, double N_turns,
                           double maxVolt, double maxCurrentRating, double maxPower, 
                           VectorXd mtb_inductance, MatrixXd mag_mtb_sens_mat, MatrixXd mtb_orientation,
                           VectorXd mtb_Ahdt, VectorXd mtb_Adt, VectorXd mtb_Bhdt, VectorXd mtb_Bdt);

        Vector3 getSingleDipoleMoment(int index, double current);

        Vector3 getDipoleMoment(VectorXd currents);
        
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

        VectorXd getVoltageOrCurrent(VectorXd voltages, VectorXd currents, std::string mode);

        Vector3 getMagneticFieldAtMagnetometer(VectorXd currents);

    private:
        int num_MTBs; 
        double A_cross;
        double N_turns;
        double max_voltage;
        double max_power;
        double max_current_rating;
        VectorXd resistance;
        VectorXd inductance;
        VectorXd Ahdt;
        VectorXd Bhdt;
        VectorXd Adt;
        VectorXd Bdt;
        MatrixXd mag_mtb_sens; 
        MatrixXd G_mtb_b;
};


#endif   // C___magnetorquer_H