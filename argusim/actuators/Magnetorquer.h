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
                           VectorXd mtb_Ahdt, VectorXd mtb_Adt, VectorXd mtb_Bhdt, VectorXd mtb_Bdt, 
                           std::vector<bool> status);
        /**
         * @brief Computes body-frame magnetic dipole moment of a single magnetorquer
         * 
         * @param index : index of the magnetorquer
         * @param current : current through the magnetorquer [UNITS: A]
         * @return dipole moment vector [UNITS: A*m^2]
         */
        Vector3 getSingleDipoleMoment(int index, double current);
        
        /**
         * @brief Computes Total magnetic dipole moment from all magnetorquers
         * 
         * @param currents : currents through each magnetorquer [UNITS: A]
         * @return Total dipole moment vector in body frame [UNITS: A*m^2]
         */
        Vector3 getDipoleMoment(VectorXd currents);
        
        /**
         * @brief Computes Torque on the body frame from input current of a single magnetorquer
         * 
         * @param index : index of the magnetorquer
         * @param current : current through the magnetorquer [UNITS: A]
         * @param magnetic_field_b : current magnetic field vector in body frame [UNITS: T]
         * @return torque due to the single magnetorquer on the satellite [UNITS: Nm]
         */
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
        
        /**
         * @brief Computes Torque on the body frame from input current of all magnetorquers
         * 
         * @param currents : currents for each magnetorquer [UNITS: A]
         * @param magnetic_field_b : current magnetic field vector in body frame [UNITS: T]
         * @return torque due to the single magnetorquer on the satellite [UNITS: Nm]
         */
        Vector3 getTorqueb(VectorXd currents, Vector3 magnetic_field_b);
        
        /**
         * @brief Computes time derivative of currents in each magnetorquer given input voltages
         * 
         * @param currents : currents through each magnetorquer [UNITS: A]
         * @param voltages : voltages for each magnetorquer [UNITS: V]
         * @return time derivative of currents through each magnetorquer [UNITS: A/s]
         */
        VectorXd getdidt(VectorXd currents, VectorXd voltages);
        
        /**
         * @brief Computes the current in each magnetorquer within each time step given input voltages
         * for the rk4 integrator (helps avoid instability issues if time step is too large wrt decay time)
         * 
         * @param voltages : voltages for each magnetorquer [UNITS: V]
         * @param currents : currents through each magnetorquer at the start of the time step [UNITS: A]
         * @param mode : "first", "half" or "full" to get the current at the start, half-way or end of the time step
         * @return currents through each magnetorquer at the specified time within the time step [UNITS: A]
         */
        VectorXd getCurrent(VectorXd voltages, VectorXd currents, std::string mode);
        
        /**
         * @brief Computes the magnetic field at the magnetometer due to the magnetorquers' dipole moments
         * 
         * @param currents : currents through each magnetorquer [UNITS: A]
         * @return magnetic field vector at the magnetometer due to the magnetorquers [UNITS: T]
         */
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
        std::vector<bool> working_status;
};


#endif   // C___magnetorquer_H