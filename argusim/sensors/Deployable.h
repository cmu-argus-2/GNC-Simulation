
#ifndef C___deployable_H
#define C___deployable_H

#include "math/EigenWrapper.h"
#include <string>
#include <vector>
#include <iostream>

class Deployable {
    public:
        Deployable();
        Deployable(int N_deployables, VectorXd dep_masses, MatrixXd dep_inertia,
            MatrixXd dep_com_stowed, MatrixXd dep_orient_stowed,
            MatrixXd dep_com_deployed, MatrixXd dep_orient_deployed,
            std::vector<bool> dep_status, int N_deploy_sensors,
            std::vector<bool> sensed_dep);
        
        /**
         * @brief Returns the deployment sensor readings
         * 
         * @return deployment sensor readings
         */
        VectorXd getDeploymentSensorReadings();
    private:
        int num_deployables;
        VectorXd deployable_masses; // [Kg]
        MatrixXd deployable_inertia; // [kg.m^2]
        MatrixXd deployable_com_stowed; // [m]
        MatrixXd deployable_orient_stowed; // [deg]
        MatrixXd deployable_com_deployed; // [m]
        MatrixXd deployable_orient_deployed; // [deg]
        std::vector<bool> deployable_status; // true if deployed, false if stowed
        int num_deploy_sensors;
        std::vector<bool> sensed_deployable; // true if it has sensor, false if not
};


#endif   // C___deployable_H