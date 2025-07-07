#include "power.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "SRP.h"

#include <cmath>
#include <functional>
#include <random>
#include <iostream>

#define NOMINAL_SOLAR_INTENSITY 1373 // W/m^2


VectorXd PowerDynamics(const VectorXd state, const VectorXd control_input, std::unordered_map<std::string, SliceDef> u_idx_map, 
                            MatrixXd G_sp_b, double solar_panel_efficiency, double solar_panel_area, std::unordered_map<std::string, SliceDef> x_idx_map, double battery_capacity, 
                            double battery_thermal_mass, double battery_radiative_loss, double solar_heat_factor, double max_pack_voltage, double battery_internal_resistance,
                            double mb_power, int num_MTBs, int num_RWs, double jetson_power)
{
    VectorXd xdot = VectorXd::Zero(x_idx_map["battery"].get_length());

    VectorXd power_vector = PowerConsumption(state, control_input, u_idx_map, G_sp_b, solar_panel_efficiency, 
                                            solar_panel_area, x_idx_map, 
                                            mb_power, num_MTBs, num_RWs, jetson_power);
    
    double net_power_draw = power_vector.sum();
    
    int num_panels = G_sp_b.cols();
    VectorXd solar_power = -power_vector(Eigen::seqN(num_MTBs, num_panels));

    double solar_heat = (1-solar_panel_efficiency)/solar_panel_efficiency*solar_power.sum();

    int idx_bat_soc = x_idx_map["battery_soc"].to_idx() - x_idx_map["battery"].get_start();
    int idx_bat_temp = x_idx_map["battery_temp"].to_idx() - x_idx_map["battery"].get_start();
    
    xdot(idx_bat_soc) = -100 * net_power_draw / battery_capacity; // Change in SoC
    xdot(idx_bat_temp) = (solar_heat * solar_heat_factor + 
                        pow(net_power_draw / max_pack_voltage, 2) * battery_internal_resistance - 
                        battery_radiative_loss * pow(state(x_idx_map["battery_temp"].to_idx()), 4)) / battery_thermal_mass;
    return xdot;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- POWER CONSUMPTION --------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */

   VectorXd PowerConsumption(const VectorXd state, const VectorXd control_input, std::unordered_map<std::string, SliceDef> u_idx_map, 
                            MatrixXd G_sp_b, double solar_panel_efficiency, double solar_panel_area, std::unordered_map<std::string, SliceDef> x_idx_map, 
                            double mb_power, int num_MTBs, int num_RWs, double jetson_power)
   {
       /* Magnetorquer Power Consumption */ 
       VectorXd currents = state(x_idx_map["mtb_currents"].to_seq());
       VectorXd mtb_power = MagnetorquerPower(control_input, currents, u_idx_map);

       /* Reaction Wheel Power Consumption */
   
       /* Solar power generation */
       VectorXd solar_power = SolarPanels(state, G_sp_b, solar_panel_efficiency, solar_panel_area, x_idx_map);
       
       /* Static Power Consumption */
       double static_power_draw = mb_power + control_input(num_MTBs+num_RWs)*jetson_power;


       VectorXd power_consumption = VectorXd::Zero(mtb_power.size() + solar_power.size() + 1);
       power_consumption(Eigen::seqN(0, mtb_power.size())) = mtb_power; // Magnetorquer power consumption
       power_consumption(Eigen::seqN(mtb_power.size(), solar_power.size())) = -solar_power; // Solar power generation
       power_consumption(power_consumption.size()-1) = static_power_draw; // Static power consumption
       return power_consumption;
   }

   VectorXd MagnetorquerPower(const VectorXd control_input, VectorXd currents, std::unordered_map<std::string, SliceDef> u_idx_map)
   { // VectorXd resistances, 
       // VectorXd power_consumption = control_input(u_idx_map["mtb_volt"].to_seq()).array() * control_input(u_idx_map["mtb_volt"].to_seq()).array() / resistances.array();
       VectorXd power_consumption = control_input(u_idx_map["mtb_volt"].to_seq()).array() * currents.array();
       return power_consumption;
   }

   // [TODO:] add reaction wheel power consumption
   
   VectorXd SolarPanels(const VectorXd state, MatrixXd G_sp_b, double solar_panel_efficiency, double solar_panel_area,
                            std::unordered_map<std::string, SliceDef> x_idx_map)
   {
       // Quaternion quat {state(6), state(7), state(8), state(9)};
       Quaternion quat = vectorToQuaternion(state(x_idx_map["quaternion"].to_seq()));
       Vector3 r_eci = state(x_idx_map["position"].to_seq());
   
       // True sun position
       Vector3 sun_pos_eci = state(x_idx_map["sun_position"].to_seq());
       Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation
   
       // Shadow Factor
       double shadow = shadow_factor(r_eci, sun_pos_eci);
   
       // Compute solar power
       VectorXd solar_power = shadow*NOMINAL_SOLAR_INTENSITY*G_sp_b.transpose()*solar_panel_efficiency*solar_panel_area*sun_pos_body/sun_pos_body.norm();
       solar_power = (solar_power.array() < 0.0).select(0, solar_power);
   
       return solar_power;
   }
   
   VectorXd Battery(const VectorXd state, const VectorXd control_input, std::unordered_map<std::string, SliceDef> u_idx_map, 
                            MatrixXd G_sp_b, double solar_panel_efficiency, double solar_panel_area, std::unordered_map<std::string, SliceDef> x_idx_map, 
                            double mb_power, int num_MTBs, int num_RWs, double jetson_power)
    {
        VectorXd new_state       = state(x_idx_map["battery"].to_seq()); // Copy the battery state
        int idx_bat_soc          = x_idx_map["battery_soc"].to_idx() - x_idx_map["battery"].get_start();
        int idx_bat_volt      = x_idx_map["battery_voltage"].to_idx() - x_idx_map["battery"].get_start();
        int idx_bat_cur          = x_idx_map["battery_current"].to_idx() - x_idx_map["battery"].get_start();
        double net_power_consumption = PowerConsumption(state, control_input, u_idx_map, G_sp_b, solar_panel_efficiency, 
                                                        solar_panel_area, x_idx_map, mb_power, num_MTBs, num_RWs, jetson_power).sum();
        new_state(idx_bat_cur)   = -net_power_consumption / new_state(idx_bat_volt); // Current in A
        new_state(idx_bat_soc) = fmax(0, fmin(100, new_state(idx_bat_soc))); // Enforce SoC limit
        return new_state;
    }