
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "ParameterParser.h"
#include "SRP.h"

#include <cmath>
#include <functional>
#include <random>
#include <iostream>

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- POWER CONSUMPTION --------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
   VectorXd PowerConsumption(const VectorXd state, const VectorXd control_input, Simulation_Parameters sc)
   {
       /* Magnetorquer Power Consumption */ 
       VectorXd mtb_power = MagnetorquerPower(control_input, sc);

       /* Reaction Wheel Power Consumption */
   
       /* Solar power generation */
       VectorXd solar_power = SolarPanels(state, sc);
       
       /* Static Power Consumption */
       double static_power_draw = sc.mb_power + control_input(sc.num_MTBs+sc.num_RWs)*sc.jetson_power;
   
       /* Get Battery State */
       double net_power_draw = mtb_power.sum() + static_power_draw - solar_power.sum();
       double solar_heat = (1-sc.solar_panel_efficiency)/sc.solar_panel_efficiency*solar_power.sum();
       
       VectorXd new_state = Battery(state, sc, net_power_draw, solar_heat);
   
   
       return new_state;
   
   }
   
   VectorXd MagnetorquerPower(const VectorXd control_input, Simulation_Parameters sc)
   {
       VectorXd power_consumption = control_input(sc.u_idx_map["mtb_volt"].to_seq()).array() * control_input(sc.u_idx_map["mtb_volt"].to_seq()).array() / sc.resistances.array();
       return power_consumption;
   }

   // [TODO:] add reaction wheel power consumption
   
   VectorXd SolarPanels(const VectorXd state, Simulation_Parameters sc)
   {
       // Quaternion quat {state(6), state(7), state(8), state(9)};
       Quaternion quat = state(sc.x_idx_map["quaternion"].to_seq());
       Vector3 r_eci = state(sc.x_idx_map["position"].to_seq());
   
       // True sun position
       Vector3 sun_pos_eci = state(sc.x_idx_map["sun_position"].to_seq());
       Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation
   
       // Shadow Factor
       double shadow = shadow_factor(r_eci, sun_pos_eci);
   
       // Compute solar power
       VectorXd solar_power = shadow*NOMINAL_SOLAR_INTENSITY*sc.G_sp_b.transpose()*sc.solar_panel_efficiency*sc.solar_panel_area*sun_pos_body/sun_pos_body.norm();
       solar_power = (solar_power.array() < 0.0).select(0, solar_power);
   
       return solar_power;
   }
   
   VectorXd Battery(const VectorXd state, Simulation_Parameters sc, double net_power_consumption, double solar_heat)
   {
       VectorXd battery_readings = VectorXd::Zero(8);
       VectorXd new_state = state;

       double power_consumed = net_power_consumption;
       new_state(sc.x_idx_map["battery_soc"].to_seq()) -= 100*power_consumed/sc.battery_capacity; // Change in SoC
       new_state(sc.x_idx_map["battery_soc"].to_seq()) = fmax(0,fmin(100, new_state(sc.x_idx_map["battery_soc"].to_seq())));
       new_state(sc.x_idx_map["battery_current"].to_seq()) = -power_consumed/new_state(sc.x_idx_map["battery_voltage"].to_seq()); // Current in A
       new_state(c.x_idx_map["battery_temp"].to_seq()) += (solar_heat*sc.solar_heat_factor + 
                                      pow(net_power_consumption/sc.max_pack_voltage,2)*sc.battery_internal_resistance - 
                                      sc.battery_radiative_loss*pow(new_state(c.x_idx_map["battery_temp"].to_seq()),4))/sc.battery_thermal_mass;
   
       return new_state;
   }
   