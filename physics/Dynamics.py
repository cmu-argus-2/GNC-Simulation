# build-in imports
import numpy as np
import quaternion

# function imports
from physics.models.Gravity import Gravity
from physics.models.AtmosphericDrag import AtmosphericDrag
from physics.models.SolarRadiationPressure import SolarRadiationPressure

from physics.integrators.RK4 import RK4


class Dynamics():
    def __init__(self, sim_params) -> None:
        
        # Pre-sim setup
        self.sim_params = sim_params
        self.gravity = Gravity(self.sim_params)
        self.drag = AtmosphericDrag(self.sim_params)
        self.srp = SolarRadiationPressure(self.sim_params)
                
        # Step Logging
        self.time = None
        self.state = None
        self.control_inputs = None
        self.spherical_gravitational_acceleration = None
        self.J2_perturbation_acceleration = None
        self.drag_acceleration = None
        self.srp_acceleration = None   
        
        # Sim initialization
        self.initialize_sat()   
        
    
    '''
        FUNCTION STEP
        Steps through the simulation by one timestep and returns the updated vector
    '''
    def step(self, moments):
        state_update = RK4(self.state, [0,0,0], self.state_transition, self.sim_params)
        
        #Quaternion update
        state_update[6:10] = state_update[6:10]/np.linalg.norm(state_update[6:10])
        
        self.state = state_update
        self.control_input = moments
        self.time += self.sim_params.solver_timestep
        
        return (self.state, self.time)       
        
    '''
        Function INITIALIZE_SAT
        Populates the intial state vector for the satellite
        State vector x = [ECI velocity, ECI position, Body frame quaternion, ]
        
        NOTE: quaternions should be co-ordinatized in the body frame representing a rotation to ECI
    '''
    def initialize_sat(self):
        # Preliminary calculations
        a = self.sim_params.semi_major_axis
        e = self.sim_params.eccentricity
        i = self.sim_params.inclination*np.pi/180
        Omega = self.sim_params.RAAN*np.pi/180
        omega = self.sim_params.AOP*np.pi/180
        nu = self.sim_params.true_anomaly*np.pi/180
        mu = self.sim_params.mu
        
        p = 1000*a*(1-e**2)
        
        R_Omega = np.array([[np.cos(Omega), -np.sin(Omega), 0], [np.sin(Omega), np.cos(Omega), 0], [0,0,1]])
        R_i = np.array([[1,0,0],[0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
        R_omega = np.array([[np.cos(omega), -np.sin(omega), 0], [np.sin(omega), np.cos(omega), 0], [0,0,1]])
        
        
        # Compute ECI position in m and velocity in m/s
        position = np.squeeze(R_Omega @ R_i @ R_omega @ np.array([p*np.cos(nu)/(1 + e*np.cos(nu)), p*np.sin(nu)/(1 + e*np.cos(nu)), 0]).reshape((-1,1)))
        velocity = np.squeeze(R_Omega @ R_i @ R_omega @ np.array([-np.sqrt(mu/p)*np.sin(nu), np.sqrt(mu/p)*(e + np.cos(nu)), 0]).reshape(-1,1))
        
        # Compute Quaternions
        q = np.array([0.5,1,1,1])/np.sqrt(3.25)
        omega = np.array([0.1,0,0])
        
        self.state = np.concatenate((position, velocity, q, omega))
        self.time = 0
    
    '''
        Models the variation in orbital position due to:
            1. gravity
            2. atmospheric drag
            3. solar radiation
        
        Models the change in spacecraft attitude due to:
            1. Control moments
        
        NOTE: attitude is modeled as a quaternion between ECI and body frame
        NOTE 2: Does not include the effects of a reaction wheel yet 
        
        INPUTS:
            1. state vector : contains [ECI position, ECI velocity, Body->ECI quaternion, Body->ECI quaternion rate]
            2. M - Control moments in body frame
        
        OUTPUTS: xdot of the state vector
    '''
    def state_transition(self, state, M):
        
        # Orbital Dynamics
        r = state[0:3] # positions (km) in ECI
        v = state[3:6] # velocities (m/s) in ECI
        
        self.spherical_gravitational_acceleration, self.J2_perturbation_acceleration = self.gravity.acceleration(r)
        self.drag_acceleration = self.drag.acceleration(r, v)
        self.srp_acceleration = self.srp.acceleration()
        
        a = self.spherical_gravitational_acceleration + self.J2_perturbation_acceleration + self.drag_acceleration + self.srp_acceleration
        
        # Attitude Dynamics
        M = np.array(M)
        
        q = np.quaternion(*state[6:10])
        qdot = np.quaternion(*state[10:14])
        
        I_sat = np.array(self.sim_params.inertia_tensor)
        
        omega_quat = 2*qdot*q # quaternion representation
        omega = quaternion.as_float_array(omega_quat)[1:] # remove the scalar term
        
        omega_dot = np.linalg.inv(I_sat)@M - np.linalg.inv(I_sat)@(np.cross(omega, I_sat@omega))
        
        return np.concatenate((v, a, quaternion.as_float_array(qdot), omega_dot))
        
    
        
        
        