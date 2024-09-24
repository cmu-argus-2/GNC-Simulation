# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch

# file imports
from world.physics.models.gravity import Gravity
from world.physics.models.drag import Drag
from world.physics.models.srp import SRP
from world.physics.models.magnetic_field import MagneticField

from world.math.integrators import RK4
from world.math.quaternions import quatrotation, crossproduct, hamiltonproduct

'''
    CLASS DYNAMICS
    Defines and stores the true world state (w*) over time
'''
class Dynamics():
    '''
        INIT
        1. config - instance of class Config containing simulation parameters for the current simulation
    '''
    def __init__(self, config) -> None:
        self.config = config

        # Update all brahe data files
        try:
            brahe.utils.download_iers_bulletin_ab(outdir=brahe.constants.DATA_PATH)
        except Exception:
            pass

        # Physics Models
        self.gravity = Gravity()
        self.drag = Drag(harris_priester_density_file='world/physics/data/harris_priester_density_data.csv')
        self.srp = SRP()
        self.magnetic_field = MagneticField()

        '''
        TRUE WORLD STATE (w*)
        [ECI position (3x1), 
         ECI velocity (3x1),
         Body-ECI quaternion (4x1),
         Body frame angular rates (3x1),
         ECI Sun position (3x1),
         ECI magnetic field (3x1)]
        
        TRUE WORLD TIME (t*)
        '''
        self.epc = Epoch(*brahe.time.mjd_to_caldate(self.config["mission"]["start_date"]))
        self.state = np.zeros((19,))

        # Orbital Position and Velocity
        self.state[0:6] = brahe.coordinates.sOSCtoCART(self.config["mission"]["initial_orbital_elements"], use_degrees=True)

        # Attitude and Quaternion Initialization
        self.state[6:10] = self.config["mission"]["initial_attitude"]
        self.state[10:13] = self.config["mission"]["initial_angular_rate"]

        # Sun Position
        self.state[13:16] = self.srp.sun_position(self.epc)

        # Magnetic Field Vector
        self.state[16:19] = self.magnetic_field.field(self.state[0:3], self.epc)

        # Actuator specific data
        self.G_rw_b = np.array(self.config["satellite"]["rw_orientation"]).T
        self.N_rw = self.G_rw_b.shape[1]
        self.I_rw = np.array(self.config["satellite"]["I_rw"])

        self.G_mtb_b = np.array(self.config["satellite"]["mtb_orientation"]).T

        self.I_sat = np.array(self.config["satellite"]["inertia"])


    '''
        FUNCTION UPDATE
        Updates the satellite state and stores the updated state within self.state

        INPUTS:
            1. Control vector as: [ω_rw ...., τ_rw, ..., τ_mtb]
        
        OUTPUTS
            None
    '''
    def update(self, input):
        
        self.state = RK4(self.state, input, self.state_derivative, self.config["solver"]["world_update_rate"])
        self.state[6:10] = self.state[6:10]/np.linalg.norm(self.state[6:10])
        self.state[13:16] = self.srp.sun_position(self.epc)
        self.state[16:19] = self.magnetic_field.field(self.state[0:3], self.epc)

        # print("jd:", self.epc.jd())
        self.epc = Epoch(*brahe.time.jd_to_caldate(self.epc.jd() + (1/self.config["solver"]["world_update_rate"])/(24*60*60)))
        # print("mjd:", self.epc.mjd())
    
    '''
        FUNCTION STATE_DERIVATIVE
        Forms the state derivative vector at the current timestep
        Having this as a separate function allows us to call RK4 on this function directly

        INPUTS:
            1. current state vector 
            2. Control vector as: [ω_rw ...., τ_rw, ..., τ_mtb]
        
        OUTPUTS:
            1. state derivative at current timestep
    '''
    def state_derivative(self, state, input):
        wdot = np.zeros_like(self.state)

        wdot[0:3] = state[3:6] # rdot = v

        acceleration = self.gravity.acceleration(state[0:3], self.epc)
        if self.config["complexity"]["use_drag"]:
            acceleration = acceleration + self.drag.acceleration(state[0:3], state[3:6], state[6:10], self.epc, self.config["satellite"])
        if self.config["complexity"]["use_srp"]:
            acceleration = acceleration + self.srp.acceleration(state[0:3], state[6:10], self.epc, self.config["satellite"])
        
        wdot[3:6] = acceleration
        
        # ATTITUDE DYNAMICS
        wdot[6:10] = 0.5*hamiltonproduct(np.insert(state[10:13], 0, 0), state[6:10])

        R_BtoECI = quatrotation(state[6:10])

        # Reaction wheels
        G_rw = R_BtoECI@self.G_rw_b  # RW orientation matrix in ECI
        h_rw = self.I_rw*(input[0:self.N_rw] + G_rw.T@state[10:13])
        tau_rw = G_rw@input[self.N_rw:2*self.N_rw]

        # Magnetorquers
        tau_mtb = crossproduct(state[16:19])@R_BtoECI@self.G_mtb_b@input[2*self.N_rw:]
        
        # Attitude Dynamics equation
        wdot[10:13] = np.linalg.inv(self.I_sat)@(-np.cross(state[10:13], G_rw@h_rw) - tau_rw - tau_mtb)

        return wdot
        