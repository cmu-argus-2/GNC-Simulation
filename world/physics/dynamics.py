# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch

# file imports
from world.physics.models.gravity import Gravity
from world.physics.models.drag import Drag
from world.physics.models.srp import SRP
from world.physics.models.magnetic_field import MagneticField
from actuators.magnetorquer import Magnetorquers
from actuators.reaction_wheels import ReactionWheel
from world.math.integrators import RK4
from world.math.quaternions import quatrotation, crossproduct, hamiltonproduct

"""
    CLASS DYNAMICS
    Defines and stores the true world state (w*) over time
"""


class Dynamics:
    """
    INIT
    1. config - instance of class Config containing simulation parameters for the current simulation
    """

    def __init__(self, config, Idx) -> None:
        self.config = config

        # Update all brahe data files
        try:
            brahe.utils.download_iers_bulletin_ab(outdir=brahe.constants.DATA_PATH)
        except Exception:
            pass

        # Physics Models
        self.gravity = Gravity()
        self.drag = Drag(
            harris_priester_density_file="world/physics/data/harris_priester_density_data.csv"
        )
        self.srp = SRP()
        self.magnetic_field = MagneticField()

        self.Idx = Idx
        """
        TRUE WORLD STATE (w*)
        [
         ECI position (3x1) [UNITS: m], 
         ECI velocity (3x1) [UNITS: m/s],
         Body-ECI quaternion (4x1) [Order : [scalar, vector]],
         Body frame angular rates (3x1) [UNITS: rad/s],
         ECI Sun position (3x1) [UNITS: m],
         ECI magnetic field (3x1) [UNITS: T]
        ]
        
        TRUE WORLD TIME (t*)
        """
        self.epc = Epoch(
            *brahe.time.mjd_to_caldate(self.config["mission"]["start_date"])
        )
        self.state = np.zeros((19,))
        
        self.Idx["NX"] = 19
        self.Idx["X"]  = dict()
        self.Idx["X"]["ECI_POS"]   = slice(0, 3)
        self.Idx["X"]["ECI_VEL"]   = slice(3, 6)
        self.Idx["X"]["TRANS"]     = slice(0, 6)
        self.Idx["X"]["QUAT"]      = slice(6, 10)
        self.Idx["X"]["ANG_VEL"]   = slice(10, 13)
        self.Idx["X"]["ROT"]       = slice(6, 13)
        self.Idx["X"]["SUN_POS"]   = slice(13, 16)
        self.Idx["X"]["MAG_FIELD"] = slice(16, 19)
        
        # Orbital Position and Velocity
        self.state[self.Idx["X"]["TRANS"]] = brahe.coordinates.sOSCtoCART(
            self.config["mission"]["initial_orbital_elements"], use_degrees=True
        )

        # Attitude and Quaternion Initialization
        self.state[self.Idx["X"]["QUAT"]] = self.config["mission"]["initial_attitude"]
        self.state[self.Idx["X"]["ANG_VEL"]] = self.config["mission"]["initial_angular_rate"]

        # Sun Position
        self.state[self.Idx["X"]["SUN_POS"]] = self.srp.sun_position(self.epc)

        # Magnetic Field Vector
        self.state[self.Idx["X"]["MAG_FIELD"]] = self.magnetic_field.field(self.state[self.Idx["X"]["ECI_POS"]], self.epc)
    
        # Actuator specific data
        self.Magnetorquers  = Magnetorquers(self.config)
        self.ReactionWheels = ReactionWheel(self.config)
        
        self.I_sat = np.array(self.config["satellite"]["inertia"])
        self.I_sat_inv = np.linalg.inv(self.I_sat)
        
        # Actuator Indexing
        N_rw  = self.config["satellite"]["N_rw"]
        N_mtb = self.config["satellite"]["N_mtb"]
        self.Idx["NU"] = N_rw + N_mtb
        self.Idx["U"]  = dict()
        self.Idx["U"]["RW_TORQUE"]  = slice(0, N_rw)
        self.Idx["U"]["MTB_TORQUE"] = slice(N_rw, N_rw + N_mtb)
        # RW speed should be a state because it depends on the torque applied and needs to be propagated
        self.Idx["X"]["RW_SPEED"]   = slice(19, 19+N_rw)
        # Extend state to include RW speed
        self.state = np.concatenate((self.state, np.zeros(N_rw)))

        # Assign initial RW speed values
        self.state[self.Idx["X"]["RW_SPEED"]] = self.config["mission"]["initial_rw_speed"]
        # Measurement Indexing
        # self.Idx["NY"]
        # self.Idx["Y"]

    """
        FUNCTION UPDATE
        Updates the satellite state and stores the updated state within self.state

        INPUTS:
            1. Control vector as: [ω_rw ...., τ_rw, ..., τ_mtb]
        
        OUTPUTS
            None
    """

    def update(self, input):
        self.state = RK4(
            self.state,
            input,
            self.state_derivative,
            1.0/self.config["solver"]["world_update_rate"],
        )
        self.state[self.Idx["X"]["QUAT"]] = self.state[self.Idx["X"]["QUAT"]] / np.linalg.norm(self.state[self.Idx["X"]["QUAT"]])
        self.state[self.Idx["X"]["SUN_POS"]] = self.srp.sun_position(self.epc)
        self.state[self.Idx["X"]["MAG_FIELD"]] = self.magnetic_field.field(self.state[self.Idx["X"]["ECI_POS"]], self.epc)

        self.epc = Epoch(
            *brahe.time.jd_to_caldate(
                self.epc.jd()
                + (1 / self.config["solver"]["world_update_rate"]) / (24 * 60 * 60)
            )
        )

    """
        FUNCTION STATE_DERIVATIVE
        Forms the state derivative vector at the current timestep
        Having this as a separate function allows us to call RK4 on this function directly

        INPUTS:
            1. current state vector 
            2. Control vector as: [ω_rw ...., τ_rw, ..., τ_mtb, ...]
        
        OUTPUTS:
            1. state derivative at current timestep
    """

    def state_derivative(self, state, input):
        wdot = np.zeros_like(self.state)

        wdot[self.Idx["X"]["ECI_POS"]] = state[self.Idx["X"]["ECI_VEL"]]  # rdot = v

        acceleration = self.gravity.acceleration(state[self.Idx["X"]["ECI_POS"]], self.epc)
        if self.config["complexity"]["use_drag"]:
            acceleration = acceleration + self.drag.acceleration(
                state[self.Idx["X"]["ECI_POS"]], state[self.Idx["X"]["ECI_VEL"]], 
                state[self.Idx["X"]["QUAT"]], self.epc, self.config["satellite"]
            )
        if self.config["complexity"]["use_srp"]:
            acceleration = acceleration + self.srp.acceleration(
                state[self.Idx["X"]["ECI_POS"]], state[self.Idx["X"]["QUAT"]], self.epc, self.config["satellite"]
            )

        wdot[self.Idx["X"]["ECI_VEL"]] = acceleration

        # ATTITUDE DYNAMICS
        wdot[self.Idx["X"]["QUAT"]] = 0.5 * hamiltonproduct(np.insert(state[self.Idx["X"]["ANG_VEL"]], 0, 0), state[self.Idx["X"]["QUAT"]])

        # Reaction wheels
        tau_rw = self.ReactionWheels.get_applied_torque(input[self.Idx["U"]["RW_TORQUE"]])
        G_rw   = self.ReactionWheels.G_rw_b
        I_rw   = self.ReactionWheels.I_rw
        h_rw   = I_rw * (state[self.Idx["X"]["RW_SPEED"]] + G_rw.T @ state[self.Idx["X"]["ANG_VEL"]])
        
        # Magnetorquers
        tau_mtb = self.Magnetorquers.get_applied_torque(input[self.Idx["U"]["MTB_TORQUE"]], state, self.Idx)
        
        # Attitude Dynamics equation
        h_sc  = self.I_sat @ state[self.Idx["X"]["ANG_VEL"]]
        wdot[self.Idx["X"]["ANG_VEL"]] = (self.I_sat_inv @ (
            -np.cross(state[self.Idx["X"]["ANG_VEL"]], (G_rw @ h_rw + h_sc)).reshape(-1, 1) + (G_rw * tau_rw) + tau_mtb.reshape(-1, 1)
        )).flatten()
        
        # RW speed dynamics:
        wdot[self.Idx["X"]["RW_SPEED"]] = -tau_rw / I_rw

        return wdot
