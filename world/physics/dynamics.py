# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch

# file imports
from world.physics.index import State, Control
from world.physics.models.gravity import Gravity
from world.physics.models.drag import Drag
from world.physics.models.srp import SRP
from world.physics.models.magnetic_field import MagneticField
from actuators.magnetorquer import Magnetorquer
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

    def __init__(self, config) -> None:
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

        """
        TRUE WORLD STATE (w*)
        [
         ECI position (3x1) [UNITS: m], 
         ECI velocity (3x1) [UNITS: m/s],
         Body->ECI quaternion (4x1) [Order : [scalar, vector]],
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

        # Orbital Position and Velocity
        self.state[State.TRANS] = brahe.coordinates.sOSCtoCART(
            self.config["mission"]["initial_orbital_elements"], use_degrees=True
        )

        # Attitude and Quaternion Initialization
        self.state[State.QUAT] = self.config["mission"]["initial_attitude"]
        self.state[State.ANG_VEL] = self.config["mission"]["initial_angular_rate"]

        # Sun Position
        self.state[State.SUN_POS] = self.srp.sun_position(self.epc)

        # Magnetic Field Vector
        self.state[State.MAG_FIELD] = self.magnetic_field.field(self.state[State.ECI_POS], self.epc)

        # Actuator specific data
        
        self.Magnetorquers = [Magnetorquer(self.config, IdMtb) for IdMtb in range(self.config["satellite"]["N_mtb"])]
        self.ReactionWheels = [ReactionWheel(self.config, IdRw) for IdRw in range(self.config["satellite"]["N_rw"])]
        
        self.I_sat = np.array(self.config["satellite"]["inertia"])
        self.I_sat_inv = np.linalg.inv(self.I_sat)
        
        # Actuator Indexing
        N_rw  = self.config["satellite"]["N_rw"]
        N_mtb = self.config["satellite"]["N_mtb"]
        # Extend state to include RW speed
        self.state = np.concatenate((self.state, np.zeros(N_rw)))

        # Assign initial RW speed values
        self.state[State.RW_SPEED] = self.config["mission"]["initial_rw_speed"]
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
        self.state[State.QUAT] = self.state[State.QUAT] / np.linalg.norm(self.state[State.QUAT])
        self.state[State.SUN_POS] = self.srp.sun_position(self.epc)
        self.state[State.MAG_FIELD] = self.magnetic_field.field(self.state[State.ECI_POS], self.epc)

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

        wdot[State.ECI_POS] = state[State.ECI_VEL]  # rdot = v

        acceleration = self.gravity.acceleration(state[State.ECI_POS], self.epc)
        if self.config["complexity"]["use_drag"]:
            acceleration = acceleration + self.drag.acceleration(
                state[State.ECI_POS], state[State.ECI_VEL],
                state[State.QUAT], self.epc, self.config["satellite"]
            )
        if self.config["complexity"]["use_srp"]:
            acceleration = acceleration + self.srp.acceleration(
                state[State.ECI_POS], state[State.QUAT], self.epc, self.config["satellite"]
            )

        wdot[State.ECI_VEL] = acceleration

        # ATTITUDE DYNAMICS
        wdot[State.QUAT] = 0.5 * hamiltonproduct(np.insert(state[State.ANG_VEL], 0, 0), state[State.QUAT])

        # Reaction wheels
        tau_rw = np.zeros(Control.N_RW) # torque per RW (RW axis)
        for i, rw in enumerate(self.ReactionWheels):
            tau_rw += rw.get_applied_torque(input[Control.RW_TORQUE][i])
        G_rw = np.zeros((3, len(self.ReactionWheels)))
        for i, rw in enumerate(self.ReactionWheels):
            G_rw[:, i] = rw.G_rw_b
        I_rw = np.array([rw.I_rw for rw in self.ReactionWheels]).reshape(-1, 1)
        h_rw   = I_rw * state[State.RW_SPEED]

        # Magnetorquers
        tau_mtb = np.zeros(3)
        Re2b = quatrotation(state[State.QUAT]).T
        bfMAG_FIELD = Re2b @ self.state[State.MAG_FIELD]  # convert ECI MAG FIELD 2 body frame
        for i, mtb in enumerate(self.Magnetorquers):
            tau_mtb += crossproduct(input[Control.MTB_TORQUE][i] * mtb.G_mtb_b) @ bfMAG_FIELD
            # mtb.get_torque(input[Control.MTB_TORQUE][i], bfMAG_FIELD)
        # self.Magnetorquer.get_torque(self, state, Idx)

        # Attitude Dynamics equation
        h_sc = self.I_sat @ state[State.ANG_VEL]
        wdot[State.ANG_VEL] = (self.I_sat_inv @ (
            -np.cross(state[State.ANG_VEL], (G_rw @ h_rw + h_sc.reshape(-1, 1)).T).T + (G_rw @ tau_rw.reshape(-1, 1)) + tau_mtb.reshape(-1, 1)
        )).flatten()

        # RW speed dynamics:
        wdot[State.RW_SPEED] = tau_rw / I_rw

        return wdot
