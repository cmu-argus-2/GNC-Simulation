from actuators.magnetorquer import Magnetorquer
from actuators.reaction_wheels import ReactionWheel
from FSW.controllers.controller import Controller

def initialize_controller(pyparams):
        Idx = {}
        # Intialize the dynamics class as the "world"
        Idx["NX"] = 19
        Idx["X"]  = dict()
        Idx["X"]["ECI_POS"]   = slice(0, 3)
        Idx["X"]["ECI_VEL"]   = slice(3, 6)
        Idx["X"]["TRANS"]     = slice(0, 6)
        Idx["X"]["QUAT"]      = slice(6, 10)
        Idx["X"]["ANG_VEL"]   = slice(10, 13)
        Idx["X"]["ROT"]       = slice(6, 13)
        Idx["X"]["SUN_POS"]   = slice(13, 16)
        Idx["X"]["MAG_FIELD"] = slice(16, 19)

        # Actuator specific data
        # self.ReactionWheels = [ReactionWheel(self.config, IdRw) for IdRw in range(self.config["satellite"]["N_rw"])]

        # Actuator Indexing
        N_rw  = pyparams["N_rw"]
        N_mtb = pyparams["N_mtb"]
        Idx["NU"]    = N_rw + N_mtb
        Idx["N_rw"]  = N_rw
        Idx["N_mtb"] = N_mtb
        Idx["U"]  = dict()
        Idx["U"]["MTB_TORQUE"]  = slice(0, N_mtb)
        Idx["U"]["RW_TORQUE"] = slice(N_mtb, N_rw + N_mtb)
        # RW speed should be a state because it depends on the torque applied and needs to be propagated
        Idx["NX"] = Idx["NX"] + N_rw
        Idx["X"]["RW_SPEED"]   = slice(19, 19 + N_rw)
        Magnetorquers = [Magnetorquer(pyparams, IdMtb) for IdMtb in range(N_mtb)] 
        ReactionWheels = [ReactionWheel(pyparams, IdRw) for IdRw in range(N_rw)]
        controller = Controller(pyparams, Magnetorquers, ReactionWheels, Idx)
        return Idx, controller
    