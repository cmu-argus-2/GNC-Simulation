import numpy as np
from astropy.time import Time
from Basilisk import __path__
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, unitTestSupport, simIncludeRW
from Basilisk.simulation import spacecraft, magneticFieldWMM, msisAtmosphere, MtbEffector, reactionWheelStateEffector
from Basilisk.utilities.RigidBodyKinematics import EP2MRP, MRP2EP
from Basilisk.architecture import messaging

class BasiliskSim():
    def __init__(self, config) -> None:
        self.config = config

        # Define simulation
        self.sim = SimulationBaseClass.SimBaseClass()
        self.sim.SetProgressBar(True)
        self.dynamics = self.sim.CreateNewProcess("dynamics")
        self.simTimeStep = macros.sec2nano(1/self.config["solver"]["world_update_rate"])

        # Define Time
        self.date = Time(self.config["mission"]["start_date"], format='mjd').iso
        self.epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(self.date)

        # Orbit Tasks
        self.dynamics.addTask(self.sim.CreateNewTask("orbit", self.simTimeStep))

        # Attitude Tasks
        self.dynamics.addTask(self.sim.CreateNewTask("attitude", self.simTimeStep))

        # Define satellite
        self.load_satellite()

        # Define physics models
        self.gravity()
        self.atmosphere_model()
        self.sun_position()
        self.magnetic_field()

        # Define Actuators
        self.magnetorquers()
        self.reaction_wheels()
        self.N_mtb = len(self.config["satellite"]["mtb_orientation"])

        # Initialize Satellite
        self.initialize_satellite()

        # Initialize Sim
        self.sim.InitializeSimulation()
        self.sim.ConfigureStopTime(macros.sec2nano(100))

    def update(self, input):
        # Create MTB dipole moments
        mtb_input = input[0:self.N_mtb]
        MTBCmd = messaging.MTBCmdMsgPayload()
        MTBCmd.mtbDipoleCmds = mtb_input
        MTB_cmd = messaging.MTBCmdMsg().write(MTBCmd)
        self.MTB.mtbCmdInMsg.subscribeTo(MTB_cmd)
        
        # Create RW motor torques
        RW_torque = messaging.ArrayMotorTorqueMsgPayload()
        RW_torque.motorTorque = input[self.N_mtb:]
        RW_cmd = messaging.ArrayMotorTorqueMsg().write(RW_torque)
        self.RW.rwMotorCmdInMsg.subscribeTo(RW_cmd)

        self.sim.TotalSim.SingleStepProcesses()
        return self.pack_state()
    
    def load_satellite(self):
        self.satellite = spacecraft.Spacecraft()
        self.satellite.ModelTag = "Argus2"
        self.sim.AddModelToTask("orbit", self.satellite)
        self.satellite.hub.mHub = config["satellite"]["mass"]
        self.satellite.hub.r_BcB_B = [[0.0], [0.0], [0.0]] # CoM location in body frame
        self.satellite.hub.IHubPntBc_B = config["satellite"]["inertia"]

    def gravity(self):
        # Define Gravity Body
        self.gravity_model = simIncludeGravBody.gravBodyFactory()
        self.gravity_model.createEarth()
        self.gravity_model.gravBodies['earth'].isCentralBody = True
        self.gravity_model.addBodiesTo(self.satellite)

    def atmosphere_model(self):
        self.atmosphere = msisAtmosphere.MsisAtmosphere()
        self.atmosphere.ModelTag = "MSIS_Atmosphere"
        self.atmosphere.epochInMsg.subscribeTo(self.epochMsg)
        ap = 8
        f107 = 110
        sw_msg = {
            "ap_24_0": ap, "ap_3_0": ap, "ap_3_-3": ap, "ap_3_-6": ap, "ap_3_-9": ap,
            "ap_3_-12": ap, "ap_3_-15": ap, "ap_3_-18": ap, "ap_3_-21": ap, "ap_3_-24": ap,
            "ap_3_-27": ap, "ap_3_-30": ap, "ap_3_-33": ap, "ap_3_-36": ap, "ap_3_-39": ap,
            "ap_3_-42": ap, "ap_3_-45": ap, "ap_3_-48": ap, "ap_3_-51": ap, "ap_3_-54": ap,
            "ap_3_-57": ap, "f107_1944_0": f107, "f107_24_-24": f107
        }

        swMsgList = []
        for c, val in enumerate(sw_msg.values()):
            swMsgData = messaging.SwDataMsgPayload()
            swMsgData.dataValue = val
            swMsgList.append(messaging.SwDataMsg().write(swMsgData))
            self.atmosphere.swDataInMsgs[c].subscribeTo(swMsgList[-1])
        
        
        self.sim.AddModelToTask("orbit", self.atmosphere)
        self.atmosphere.addSpacecraftToModel(self.satellite.scStateOutMsg)

    def sun_position(self):
        sun_model = simIncludeGravBody.gravBodyFactory()
        sun_model.createSun()
        sun_model.gravBodies['sun'].isCentralBody = False
        self.sun = sun_model.createSpiceInterface(time = self.date)
        self.sim.AddModelToTask("orbit", self.sun, -1)

    def magnetic_field(self):
        self.magneticField = magneticFieldWMM.MagneticFieldWMM()
        self.magneticField.ModelTag = "WMM"
        self.magneticField.dataPath = __path__[0] + '/supportData/MagneticField/'
        self.magneticField.epochInMsg.subscribeTo(self.epochMsg)
        self.magneticField.addSpacecraftToModel(self.satellite.scStateOutMsg)
        self.sim.AddModelToTask("orbit", self.magneticField)

    def magnetorquers(self):
        self.MTB = MtbEffector.MtbEffector()
        self.MTB.ModelTag = "MtbEff"
        self.satellite.addDynamicEffector(self.MTB)
        self.sim.AddModelToTask("orbit", self.MTB)

        MTB_layout = self.config["satellite"]["mtb_orientation"]
        MTB_config = messaging.MTBArrayConfigMsgPayload()
        MTB_config.numMTB = len(MTB_layout)
        MTB_config.GtMatrix_B = np.array(MTB_layout).flatten().tolist()
        MTB_config.maxMtDipoles = [self.config["satellite"]["mtb_max_dipole"]]*MTB_config.numMTB

        MTB_params_msg = messaging.MTBArrayConfigMsg().write(MTB_config)

        MTB_cmd = messaging.MTBCmdMsgPayload()
        MTB_cmd.mtbDipoleCmds = [0.0]*MTB_config.numMTB
        MTB_cmd_msg = messaging.MTBCmdMsg().write(MTB_cmd)

        self.MTB.mtbParamsInMsg.subscribeTo(MTB_params_msg)
        self.MTB.mtbCmdInMsg.subscribeTo(MTB_cmd_msg)
        self.MTB.magInMsg.subscribeTo(self.magneticField.envOutMsgs[0])

    def reaction_wheels(self):
        RW_set = simIncludeRW.rwFactory()
        RW_layout = config["satellite"]["rw_orientation"]
        for n in range(len(RW_layout)):
            RW_set.create("custom", RW_layout[n], useRWfriction = False, Js = self.config["satellite"]["I_rw"], Omega_max = float(self.config["satellite"]["rw_omega_max"]), useMaxTorque = False)
        
        self.RW = reactionWheelStateEffector.ReactionWheelStateEffector()
        RW_set.addToSpacecraft("ReactionWheels", self.RW, self.satellite)

        RW_torque = messaging.ArrayMotorTorqueMsgPayload()
        RW_torque.motorTorque = [0.0]*len(RW_layout)
        RW_cmd = messaging.ArrayMotorTorqueMsg().write(RW_torque)
        self.RW.rwMotorCmdInMsg.subscribeTo(RW_cmd)


    def initialize_satellite(self):
        initial_oe = self.config["mission"]["initial_orbital_elements"]
        oe = orbitalMotion.ClassicElements()
        oe.a = initial_oe[0]
        oe.e = initial_oe[1]
        oe.i = initial_oe[2]*np.pi/180
        oe.Omega = initial_oe[3]*np.pi/180
        oe.omega = initial_oe[4]*np.pi/180
        oe.f = initial_oe[5]*np.pi/180
        self.satellite.hub.r_CN_NInit, self.satellite.hub.v_CN_NInit = orbitalMotion.elem2rv(self.gravity_model.gravBodies['earth'].mu, oe)
        self.satellite.hub.sigma_BNInit = EP2MRP(self.config["mission"]["initial_attitude"])
        self.satellite.hub.omega_BN_BInit = self.config["mission"]["initial_angular_rate"]

    def pack_state(self):
        state = np.zeros((19,))
        
        # Parse Data from sim objects
        sat_state = self.satellite.scStateOutMsg.read()
        sun_pos = self.sun.planetStateOutMsgs[0].read().PositionVector
        mag_field = self.magneticField.envOutMsgs[0].read().magField_N

        print(self.RW.rwSpeedOutMsg.read().wheelSpeeds)

        state[0:3] = sat_state.r_CN_N
        state[3:6] = sat_state.v_CN_N
        state[6:10] = MRP2EP(sat_state.sigma_BN)
        state[10:13] = sat_state.omega_BN_B
        state[13:16] = sun_pos
        state[16:19] = mag_field

        return state

import yaml
config_file = "config.yaml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
d = BasiliskSim(config)
r = np.zeros((10,))
input = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
for i in range(100):
    print(d.update(input))


