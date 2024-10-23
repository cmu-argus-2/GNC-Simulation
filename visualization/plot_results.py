
import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_position(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["ECI_POS"]])
    plt.title('Position [m]')
    plt.xlabel('Time [s]')
    plt.ylabel('Position ECI')
    plt.show()
    plt.savefig('position_ECI.png')

def plot_velocity(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["ECI_VEL"]])
    plt.title('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')
    plt.show()
    plt.savefig('velocity_ECI.png')

def plot_quaternion(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]])
    plt.title('Quaternion')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.show()
    plt.savefig('quaternion.png')
    
def plot_angular_velocity(time, world_states, Idx):
    plt.figure()
    angular_velocity = world_states[:, Idx["X"]["ANG_VEL"]] * 180/np.pi
    plt.plot(time, angular_velocity[:, 0], label='X Angular Velocity')
    plt.plot(time, angular_velocity[:, 1], label='Y Angular Velocity')
    plt.plot(time, angular_velocity[:, 2], label='Z Angular Velocity')
    angular_velocity_norm = np.linalg.norm(angular_velocity, axis=1)
    plt.plot(time, angular_velocity_norm, label='L2 Norm of Angular Velocity')
    plt.legend()
    plt.axhline(y=3, color='r', linestyle='--')
    plt.title('Angular Velocity [deg/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity')
    plt.show()
    plt.savefig('ang_velocity.png')
    

def plot_magnetorquer_power_consumption(time, power_consumption, Idx):
    plt.figure()
    plt.plot(time, power_consumption["MTB"][0, :], label='MTB 1')
    plt.plot(time, power_consumption["MTB"][1, :], label='MTB 2')
    plt.plot(time, power_consumption["MTB"][2, :], label='MTB 3')
    plt.legend()
    plt.title('Magnetorquer Power Consumption')
    plt.xlabel('Time [s]')
    plt.ylabel('Power Consumption')
    plt.show()
    plt.savefig('magnetorquer_power_consumption.png')    

def plot_magnetorquer_inputs(time, control_inputs, Idx):
    plt.figure()
    plt.plot(time, control_inputs[Idx["U"]["MTB_TORQUE"]])
    plt.title('Magnetorquer Dipole Moment')
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetorquer Inputs')
    plt.show()
    plt.savefig('magnetorquer_inputs.png')
    # add upper limit on moment
    # add lower limit on moment
    # dip_moment =  
    
    
def plot_control_inputs(time, control_inputs, Idx):
    plt.figure()
    plt.plot(time, control_inputs)
    plt.title('Control Inputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Inputs')
    plt.show()


def main_plotter(results, Idx):
    
    plot_position(results["Time"], results["WorldStates"], Idx)
    plot_velocity(results["Time"], results["WorldStates"], Idx)
    plot_quaternion(results["Time"], results["WorldStates"], Idx)
    plot_angular_velocity(results["Time"], results["WorldStates"], Idx)
    plot_control_inputs(results["Time"], results["ControlInputs"], Idx)
    plot_magnetorquer_power_consumption(results["Time"], results["PowerConsumption"], Idx)
# Example usage
# main_plotter(Results)
if __name__ == "__main__":

    with open('./../Results.pkl', 'rb') as f:
        results = pickle.load(f)["Results"]

    Idx = dict()
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
    N_rw = 1
    N_mtb = 3
    Idx["NU"]    = N_rw + N_mtb
    Idx["N_rw"]  = N_rw
    Idx["N_mtb"] = N_mtb
    Idx["U"]  = dict()
    Idx["U"]["RW_TORQUE"]  = slice(0, N_rw)
    Idx["U"]["MTB_TORQUE"] = slice(N_rw, N_rw + N_mtb)
    # RW speed should be a state because it depends on the torque applied and needs to be propagated
    Idx["NX"] = Idx["NX"] + N_rw
    Idx["X"]["RW_SPEED"]   = slice(19, 19+N_rw)

    main_plotter(results, Idx)