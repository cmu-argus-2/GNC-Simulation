
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys 
import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_folder) == "visualization":
    sys.path.append("..") # Adds higher directory to python modules path.

from world.math.quaternions import quatrotation


def plot_position(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    plt.plot(time, world_states[:, Idx["X"]["ECI_POS"]][:, 0], label='X')
    plt.plot(time, world_states[:, Idx["X"]["ECI_POS"]][:, 1], label='Y')
    plt.plot(time, world_states[:, Idx["X"]["ECI_POS"]][:, 2], label='Z')
    plt.legend() 
    plt.title('Position [m]')
    plt.xlabel('Time [s]')
    plt.ylabel('Position ECI')
    plt.savefig('position_ECI.png')

def plot_velocity(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    plt.plot(time, world_states[:, Idx["X"]["ECI_VEL"]][:, 0], label='Vx')
    plt.plot(time, world_states[:, Idx["X"]["ECI_VEL"]][:, 1], label='Vy')
    plt.plot(time, world_states[:, Idx["X"]["ECI_VEL"]][:, 2], label='Vz')
    plt.legend() 
    plt.title('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')
    plt.savefig('velocity_ECI.png')

def plot_quaternion(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]][:, 0], label='q0')
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]][:, 1], label='q1')
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]][:, 2], label='q2')
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]][:, 3], label='q3')
    plt.legend() 
    plt.title('Quaternion')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.savefig('quaternion.png')
    
def plot_angular_velocity(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    angular_velocity = world_states[:, Idx["X"]["ANG_VEL"]] * 180/np.pi
    plt.plot(time, angular_velocity[:, 0], label='X')
    plt.plot(time, angular_velocity[:, 1], label='Y')
    plt.plot(time, angular_velocity[:, 2], label='Z')
    angular_velocity_norm = np.linalg.norm(angular_velocity, axis=1)
    plt.plot(time, angular_velocity_norm, label='L2 Norm of Angular Velocity')
    plt.legend()
    plt.axhline(y=3, color='r', linestyle='--')
    plt.title('Angular Velocity [deg/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity')
    plt.savefig('ang_velocity.png')
    

def plot_magnetorquer_power_consumption(time, power_consumption, Idx, fignum):
    plt.figure(num=fignum, figsize=(10, 8))

    # X-axis magnetorquers
    plt.subplot(3, 1, 1)
    plt.plot(time, power_consumption["MTB"][0, :], label='MTB X1')
    plt.plot(time, power_consumption["MTB"][3, :], label='MTB X2')
    plt.legend()
    plt.title('Magnetorquer Power Consumption - X Axis')
    plt.xlabel('Time [s]')
    plt.ylabel('Power Consumption')

    # Y-axis magnetorquers
    plt.subplot(3, 1, 2)
    plt.plot(time, power_consumption["MTB"][1, :], label='MTB Y1')
    plt.plot(time, power_consumption["MTB"][4, :], label='MTB Y2')
    plt.legend()
    plt.title('Magnetorquer Power Consumption - Y Axis')
    plt.xlabel('Time [s]')
    plt.ylabel('Power Consumption')

    # Z-axis magnetorquers
    plt.subplot(3, 1, 3)
    plt.plot(time, power_consumption["MTB"][2, :], label='MTB Z1')
    plt.plot(time, power_consumption["MTB"][5, :], label='MTB Z2')
    plt.legend()
    plt.title('Magnetorquer Power Consumption - Z Axis')
    plt.xlabel('Time [s]')
    plt.ylabel('Power Consumption')

    plt.tight_layout()
    plt.savefig('magnetorquer_power_consumption.png')

def plot_magnetorquer_inputs(time, control_inputs, Idx, fignum):
    plt.figure(num=fignum)
    plt.plot(time, control_inputs[Idx["U"]["MTB_TORQUE"]])
    plt.title('Magnetorquer Dipole Moment [Cm]')
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetorquer Inputs')
    plt.savefig('magnetorquer_inputs.png')
    # add upper limit on moment
    # add lower limit on moment
    # dip_moment =  
    
    
def plot_control_inputs(time, control_inputs, Idx, fignum):
    plt.figure(num=fignum)
    plt.plot(time, control_inputs)
    plt.title('Control Inputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Inputs')


def plot_sun_pointing_error(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    sun_vector = world_states[:, Idx["X"]["SUN_POS"]]
    quaternion = world_states[:, Idx["X"]["QUAT"]]
    sun_vec_rotated = np.zeros((len(time), 3))
    for i in range(len(time)):
        sun_vec_rotated[i] = quatrotation(quaternion[i]).T @ sun_vector[i] / np.linalg.norm(sun_vector[i])
    sun_pointing_error = np.zeros((len(time), 1))
    for i in range(len(time)):
        sun_pointing_error[i] = np.arccos(np.clip(np.dot(sun_vec_rotated[i], np.array([0.0,0.0,1.0])), -1.0, 1.0))
    plt.plot(time, np.rad2deg(sun_pointing_error))
    plt.title('Sun Pointing Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Sun Pointing Error [deg]')
    plt.savefig('sun_pointing_error.png')


def plot_sun_vector_ECI(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    sun_vector = world_states[:, Idx["X"]["SUN_POS"]]
    quaternion = world_states[:, Idx["X"]["QUAT"]]
    plt.plot(time, sun_vector[:, 0], label='X')
    plt.plot(time, sun_vector[:, 1], label='Y')
    plt.plot(time, sun_vector[:, 2], label='Z')
    plt.legend()
    plt.title('Sun Vector_ECI')
    plt.xlabel('Time [s]')
    plt.ylabel('Sun Vector [m]')
    plt.savefig('sun_vector_ECI.png')
    
    
def plot_nadir_pointing_error(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    nadir_vector = -world_states[:, Idx["X"]["ECI_POS"]]
    quaternion = world_states[:, Idx["X"]["QUAT"]]
    nadir_vec_rotated = np.zeros((len(time), 3))
    for i in range(len(time)):
        nadir_vec_rotated[i] = quatrotation(quaternion[i]).T @ nadir_vector[i] / np.linalg.norm(nadir_vector[i])
    nadir_pointing_error = np.zeros((len(time), 1))
    for i in range(len(time)):
        nadir_pointing_error[i] = np.arccos(np.clip(np.dot(nadir_vec_rotated[i], np.array([0.0,0.0,-1.0])), -1.0, 1.0))
    plt.plot(time, np.rad2deg(nadir_pointing_error))
    plt.title('Nadir Pointing Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Nadir Pointing Error [deg]')
    plt.savefig('nadir_pointing_error.png')

def plot_magnetic_field_ECI(time, world_states, Idx, fignum):
    plt.figure(num=fignum)
    magnetic_field = world_states[:, Idx["X"]["MAG_FIELD"]]
    plt.plot(time, magnetic_field[:, 0], label='X')
    plt.plot(time, magnetic_field[:, 1], label='Y')
    plt.plot(time, magnetic_field[:, 2], label='Z')
    plt.legend()
    plt.title('Magnetic Field ECI')
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetic Field [T]')
    plt.savefig('magnetic_field_ECI.png')

def main_plotter(results, Idx):
    fignum = 1
    plot_position(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_velocity(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_quaternion(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_angular_velocity(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_sun_vector_ECI(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_magnetic_field_ECI(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_control_inputs(results["Time"], results["ControlInputs"], Idx, fignum)
    fignum += 1
    plot_magnetorquer_power_consumption(results["Time"], results["PowerConsumption"], Idx, fignum)
    fignum += 1
    plot_sun_pointing_error(results["Time"], results["WorldStates"], Idx, fignum)
    fignum += 1
    plot_nadir_pointing_error(results["Time"], results["WorldStates"], Idx, fignum)
    plt.show()

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