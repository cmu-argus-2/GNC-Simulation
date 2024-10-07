
import matplotlib.pyplot as plt

def plot_position(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["ECI_POS"]])
    plt.title('Position [m]')
    plt.xlabel('Time [s]')
    plt.ylabel('Position ECI')
    plt.show()

def plot_velocity(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["ECI_VEL"]])
    plt.title('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')
    plt.show()

def plot_quaternion(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["QUAT"]])
    plt.title('Quaternion')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.show()
    
    
def plot_angular_velocity(time, world_states, Idx):
    plt.figure()
    plt.plot(time, world_states[:, Idx["X"]["ANG_VEL"]] * (180 / 3.141592653589793))
    plt.title('Angular Velocity [deg/s]')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity')
    plt.show()


def plot_control_inputs(time, control_inputs, Idx):
    plt.figure()
    plt.plot(time, control_inputs)
    plt.title('Control Inputs')
    plt.xlabel('Time')
    plt.ylabel('Control Inputs')
    plt.show()


def main_plotter(results, Idx):
    plot_position(results["Time"], results["WorldStates"], Idx)
    plot_velocity(results["Time"], results["WorldStates"], Idx)
    plot_quaternion(results["Time"], results["WorldStates"], Idx)
    plot_angular_velocity(results["Time"], results["WorldStates"], Idx)
    plot_control_inputs(results["Time"], results["ControlInputs"], Idx)

# Example usage
# main_plotter(Results)
