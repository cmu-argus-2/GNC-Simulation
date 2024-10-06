import numpy as np


class ReactionWheel:
    def __init__(self, config):
        """
        Initialize the ReactionWheel with maximum torque and speed.

        :param max_torque: Maximum torque the reaction wheel can provide (Nm)
        :param max_speed: Maximum speed the reaction wheel can spin (rad/s)
        """
        self.G_rw_b = np.array(config["satellite"]["rw_orientation"]).T
        self.N_rw = self.G_rw_b.shape[1]
        self.I_rw = np.array(config["satellite"]["I_rw"])
        self.max_torque = config["satellite"]["max_torque"]
        self.max_speed = config["satellite"]["max_speed"]

    def get_applied_torque(self, actuator_cmd):
        """
        Get the torque applied by the reaction wheel based on the actuator command.

        :param actuator_cmd: Command to the reaction wheel (Nm)
        :return: Torque applied by the reaction wheel (Nm)
        """
        # Torque saturation
        output_torque = np.clip(actuator_cmd, -self.max_torque, self.max_torque)
        # Wheel Speed saturation
        # ...

        return output_torque

    # [UNUSED]: Have RW speed propagated here or as part of the state vector?
    def update_speed(self, delta_time):
        """
        Update the speed of the reaction wheel based on the current torque and time step.

        :param delta_time: Time step for the update (s)
        """
        acceleration = self.current_torque  # Assuming unit inertia for simplicity
        self.current_speed += acceleration * delta_time
        if abs(self.current_speed) > self.max_speed:
            self.current_speed = self.max_speed if self.current_speed > 0 else -self.max_speed
