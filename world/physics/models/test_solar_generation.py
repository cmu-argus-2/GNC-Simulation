from functools import partial
import yaml
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

import brahe
from brahe.constants import R_EARTH, GM_EARTH
from brahe.epoch import Epoch

from world.math.time import increment_epoch
from world.math.integrators import RK4
from world.physics.models.gravity import Gravity
from world.physics.models.solar_generation import SolarGeneration


def get_pointing_attitude(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Calculate the required satellite attitude for pointing nadir with the body y-axis along the orbital angular momentum vector.
    The attitude is returned as a rotation matrix from the body frame to the ECI frame.

    :param position: The position of the satellite in the ECI frame.
    :param velocity: The velocity of the satellite in the ECI frame.
    :return: The rotation matrix from the body frame to the ECI frame.
    """
    z_axis = position / np.linalg.norm(position)  # along radial vector
    y_axis = np.cross(z_axis, velocity)  # along orbital angular momentum
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack((x_axis, y_axis, z_axis))
    return R_body_to_eci


def get_estimated_orbital_state(epoch: Epoch, ltdn: float = 2 * np.pi * (22 / 24)) -> np.ndarray:
    """
    Calculates a position and velocity of the satellite that will put it into an orbit that matches the expected orbit
    that SpaceX will deploy us in as closely as possible.

    The resulting orbit will have the following orbital elements:
    Semi-major axis: R_EARTH + 600km.
    Eccentricity: 0.
    Inclination: 97.79 degrees (sun-synchronous).
    RAAN: chosen to correspond to the specified local time of day at the descending node at the given epoch.
    Argument of perigee: 0 degrees (doesn't matter for a circular orbit).
    True anomaly: the spacecraft will be at the descending node at the specified epoch.

    Information about the secondary payload orbit for SpaceX's Transporter-15 mission can be found here:
    https://impulso.space/launch/f2cf2fd6-1331-4627-88f0-a0ef47c2ec02/

    :param epoch: The epoch at which to estimate the state.
    :param ltdn: The local time of day at the descending node. This is constant for a sun-synchronous orbit.
                 This is represented as a float in the range [0, 2 * np.pi), where 0 corresponds to midnight,
                 np.pi / 2 corresponds to 6am, np.pi corresponds to noon, and 3 * np.pi / 2 corresponds to 6pm.
                 The default value is from the most recent SpaceX Transporter launch and corresponds to 10pm.
    :return: The estimated position and velocity of the satellite in the ECI frame, as a 6-element numpy array.
    """
    semi_major_axis = R_EARTH + 600e3
    inclination = np.deg2rad(97.79)
    speed = np.sqrt(GM_EARTH / semi_major_axis)  # From the vis-viva equation for a circular orbit.

    # Calculate the position and velocity of the satellite assuming that the descending node is along the x-axis.
    position_eci = np.array([semi_major_axis, 0, 0])
    velocity_eci = Rotation.from_euler("x", inclination + np.pi, degrees=False).apply([0, speed, 0])

    # Rotate the position and velocity vectors to the correct orientation.
    sun_position_eci = brahe.ephemerides.sun_position(epoch)
    sun_angle = np.arctan2(sun_position_eci[1], sun_position_eci[0])  # ignore z component
    midnight_angle = sun_angle - np.pi
    rot = Rotation.from_euler("z", midnight_angle + ltdn, degrees=False)
    return np.concatenate(rot.apply((position_eci, velocity_eci)))


def propagate_orbit_and_solar(state_0: np.ndarray, solar_generation: SolarGeneration,
                              initial_epoch: Epoch, dt: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate the orbit of the satellite and calculate the generated power output of the solar panels.

    :param state_0: The initial state of the satellite, as a 6-element numpy array.
    :param solar_generation: The SolarGeneration object used to calculate the power output of the solar panels.
    :param initial_epoch: The initial epoch of the simulation.
    :param dt: The time step of the simulation.
    :param N: The number of time steps to simulate.
    :return: A tuple containing the states of the satellite at each time step and the generated power output at each time step.
    """
    gravity = Gravity()
    states = np.zeros((N, 6))
    generated_power = np.zeros(N)
    states[0, :] = state_0
    generated_power[0] = solar_generation.get_power_output(initial_epoch, state_0[:3],
                                                           get_pointing_attitude(state_0[:3], state_0[3:]))

    def state_derivative(state: np.ndarray, u: None, epoch: Epoch) -> np.ndarray:
        pos = state[:3]
        vel = state[3:]
        acceleration = gravity.acceleration(pos, epoch)
        return np.concatenate((vel, acceleration))

    def f(state: np.ndarray, epoch: Epoch) -> np.ndarray:
        return RK4(state, None, partial(state_derivative, epoch=epoch), dt)

    epoch = initial_epoch
    for i in range(1, N):
        states[i, :] = f(states[i - 1, :], epoch)
        generated_power[i] = solar_generation.get_power_output(epoch, states[i, :3],
                                                               get_pointing_attitude(states[i, :3], states[i, 3:]))
        epoch = increment_epoch(epoch, dt)

    return states, generated_power


def main():
    with open("../../../config.yaml") as f:
        config = yaml.safe_load(f)

    initial_epoch = Epoch(
        *brahe.time.mjd_to_caldate(config["mission"]["start_date"])
    )
    period = 2 * np.pi * np.sqrt((R_EARTH + 600e3) ** 3 / GM_EARTH)
    N = 1000
    dt = period / N

    def get_power_curve(ltdn: float = 2 * np.pi * (22 / 24),
                        deployables_dir: np.ndarray = np.array([0, 0, 1]),
                        deployables_tilt_angle: float = np.pi / 4) -> np.ndarray:
        """
        Calculate the generated power of the satellite over the course of one orbit.

        :param ltdn: The local time of day at the descending node. This is constant for a sun-synchronous orbit.
                     This is represented as a float in the range [0, 2 * np.pi), where 0 corresponds to midnight,
                     np.pi / 2 corresponds to 6am, np.pi corresponds to noon, and 3 * np.pi / 2 corresponds to 6pm.
                     The default value is from the most recent SpaceX Transporter launch and corresponds to 10pm.
        :param deployables_dir: A 3-element numpy array representing the direction of the deployable solar panels.
                                Must be a unit vector with exactly one non-zero element.
        :param deployables_tilt_angle: The angle in radians by which the deployable solar panels are tilted.
                                       See the diagram in SolarGeneration.get_solar_config for the definition of the tilt angle.
        :return: A numpy array of size (N,) containing the generated power of the satellite at each time step.
        """
        solar_generation = SolarGeneration(deployables_dir=deployables_dir,
                                           deployables_tilt_angle=deployables_tilt_angle)
        _, generated_power = propagate_orbit_and_solar(get_estimated_orbital_state(initial_epoch, ltdn),
                                                       solar_generation,
                                                       initial_epoch, dt, N)
        return generated_power

    plt.figure()
    plt.plot(np.arange(N) * dt, get_power_curve())
    plt.xlabel("Time [s]")
    plt.ylabel("Power [W]")
    plt.title(f"Generated Power vs. Time for LTDN=10pm and Default Solar Panel Configuration")
    plt.show()

    ltdns = np.linspace(0, 2 * np.pi, 96)  # every 15 minutes
    mean_power = [np.mean(get_power_curve(ltdn=ltdn)) for ltdn in tqdm(ltdns)]

    plt.figure()
    plt.plot(np.rad2deg(ltdns), mean_power)
    plt.xlabel("Local Time of Descending Node [deg]")
    plt.ylabel("Mean Power [W]")
    plt.title("Mean Generated Power vs. LTDN for Default Solar Panel Configuration")
    plt.show()

    deployables_tilt_angles = np.linspace(0, np.pi / 2, 10)
    for deployables_dir in np.row_stack((np.eye(3), -np.eye(3))):
        print(f"Performing sweep for {deployables_dir=}")
        mean_power = [np.mean(get_power_curve(deployables_dir=deployables_dir,
                                              deployables_tilt_angle=deployables_tilt_angle))
                      for deployables_tilt_angle in tqdm(deployables_tilt_angles)]

        plt.figure()
        plt.plot(np.rad2deg(deployables_tilt_angles), mean_power)
        plt.xlabel("Deployables Tilt Angle [deg]")
        plt.ylabel("Mean Power [W]")
        plt.title(
            f"Mean Generated Power vs. Deployables Tilt Angle for LTDN = 10pm and Deployables in {deployables_dir} Direction")
        plt.show()


if __name__ == "__main__":
    main()
