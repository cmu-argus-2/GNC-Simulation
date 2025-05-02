from multiprocessing import Pool
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
import yaml
from scipy.spatial.transform import Rotation as R

from argusim.visualization.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from argusim.visualization.isolated_trace import itm
from argusim.build.world.pyphysics import ECI2GEOD

# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


def ground_track(data_dict, save_dir):
    ground_track = itm.figure()
    m = Basemap()  # cylindrical projection by default
    m.bluemarble()
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(np.arange(-90, 90, 15), linewidth=0.2, labels=[1, 1, 0, 0])
    m.drawmeridians(np.arange(-180, 180, 30), linewidth=0.2, labels=[0, 0, 0, 1])


    x_km = data_dict["r_x ECI [m]"] / 1000
    y_km = data_dict["r_y ECI [m]"] / 1000
    z_km = data_dict["r_z ECI [m]"] / 1000
    time_vec = data_dict["Time [s]"]

    # TODO convert from ECI to ECEF
    lon = np.zeros_like(x_km)
    lat = np.zeros_like(x_km)
    for k in range(len(x_km)):
        lon[k], lat[k], _ = ECI2GEOD([x_km[k] * 1000, y_km[k] * 1000, z_km[k] * 1000], time_vec[k])

    # https://matplotlib.org/basemap/stable/users/examples.html
    itm.figure(ground_track)
    m.scatter(lon, lat, s=0.5, c="y", marker=".", latlon=True)
    m.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
    m.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")

    save_figure(ground_track, save_dir, "ground_track.png", True)

def pos_plot(data_dict, save_dir):
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["r_x ECI [m]"], data_dict["r_y ECI [m]"], data_dict["r_z ECI [m]"]],
                linewidth=0.5, seriesLabel="True ECI Position"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_gps_posx ECI [m]"], data_dict["fsw_gps_posy ECI [m]"], data_dict["fsw_gps_posz ECI [m]"]],
                linewidth=0.5, seriesLabel="FSW ECI Position"
            )
    
    annotateMultiPlot(title="ECI Position [m]", ylabels=["$r_x$", "$r_y$", "$r_z$"])
    save_figure(itm.gcf(), save_dir, "eci_pos.png", True)
    
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["v_x ECI [m/s]"], data_dict["v_y ECI [m/s]"], data_dict["v_z ECI [m/s]"]],
                linewidth=0.5, seriesLabel="True ECI Velocity"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_gps_velx ECI [m/s]"], data_dict["fsw_gps_vely ECI [m/s]"], data_dict["fsw_gps_velz ECI [m/s]"]],
                linewidth=0.5, seriesLabel="FSW ECI Velocitu"
            )
    
    annotateMultiPlot(title="ECI Velocity [m/s]", ylabels=["$r_x$", "$r_y$", "$r_z$"])
    save_figure(itm.gcf(), save_dir, "eci_vel.png", True)

def attitude_plot(data_dict, save_dir):
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["q_w"], data_dict["q_x"], data_dict["q_y"], data_dict["q_z"]],
                linewidth=0.5, seriesLabel="True Attitude"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_qw"], data_dict["fsw_qx"], data_dict["fsw_qy"], data_dict["fsw_qz"]],
                linewidth=0.5, seriesLabel="FSW Attitude"
            )
    
    annotateMultiPlot(title="Attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
    save_figure(itm.gcf(), save_dir, "attitude.png", True)

def omega_plot(data_dict, save_dir):
    data_dict["omega_norm [rad/s]"] = np.array([np.sqrt(data_dict["omega_x [rad/s]"][i]**2 + data_dict["omega_y [rad/s]"][i]**2 + \
                                       data_dict["omega_z [rad/s]"][i]**2) for i in range(len(data_dict["omega_x [rad/s]"]))])
    data_dict["gyro_norm [rad/s]"] = np.array([np.sqrt(data_dict["gyro_x [rad/s]"][i]**2 + data_dict["gyro_y [rad/s]"][i]**2 + \
                                       data_dict["gyro_z [rad/s]"][i]**2) for i in range(len(data_dict["gyro_x [rad/s]"]))])
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["gyro_x [rad/s]"], data_dict["gyro_y [rad/s]"], data_dict["gyro_z [rad/s]"], data_dict["gyro_norm [rad/s]"]],
                linewidth=0.5, seriesLabel=r"Measured $\omega$"
            )
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["omega_x [rad/s]"], data_dict["omega_y [rad/s]"], data_dict["omega_z [rad/s]"], data_dict["omega_norm [rad/s]"]],
                linewidth=0.5, seriesLabel=r"True $\omega$"
            )
    
    annotateMultiPlot(title="Angular Velocity [rad/s]", ylabels=[r"$\omega_x$", r"$\omega_y$", r"$\omega_z$", r"$||\omega||$"])
    save_figure(itm.gcf(), save_dir, "omega.png", True)
    
def bias_plot(data_dict, save_dir):
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["bias_x [rad/s]"], data_dict["bias_y [rad/s]"], data_dict["bias_z [rad/s]"]],
                linewidth=0.5, seriesLabel="True Bias"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_bias_x [rad/s]"], data_dict["fsw_bias_y [rad/s]"], data_dict["fsw_bias_z [rad/s]"]],
                linewidth=0.5, seriesLabel="FSW Bias"
            )
    
    annotateMultiPlot(title="Bias [rad/s]", ylabels=[r"$b_x$", r"$b_y$", r"$b_z$"])
    save_figure(itm.gcf(), save_dir, "bias.png", True)

def input_plot(data_dict, save_dir):
    data_fields = [data_dict['V_MTB_' + str(i) + " [V]"] for i in range(6)]
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                data_fields,
                linewidth=0.5
            )
    ylabels = ["$V_{XP}$", "$V_{XM}$", "$V_{YP}$", "$V_{YM}$", "$V_{ZP}$", "$V_{ZM}$"]
    annotateMultiPlot(title="Control Input (V)", ylabels=ylabels)
    save_figure(itm.gcf(), save_dir, "control_input.png", True)

def sun_point_plot(result_folder, data_dict, save_dir):
    
    # Load params
    with open(os.path.join(result_folder, 'trial_params.yaml')) as f:
        params = yaml.safe_load(f)
    
    J = np.array(params[params.index('inertia')+1]).reshape((3,3))

    q_w = np.array(data_dict["q_w"])
    q_x = np.array(data_dict["q_x"])
    q_y = np.array(data_dict["q_y"])
    q_z = np.array(data_dict["q_z"])

    sun_error = np.zeros((len(q_w), 3))
    sun_body = np.zeros((len(q_w), 3))
    for i in range(len(q_w)):
        sun_eci = np.array([data_dict["rSun_x ECI [m]"][i], data_dict["rSun_y ECI [m]"][i], data_dict["rSun_z ECI [m]"][i]])
        sun_eci = sun_eci / np.linalg.norm(sun_eci)
        sun_body[i,:] = R.from_quat([q_x[i], q_y[i], q_z[i], q_w[i]]).as_matrix().T@sun_eci
        h = J@np.array([data_dict["omega_x [rad/s]"][i], data_dict["omega_y [rad/s]"][i], data_dict["omega_z [rad/s]"][i]])
        h = h / np.linalg.norm(h)
        sun_error[i,:] = sun_body[i,:] - h
    
    x = np.column_stack((data_dict["Time [s]"]- data_dict["Time [s]"][0], sun_body))
    np.savetxt(os.path.join(result_folder, 'test.txt'), x, delimiter=',')
    
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [sun_error[:,0], sun_error[:,1], sun_error[:,2], np.linalg.norm(sun_error, axis=1)],
                linewidth=0.5
            )

    ylabels = ["$X$", "$Y$", "$Z$", "||"]
    annotateMultiPlot(title="Sun vs Angular Momentum", ylabels=ylabels)
    itm.ylim([0,2])
    save_figure(itm.gcf(), save_dir, "sun_point.png", True)


def battery_diagnostics_plot(data_dict, save_dir):
    itm.figure()
    net_power = np.zeros_like(data_dict["Battery SoC [%]"])
    net_power[1:] = 270000 * 0.01* (data_dict["Battery SoC [%]"][1:] - data_dict["Battery SoC [%]"][0:len(data_dict["Battery SoC [%]"])-1]) / (data_dict["Time [s]"][1]- data_dict["Time [s]"][0])
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["Battery SoC [%]"], data_dict["Battery Temperature [K]"], net_power],
                linewidth=0.5
            )
    
    annotateMultiPlot(title="Battery Diagnostics", ylabels=["SoC", "temperature", "net power"])
    save_figure(itm.gcf(), save_dir, "battery.png", True)
    
def true_sun_plot(data_dict, save_dir):
    sun_eci = np.zeros((len(data_dict["rSun_x ECI [m]"]), 3))
    
    for i in range(len(data_dict["rSun_x ECI [m]"])):
        sun_eci[i,:] = np.array([data_dict["rSun_x ECI [m]"][i], data_dict["rSun_y ECI [m]"][i], data_dict["rSun_z ECI [m]"][i]])
        sun_eci[i,:] = sun_eci[i,:] / np.linalg.norm(sun_eci[i,:])
    
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [sun_eci[:,0], sun_eci[:,1], sun_eci[:,2]],
                linewidth=0.5, seriesLabel="True ECI Sun Pos"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_sun_eci_x"], data_dict["fsw_sun_eci_y"], data_dict["fsw_sun_eci_z"]],
                linewidth=0.5, seriesLabel="FSW ECI Sun Pos"
            )

    ylabels = ["$X$", "$Y$", "$Z$",]
    annotateMultiPlot(title="Sun Position Sim vs FSW", ylabels=ylabels)
    save_figure(itm.gcf(), save_dir, "sun_eci.png", True)
    
def true_mag_plot(data_dict, save_dir):
    mag_eci = np.zeros((len(data_dict["xMag ECI [T]"]), 3))
    
    for i in range(len(data_dict["xMag ECI [T]"])):
        mag_eci[i,:] = np.array([data_dict["xMag ECI [T]"][i], data_dict["yMag ECI [T]"][i], data_dict["yMag ECI [T]"][i]])
        mag_eci[i,:] = mag_eci[i,:] / np.linalg.norm(mag_eci[i,:])
    
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [mag_eci[:,0], mag_eci[:,1], mag_eci[:,2]],
                linewidth=0.5, seriesLabel="True ECI Mag field"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_mag_eci_x"], data_dict["fsw_mag_eci_y"], data_dict["fsw_mag_eci_z"]],
                linewidth=0.5, seriesLabel="FSW ECI Mag Field"
            )

    ylabels = ["$X$", "$Y$", "$Z$",]
    annotateMultiPlot(title="ECI Mag Field Sim vs FSW", ylabels=ylabels)
    save_figure(itm.gcf(), save_dir, "mag_eci.png", True)
    
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["mag_x_body [T]"], data_dict["mag_y_body [T]"], data_dict["mag_z_body [T]"]],
                linewidth=0.5, seriesLabel="True Body Mag field"
            )
    
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["fsw_mag_x_body [T]"], data_dict["fsw_mag_y_body [T]"], data_dict["fsw_mag_z_body [T]"]],
                linewidth=0.5, seriesLabel="FSW Body Mag Field"
            )

    ylabels = ["$X$", "$Y$", "$Z$",]
    annotateMultiPlot(title="Body Mag Field Sim vs FSW", ylabels=ylabels)
    save_figure(itm.gcf(), save_dir, "mag_body.png", True)