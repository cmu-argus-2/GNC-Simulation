from multiprocessing import Pool
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import os

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

def attitude_plot(data_dict, save_dir):
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["q_w"], data_dict["q_x"], data_dict["q_y"], data_dict["q_z"]],
                linewidth=0.5
            )
    
    annotateMultiPlot(title="True attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
    save_figure(itm.gcf(), save_dir, "attitude_true.png", True)

def omega_plot(data_dict, save_dir):
    data_dict["omega_norm [rad/s]"] = np.array([np.sqrt(data_dict["omega_x [rad/s]"][i]**2 + data_dict["omega_y [rad/s]"][i]**2 + \
                                       data_dict["omega_z [rad/s]"][i]**2) for i in range(len(data_dict["omega_x [rad/s]"]))])
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["omega_x [rad/s]"], data_dict["omega_y [rad/s]"], data_dict["omega_z [rad/s]"], data_dict["omega_norm [rad/s]"]],
                linewidth=0.5
            )
    
    annotateMultiPlot(title="True omega [rad/s]", ylabels=[r"$\omega_x$", r"$\omega_y$", r"$\omega_z$", r"$||\omega||$"])
    save_figure(itm.gcf(), save_dir, "omega_true.png", True)

def battery_diagnostics_plot(data_dict, save_dir):
    itm.figure()
    multiPlot(
                data_dict["Time [s]"]- data_dict["Time [s]"][0],
                [data_dict["Battery SoC [%]"], data_dict["Battery Temperature [K]"]],
                linewidth=0.5
            )
    
    annotateMultiPlot(title="Battery Diagnostics", ylabels=["SoC", "temperature"])
    save_figure(itm.gcf(), save_dir, "battery.png", True)