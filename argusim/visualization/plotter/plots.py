from multiprocessing import Pool
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import os

from argusim.visualization.plotter.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from argusim.visualization.plotter.isolated_trace import itm
from argusim.visualization.plotter.parse_bin_file import parse_bin_file_wrapper
from argusim.visualization.plotter.plot_pointing import pointing_plots
from argusim.visualization.plotter.actuator_plots import actuator_plots
from argusim.visualization.plotter.sensor_plots import gyro_plots, sunsensor_plots, magsensor_plots
from argusim.visualization.plotter.att_animation import att_animation
from argusim.visualization.plotter.att_det_plots import plot_state_est_cov, EKF_err_plots, EKF_st_plots
from argusim.visualization.plotter.plot_true_states import plot_true_st, plot_true_gyro_bias
import yaml

# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


class MontecarloPlots:
    def __init__(
        self,
        trials,
        trials_directory,
        plot_directory,
        PERCENTAGE_OF_DATA_TO_PLOT,
        close_after_saving=True,
    ):
        self.trials = trials
        self.trials_dir = trials_directory
        self.plot_dir = plot_directory
        self.PERCENTAGE_OF_DATA_TO_PLOT = PERCENTAGE_OF_DATA_TO_PLOT
        self.close_after_saving = close_after_saving
        self.NUM_TRIALS = len(self.trials)

    def _get_files_across_trials(self, filename):
        filepaths = []
        for trial_number in self.trials:
            filepath = os.path.join(self.trials_dir, f"trial{trial_number}/{filename}")
            if os.path.exists(filepath):
                filepaths.append((trial_number, filepath))
            else:
                print(RED + f"Trial {trial_number} is missing {filename}" + RESET)
        return filepaths

    def true_state_plots(self):


        with open(os.path.join(self.trials_dir, "../../../configs/params.yaml"), "r") as f:
            pyparams = yaml.safe_load(f)  
        pyparams["trials"]             = self.trials
        pyparams["trials_dir"]         = self.trials_dir
        pyparams["plot_dir"]           = self.plot_dir
        pyparams["close_after_saving"] = self.close_after_saving
        if (pyparams["PlotFlags"]["true_state_plots"] or 
            pyparams["PlotFlags"]["pointing_plots"] or 
            pyparams["PlotFlags"]["actuator_plots"]):
            # ==========================================================================
            filepaths = self._get_files_across_trials("state_true.bin")

            START = time.time()
            args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            # ==========================================================================
            if pyparams["PlotFlags"]["true_state_plots"]:
                plot_true_st(pyparams, data_dicts, filepaths)
            
            # ==========================================================================
            # Pointing Plots: Controller target versus real attitude
            if pyparams["PlotFlags"]["pointing_plots"]:
                pointing_plots(pyparams, data_dicts, filepaths)

            # ==========================================================================
            # Actuator Plots: Reaction Wheel Speed and Torque, Magnetorquer Torque
            if pyparams["PlotFlags"]["actuator_plots"]:
                actuator_plots(pyparams, data_dicts, filepaths)

            # ==========================================================================
            # Attitude Animation
            if pyparams["PlotFlags"]["att_animation"]:
                att_animation(pyparams, data_dicts)
        
        # ========================= True gyro bias plots =========================
        if pyparams["PlotFlags"]["true_state_plots"]:
            filepaths = self._get_files_across_trials("gyro_bias_true.bin")

            START = time.time()
            args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            # --------------------------------------------------------------------------
            plot_true_gyro_bias(pyparams, data_dicts, filepaths)


    def sensor_measurement_plots(self):
        with open(os.path.join(self.trials_dir, "../../../configs/params.yaml"), "r") as f:
                pyparams = yaml.safe_load(f)       
        
        if pyparams["PlotFlags"]["sensor_measurements"]:
            # ======================= Gyro measurement plots =======================
            filepaths = self._get_files_across_trials("gyro_measurement.bin")

            START = time.time()
            args = [(filepath, 100) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            
            pyparams["plot_dir"]           = self.plot_dir
            pyparams["close_after_saving"] = self.close_after_saving
            # --------------------------------------------------------------------------
            gyro_plots(pyparams, data_dicts, filepaths)

            # ====================== Sun Sensor measurement plots ======================
            filepaths = self._get_files_across_trials("sun_sensor_measurement.bin")

            START = time.time()
            args = [(filepath, 100) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            # --------------------------------------------------------------------------
            sunsensor_plots(pyparams, data_dicts, filepaths)

            # ====================== Magnetometer measurement plots ======================
            filepaths = self._get_files_across_trials("magnetometer_measurement.bin")

            START = time.time()
            args = [(filepath, 100) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            # --------------------------------------------------------------------------
            magsensor_plots(pyparams, data_dicts, filepaths)


    def _plot_state_estimate_covariance(self, pyparams):
        if not pyparams:
            return
        filepaths = self._get_files_across_trials("state_covariance.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        # ==========================================================================
        pyparams["NUM_TRIALS"]             = self.NUM_TRIALS

        plot_state_est_cov(pyparams, data_dicts, filepaths)
        

    def EKF_error_plots(self):
        with open(os.path.join(self.trials_dir, "../../../configs/params.yaml"), "r") as f:
            pyparams = yaml.safe_load(f)     
        
        if pyparams["PlotFlags"]["MEKF_plots"]:
            filepaths = self._get_files_across_trials("EKF_error.bin")

            START = time.time()
            args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")
            # ==========================================================================  
            pyparams["trials"]                 = self.trials
            pyparams["trials_dir"]             = self.trials_dir
            pyparams["plot_dir"]               = self.plot_dir
            pyparams["close_after_saving"]     = self.close_after_saving
            pyparams["NUM_TRIALS"]             = self.NUM_TRIALS

            pyparams = EKF_err_plots(pyparams, data_dicts)

            # --------------------------------------------------------------------------
            self._plot_state_estimate_covariance(pyparams)  # show 3 sigma bounds

    def EKF_state_plots(self):
        with open(os.path.join(self.trials_dir, "../../../configs/params.yaml"), "r") as f:
            pyparams = yaml.safe_load(f)     
        
        if pyparams["PlotFlags"]["MEKF_plots"]:
            # ======================= Estimated gyro bias =======================
            filepaths = self._get_files_across_trials("EKF_state.bin")

            START = time.time()
            args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
            with Pool() as pool:
                data_dicts = pool.map(parse_bin_file_wrapper, args)
            END = time.time()
            print(f"Elapsed time to read in data: {END-START:.2f} s")

              
            pyparams["trials"]                 = self.trials
            pyparams["trials_dir"]             = self.trials_dir
            pyparams["plot_dir"]               = self.plot_dir
            pyparams["close_after_saving"]     = self.close_after_saving
            pyparams["NUM_TRIALS"]             = self.NUM_TRIALS

            EKF_st_plots(pyparams, data_dicts, filepaths)