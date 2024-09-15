# built-in imports
import logging
from datetime import datetime
import os

class Logger():
    def __init__(self, sim_params):
        
        self.log_directory = sim_params.log_directory
        
        if not(os.path.exists(self.log_directory)):
            os.mkdir(self.log_directory)
        
        self.logfile_name = os.path.join(self.log_directory,datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_simulation_log.txt")
        logging.basicConfig(filename=self.logfile_name, level=logging.INFO)
        
        self.log_fields = ["Time",
                           "Position ECI",
                           "Velocity ECI",
                           "Quaternion",
                           "Quaternion Rate",
                           "Control Inputs",
                           "Spherical Gravitational Acceleration",
                           "J2 Perturbation Acceleration",
                           "Drag Acceleration",
                           "SRP Acceleration",
                           ]
        
    def log(self, record):
        logging.info('%s', dict(zip(self.log_fields, record)))