"""
Logger Module

This module provides a Logger class that centralizes the logging process across various
modules within a flight software system. The logger ensures that all log messages, irrespective of their
origin, are uniformly directed to both the console and a log file, facilitating easier monitoring and debugging.
The Logger adds the filename and line number of the logging call, making it easier to trace log messages.

The Logger supports dynamic configuration of log levels and output destinations. It is designed to handle
concurrent logging calls from multiple modules in a thread-safe manner.

Example Usage:
    From any module in the software system, import and use the Logger to log messages of various severity:
    ```
    from flight import Logger

    # Example of configuring the logger and initializing the log file
    Logger.configure(log_level=logging.DEBUG, log_file='log/specific_system.log')
    Logger.initialize_log(module_name=sys.modules[__name__], init_msg="Logger purpose: ...")

    # Example of logging messages
    Logger.log('INFO', "This is an info message")
    Logger.log('DEBUG', "This is a debug message")
    Logger.log('ERROR', "This is an error message")
    ```

Configuration:
    - `log_file`: Specifies the path of the log file where logs will be written.
    - `log_level`: Sets the threshold for the log messages that are to be handled. Lower levels will be ignored.

Initialization:
    Logger is initialized automatically on first use with default settings, but it can be re-configured anytime 
    with `Logger.configure()` method to suit different operational needs such as during different phases of a mission.

Author(s): Eddie
Date: [Creation or Last Update Date]
"""

import logging
import os
import inspect

# Default configuration upon module load (can be reconfigured elsewhere in the code)
# Logger.configure(log_level=logging.DEBUG, log_file='log/payload.log')


def map_log_level(log_level):
    """Maps the input log level to the corresponding logging module level."""
    if log_level.upper() == "INFO":
        return logging.INFO
    elif log_level.upper() == "DEBUG":
        return logging.DEBUG
    elif log_level.upper() == "WARNING":
        return logging.WARNING
    elif log_level.upper() == "ERROR":
        return logging.ERROR


class Logger:
    logger = None
    log_file_path = "log/payload.log"
    levels = ["INFO", "DEBUG", "WARNING", "ERROR"]

    @classmethod
    def configure(cls, log_level="INFO", log_file="log/payload.log"):
        """Configures the class logger with specific handlers and levels."""
        cls.log_file_path = os.path.join(os.getcwd(), log_file)
        # Create directory for log file if it does not exist
        if not os.path.exists(os.path.dirname(cls.log_file_path)):
            os.makedirs(os.path.dirname(cls.log_file_path))

        # Set up the logger
        cls.logger = logging.getLogger("AppLogger")
        cls.logger.setLevel(map_log_level(log_level))  # Set the overall minimum logging level
        cls.logger.handlers = []  # Clear existing handlers

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(cls.log_file_path, mode="a")  # Append mode
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(map_log_level(log_level))

        # Create formatters and add them to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(caller)-28s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers to the logger
        cls.logger.addHandler(c_handler)
        cls.logger.addHandler(f_handler)

    @classmethod
    def log(cls, level, msg):
        """Logs a message with a specific level from anywhere in the code."""
        if cls.logger is None:
            cls.configure()
        if level.upper() in ["INFO", "DEBUG", "WARNING", "ERROR"]:
            caller_info = cls.get_caller_info()
            cls.logger.log(getattr(logging, level.upper()), msg, extra={"caller": caller_info})
        else:
            cls.logger.error("Invalid logging level specified.")

    @staticmethod
    def get_caller_info():
        """Gets the caller's information for logging purposes."""
        try:
            # Adjusted the index as needed
            caller_frame_record = inspect.stack()[2]
            frame = caller_frame_record[0]
            info = inspect.getframeinfo(frame)
            return f"{os.path.basename(info.filename)}:{info.lineno:4d}"
        except IndexError:
            return "UnknownCaller:0"

    @classmethod
    def initialize_log(cls, module_name, init_msg):
        """Initializes the log file with a specific message detailing the initialization context."""
        try:
            with open(cls.log_file_path, "w") as file:
                pass  # Clear the log file
            cls.logger.info("Log file initialized.", extra={"caller": "LoggerInit"})
            cls.logger.info(f"Logger initialized by: {module_name}", extra={"caller": "LoggerInit"})
            cls.logger.info(init_msg, extra={"caller": "LoggerInit"})
        except Exception as e:
            cls.logger.error("Failed to initialize log file.", extra={"caller": "LoggerError"})
