# Starts up trials and manages them to ensure no more than a specified number run concurrently.
'''

    A single trial involves sampling the initial parameters once, fixing the RNG seed, and invoking sim.py.

    A job involves running multiple trials (100, 1000, etc.), each with a different RNG seed and, thus, a different set of initial parameters. Each trial is an invocation of sim.py.

    run_job.py takes care of running a simulation job from start to finish. This involves:

        Building the C++ core libraries
        Creating the job's directory where results from each trial are logged
        Recording the state of the repo (git hash + uncommitted changes) so we know the exact state of the simulation that led to certain results. For repeatability.
        Launching the trials manager (run_trials.py)
        Plotting the results once all trials have finished

    Step 4 involves run_trials.py. This file's sole job is to ensure multiple trials ( sim.py instances) run in parallel and then start up new trials when previous ones have finished. Basically, we want to keep the computer busy with running our trials.

'''

import os
import sys
import time
import subprocess

# ============================================= JOB CONFIGURATION CONSTANTS ===========================================
trial_command = sys.argv[1]
trial_command_dir = sys.argv[2]
job_directory_abs = sys.argv[3]
parameter_file_abs = sys.argv[4]
TIMEOUT = int(sys.argv[5])
MAX_CONCURRENT_TRIALS = int(sys.argv[6])
TRIALS = [int(arg) for arg in sys.argv[7:]]
NUM_TRIALS = len(TRIALS)
job_name = os.path.basename(job_directory_abs)

# ensure paths exist
assert os.path.exists(parameter_file_abs), f"Nonexistent: {parameter_file_abs}"
assert os.path.exists(trial_command_dir), f"Nonexistent: {trial_command_dir}"
# ==================================================== HELPER CODE ====================================================


# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


def get_process_uptime(pid):  # from chatGPT
    try:
        # Read the 'stat' file for the given PID
        with open(f"/proc/{pid}/stat", "r") as file:
            stats = file.read().split()
        # The 22nd value in /proc/[pid]/stat is the start time of the process
        # in clock ticks since the system boot.
        start_ticks = int(stats[21])

        # Get the system uptime and clock tick rate
        with open("/proc/uptime", "r") as file:
            uptime_seconds = float(file.read().split()[0])
        clock_ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

        # Calculate the process start time in seconds
        start_time_seconds = uptime_seconds - (start_ticks / clock_ticks_per_second)

        # Current time - start time
        process_uptime_seconds = time.time() - (time.time() - start_time_seconds)
        return process_uptime_seconds
    except FileNotFoundError:
        return f"No process with PID {pid} found."
    except IndexError:
        return "Process information could not be parsed."


if __name__ == "__main__":
    process_list = []
    TRIALS_STARTED = 0
    TRIALS_FINISHED = 0

    def updated_process_list():
        global TRIALS_FINISHED
        for process_info in process_list:
            return_code = process_info["process"].poll()
            trial = process_info["trial"]
            pid = process_info["process"].pid
            sent_kill_signal = process_info["sent_kill_signal"]
            if return_code is not None:  # process finished
                if not process_info["is_finished"]:
                    return_code = process_info["process"].returncode
                    color = GREEN if return_code == 0 else RED
                    print(color + f"Trial {trial} finished with return code: {return_code}" + RESET)
                    process_info["is_finished"] = True
                    TRIALS_FINISHED += 1
            else:
                if get_process_uptime(pid) > TIMEOUT and not sent_kill_signal:
                    print(RED + f"Killing trial {trial} due to timeout" + RESET)
                    os.system(f"kill -9 {pid}")
                    process_info["sent_kill_signal"] = True

    trial_env = os.environ.copy()
    while TRIALS_FINISHED < NUM_TRIALS:
        if TRIALS_STARTED < NUM_TRIALS:  # need to run more trials
            number_of_active_trials = TRIALS_STARTED - TRIALS_FINISHED
            if number_of_active_trials < MAX_CONCURRENT_TRIALS:  # can run another trial without freezing up CPU
                TRIAL_NUMBER = TRIALS[TRIALS_STARTED]
                trial_directory_abs = os.path.join(job_directory_abs, "trials", f"trial{TRIAL_NUMBER}")
                trial_output_file = os.path.join(trial_directory_abs, "output.txt")
                
                trial_env["TRIAL_DIRECTORY"] = trial_directory_abs
                trial_env["PARAMETER_FILEPATH"] = parameter_file_abs
                trial_env["TRIAL_NUMBER"] = str(TRIAL_NUMBER)

                os.system(f"mkdir -p {trial_directory_abs}")
                with open(trial_output_file, "w") as outfile:
                    process = subprocess.Popen(
                        trial_command.strip("'").split(" "),
                        env=trial_env,
                        stderr=outfile,
                        stdout=outfile,
                    )

                    process_info = {
                        "trial": TRIAL_NUMBER,
                        "process": process,
                        "is_finished": False,
                        "sent_kill_signal": False,
                    }

                    process_list.append(process_info)

                print(f"Started trial {TRIAL_NUMBER}")
                TRIALS_STARTED += 1
        updated_process_list()
        time.sleep(0.01)
    print(GREEN + f"All trials finished for job: {job_name}" + RESET)#
