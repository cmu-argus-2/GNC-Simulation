# Runs a simulation job from start to finish. This involves:
# 1. Building the C++ core libraries
# 2. Creating the job's directory where results from each trial are logged
# 3. Recording the state of the repo (git hash + uncommitted changes) so we know the exact state of the simulation that led to certain results. For repeatability.
# 4. Launching the trials manager (run_trials.py)
# 5. Plotting the results once all trials have finished

import os
import time
import psutil
import readline  # so user can edit their input in a python prompt generated by input() (https://sharetechnotes.com/technotes/python-input-using-backspace-and-arrow-keys/)
import datetime
import subprocess
import sys

# ============================================= JOB CONFIGURATION CONSTANTS ===========================================
TIMEOUT = 3600  # [s] kill a trial that takes longer than this amount
MAX_CONCURRENT_TRIALS = 10

parameter_file_name = "params.yaml"

# ==================================================== HELPER CODE ====================================================

# "_rel" postfix indicates a relative filepath
repo_root_rel = "../"

# paths relative to "GNC-Simulation/":
pose_simulator_build_script_rel = "build_sim_debug.sh"
montecarlo_rel = "montecarlo/"

# "_abs" suffix indicates an absolute filepath
repo_root_abs = os.path.realpath(repo_root_rel)
pose_simulator_build_script_abs = os.path.join(repo_root_abs, pose_simulator_build_script_rel)
parameter_file_abs = os.path.join(repo_root_abs, montecarlo_rel, "configs/", parameter_file_name)
results_directory_abs = os.path.join(repo_root_abs, montecarlo_rel, "results/")
plot_script_abs = os.path.join(repo_root_abs, "visualization/plotter/plot.py")

# trial_command = "./main"
# trial_command_dir = f"{repo_root_abs}/build/simulation_manager"
trial_command = "'python3 sim.py'"
trial_command_dir = repo_root_abs

# ensure repo_root_abs actually points to the "GNC-SIMULATION" repo
assert os.path.basename(repo_root_abs) == "GNC-Simulation"

# ensure paths exist
assert os.path.exists(repo_root_abs), f"Nonexistent: {repo_root_abs}"
assert os.path.exists(pose_simulator_build_script_abs), f"Nonexistent: {pose_simulator_build_script_abs}"
assert os.path.exists(parameter_file_abs), f"Nonexistent: {parameter_file_abs}"
os.system(f"mkdir -p {results_directory_abs}")
assert os.path.exists(results_directory_abs), f"Nonexistent: {results_directory_abs}"
assert os.path.exists(trial_command_dir), f"Nonexistent: {trial_command_dir}"
assert os.path.exists(plot_script_abs), f"Nonexistent: {plot_script_abs}"


# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


def build_simulator():
    process = subprocess.Popen(
        [
            pose_simulator_build_script_abs,
        ],
        cwd=os.path.dirname(pose_simulator_build_script_abs),
    )
    while process.poll() is None:  # wait while the code is being built
        time.sleep(0.1)
    return_code = process.returncode
    if return_code != 0:
        raise AssertionError(RED + f"Build failed with code {return_code}" + RESET)


def create_job_directory():
    job_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(GREEN + f'job_name:{RESET} "{job_name}"')
    job_directory_abs = os.path.join(results_directory_abs, job_name)
    os.system(f"mkdir -p {job_directory_abs}")
    return (job_name, job_directory_abs)


def record_git_state(job_directory_abs):  # generated by ChatGPT
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root_abs, universal_newlines=True
    ).strip()
    branch_name = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root_abs,
        universal_newlines=True,
    ).strip()
    diff = subprocess.check_output(["git", "diff"], cwd=repo_root_abs, universal_newlines=True)
    output_file_abs = os.path.join(job_directory_abs, "git.txt")
    with open(output_file_abs, "w") as file:
        file.write(f"Branch name: {branch_name}\n\n")
        file.write(f"Commit hash: {commit_hash}\n\n")
        file.write("Uncommitted changes:\n")
        file.write(diff)

    print(GREEN + f"Branch, Commit hash, and diff saved to:{RESET}", output_file_abs)


def plot_results(job_name):
    print(GREEN + "Plotting..." + RESET)
    process = subprocess.Popen(["python3", plot_script_abs, job_name], cwd=os.path.dirname(plot_script_abs))
    while process.poll() is None:  # wait while the plotting is finished
        time.sleep(0.1)
    return_code = process.returncode
    if return_code != 0:
        raise AssertionError(RED + f"Plotting failed with code {return_code}" + RESET)
    print(GREEN + "Plotting complete" + RESET)


def get_process_uptime(pid):
    try:  # from ChatGPT
        process = psutil.Process(pid)  # Get process by PID
        create_time = process.create_time()  # Get the process create time
        current_time = datetime.datetime.now().timestamp()  # Get the current time
        return current_time - create_time
    except psutil.NoSuchProcess:
        return f"No process with PID {pid} found."


if __name__ == "__main__":
    # ===================================================== setup =====================================================
    build_simulator()

    description = input(YELLOW + "Enter a description for this job: " + RESET)

    job_name, job_directory_abs = create_job_directory()

    description_abs = os.path.join(job_directory_abs, "description.txt")
    with open(description_abs, "w") as file:
        file.write(description)

    # record the state of the repo at the time the montecarlo job is run, for reproducibility
    record_git_state(job_directory_abs)

    # save a copy of the parameter file to this job's directory, for reproducibility
    os.system(f"cp {parameter_file_abs} {job_directory_abs}")

    if len(sys.argv) == 1:
        NUM_TRIALS = 1
    else:
        NUM_TRIALS = int(sys.argv[1])
    
    # add 1 so the first trial number is "1" instead of "0"
    TRIALS = [str(x + 1) for x in list(range(NUM_TRIALS))]

    # ======================================= actually run the montecarlo trials ======================================
    trials_manager_process = subprocess.Popen(
        [
            "python3",
            os.path.join(os.getcwd(), "run_trials.py"),
            trial_command,
            trial_command_dir,
            job_directory_abs,
            parameter_file_abs,
            f"{TIMEOUT}",
            f"{MAX_CONCURRENT_TRIALS}",
        ]
        + TRIALS,
    )
    while trials_manager_process.poll() is None:  # trials haven't finished
        time.sleep(1)

    # ================================================= plot the data =================================================
    plot_results(job_name)

    print(GREEN + f"Job {job_name} is complete!" + RESET)
