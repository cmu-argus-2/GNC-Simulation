# 3dDynamicsSim

This repo simulates the 3D Dynamics of a satellite in orbit. The core dynamics are computed, and the state is propagated in C++. This is exposed to Python via pybind11. 

Credits to Derek Fan for his approach to C++ Eigen <--> Python bindings. I used his method with pybind11. 

## SPICE data download
Run `./build_sim_debug.sh` to download all necessary SPICE kernel files during build

## Setting Up
### First time
1. Run `git submodule update --recursive --init` to set up dependencies
2. Run `sudo apt install python3-tk -y` to install tkinter
3. Run `python -m venv .venv --system-site-packages` to create a virtual environment (venv)
4. Run `source .venv/bin/activate` to activate the venv
5. Run `pip install -r requirements.txt`

### Each time
1. Run `source .venv/bin/activate` to activate the venv
2. Once done with development, run `deactivate` to exit the venv

## Running the sim
1. `cd simulation_manager/`
2. Run `python3 run_job.py`
3. Results and plots are written into the `results/` directory
 
## Manual Build Instructions
Run `./build_sim_debug.sh`

## Debugging

### Plot Visualization

#### Web Viewer Comparison tool 
1. `cd visualization/web_visualizer`
2. Open a web browser and go to `http://127.0.0.1:5000/`
3. `python3 job_comparison_tool.py`

#### Interactive - all trials
1. `cd visualization/plotter`
2. `python3 plot.py <JOB_NAME> -i`

#### Interactive - specific trials
1. `cd visualization/plotter`
2. `python3 plot.py <JOB_NAME> -i -t [list of trial numbers to debug]`

#### Replotting an existing job after changing the plotting scripts
1. `cd visualization/plotter`
2. `python3 plot.py <JOB_NAME>`

### ModuleNotFoundError: No module named XXXX
Remember to run in a virtual environment

### Trial X finished with return code: Y
Inspect `montecarlo/results/<JOB_NAME>/trials/trialX/output.txt` 

### GDB
Run `launch.json` shows the configurations for debugging python and C++ code.


## Tweaking parameters
Edit `montecarlo/configs/params.yaml`

## Style Guide
### C++
Install the following VSCode extensions:
1. clangd for powerful C++ intellisense (identifier: `llvm-vs-code-extensions.vscode-clangd`)
2. doxygen docstring generator (identifier: `cschlosser.doxdocgen`)

## Code Architecture
Refer to the code architecture <a href="https://www.notion.so/Physics-Model-Simulation-Architecture-10648018d82a80d4a90ce8fb38b47777">here</a>
