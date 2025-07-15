# 3dDynamicsSim

This repo simulates the 3d Dynamics of a satellite in orbit. The core dynamics are comptued and the state is propogated in C++. This is exposed to python via pybind. 

## Setting Up
### First time
Run the following commands:
- `python3 -m venv .venv --system-site-packages`
- `source .venv/bin/activate`
- `./install.sh`

### Each time
- Run `source .venv/bin/activate` to activate the venv
- #### DEBUG MODE (RECOMMENDED)
  - Run `./run_debug.sh <NUM_TRIALS>`
- #### RUN MODE
  - Run `./run.sh <NUM_TRIALS>`
- Once done with devlopment, run `deactivate` to exit the venv

## Debugging

### Plot Visualization

#### Web Viewer Comparison tool 
1. `cd argusim/visualization/web_visualizer`
2. Open a web browser and go to `http://127.0.0.1:3000/`
3. `python3 job_comparison_tool.py`

#### Interactive - all trials
1. `cd argusim/visualization/plotter`
2. `python3 plot.py <PATH_TO_JOB_DIRECTORY> -i`

#### Interactive - specific trials
1. `cd argusim/visualization/plotter`
2. `python3 plot.py <PATH_TO_JOB_DIRECTORY> -i -t [list of trial numbers to debug]`

#### Replotting an existing job after changing the plotting scripts
1. `cd argusim/visualization/plotter`
2. `python3 plot.py <PATH_TO_JOB_DIRECTORY>`

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
2. doxygen dostring generator (identifier: `cschlosser.doxdocgen`)

## Code Architecture
Refer to the code architecture <a href="https://www.notion.so/Physics-Model-Simulation-Architecture-10648018d82a80d4a90ce8fb38b47777">here</a>

## Licensing

This project includes code adapted from:
- SatelliteDynamics.jl, licensed under MIT License. Copyright (c) 2024 Duncan Eddy <duncan.eddy@gmail.com>, Stanford Intelligent Systems Laboratory, and contributors. Link here: https://github.com/sisl/SatelliteDynamics.jl
- SatelliteToolbox.jl, licensed under MIT License. Copyright (c) 2014-2025: Ronan Arraes Jardim Chagas. Link here: https://github.com/JuliaSpace/SatelliteToolbox.jl

The copyright notice is available in the ./LICENSES folder. 

The software adapted from SatelliteDynamics.jl is present within the following file(s):
- drag.cpp : ... (waiting on Tuushar)
- ...

The software adapted from SatelliteToolbox.jl is present within the following file(s):
-  utils_and_transforms.cpp : functions for converting raan to ltan/ltdn and associated subfunctions.
