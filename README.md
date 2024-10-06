# 3dDynamicsSim

This Repo contians C++ code for simulating the 3d Dynamics of a satellite  in orbit.

Credits to Derek Fan for his appraoch to C++ Eigen <--> Python bindings. I used his method with pybind11. 

## SPICE data download
- use build_sim_debug.sh to download all necessary SPICE kernel files during build

## Build Instructions
- `git submodule update --recursive --init `
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

## Running the sim
- `cd montecarlo/`
- Run `python3 run_job.py`
- Results are written into the `results/` directory
 
## Debugging
- launch.json shows the configurations for debugging python and C++ code.

## Visualization
- `cd montecarlo/plots_and_analysis/web_visualizer`
- `python3 job_comparison_tool.py`

## Tweaking parameters
- Edit `montecarlo/configs/params.yaml`


# Satellite orbit demo
There are also 2 files for simulating orbital dynamics in the plane. SatelliteAroudnEarth is exactly what it sounds like. SatelliteAroundEarthAndMoon shows a satellite moving on a trajectory resembling a figure 8 around earth and moon. 
