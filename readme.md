# 3dDynamicsSim

This Repo contains C++ code for simulating the 3d Dynamics of a satellite  in orbit.

Credits to Derek Fan for his appraoch to C++ Eigen <--> Python bindings. I used his method with pybind11. 

## SPICE data download
- run the data_downloader.py to download 'de440.bsp' from https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ and place within data/, or do it manually

## Build Instructions
- `git submodule update --recursive --init `
- run build_sim_debug.sh

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
