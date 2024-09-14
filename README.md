# 3dDynamicsSim

This Repo contians C++ code for simulating the 3d Dynamics of a satellite  in orbit.

## Running the sim
- `cd montecarlo/`
- Run `python3 run_job.py`
- Results are written into the `results/` directory
 
## Visualization
- `cd montecarlo/plots_and_analysis/web_visualizer`
- `python3 job_comparison_tool.py`

## Tweaking parameters
- Edit `montecarlo/configs/params.yaml`


# Satellite orbit demo
There are also 2 files for simulating orbital dynamics in the plane. SatelliteAroudnEarth is exactly what it sounds like. SatelliteAroundEarthAndMoon shows a satellite moving on a trajectory resembling a figure 8 around earth and moon. 

## Instructions
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `./SatelliteAroundEarth`
- `./SatelliteAroundEarthAndMoon`

## Visualization
 - `python3 visualize_orbit.py`