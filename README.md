# 3dDynamicsSim

This Repo contians C++ code for simulating the 3d Dynamics of a Rigid Body from bare bones Newton-Euler Equations I found in a textbook and using RK4 numerical integration. There is also a python file which can show a visualization of the Rogid Body's Dynamics.

## Instructions
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `./3dDynamicsSim_main`
- `cd ../`

## Visualization
- `python3 animate.py`



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