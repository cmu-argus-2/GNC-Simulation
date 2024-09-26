# GNC-Simulation - Basilisk Sim

## Issues
1. Atmospheric density does not compute and stays 0 (this is only true for MSIS atmosphere. If I use Exponential Atmosphere instead, it works)
2. Magnetorquer does not apply any torque on satellite (even though magnetic field is active and has values)
3. Providing torque to the RW does not induce rotation for the satellite
4. Segmentation faults : setting a large max magnetorquer dipole moment causes the program to crash with a segmentation fault (No additional info is provided and therefore no clear way to fix it)
