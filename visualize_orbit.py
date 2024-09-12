import numpy as np
from matplotlib import pyplot as plt


EARTH_RADIUS = 63781370 # [m]
# filepath = "build/SatelliteAroundEarth.txt"
filepath = "build/SatelliteAroundEarthAndMoon.txt"

data = np.loadtxt(filepath) # [m]

x = data[:, 0]
y = data[:, 1]


plt.plot(x/EARTH_RADIUS,y/EARTH_RADIUS)
plt.xlabel("x [EARTH RADII]")
plt.ylabel("y [EARTH RADII]")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.show()