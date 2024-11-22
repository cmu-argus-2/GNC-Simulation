import pathlib
from setuptools import setup, find_packages

# Read common requirements
requirements = []
with open(pathlib.Path.cwd() / 'requirements.txt') as fp:
    for line in fp:
        requirements.append(line.strip())

setup(
    name='argusim',
    version='1.0.0',
    packages = find_packages(),
    install_requires = requirements,
    include_package_data = True,
    author               = "Karthik Karumanchi",
    author_email         = "kkaruman@andrew.cmu.edu",
    description          = "Pybind'ed C++ orbital dynamics simulation for the Argus cubesat",
    url                  = "https://github.com/cmu-argus-2/GNC-Simulation.git",
)