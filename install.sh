#!/bin/bash

set -e

# Build C++ Sim
cd argusim/
sh build_sim_debug.sh
cd ../

# Install as a python package as a package
pip install -e .