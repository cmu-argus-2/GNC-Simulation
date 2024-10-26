#!/bin/bash
set -e # exit when any command fails

# For First time installation
git submodule update --recursive --init

sudo apt install python3-tk -y
sudo apt install cmake -y
sudo apt install gfortran -y
sudo apt install libeigen3-dev -y