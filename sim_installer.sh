#!/bin/bash
set -e # exit when any command fails

# For First time installation
git submodule update --recursive --init

sudo apt-get install python3-tk -y
sudo apt-get install cmake -y
sudo apt-get install gfortran -y
sudo apt-get install libeigen3-dev -y