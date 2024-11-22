#!/bin/bash
set -e # exit when any command fails

# Install Dependencies
git submodule update --recursive --init

sudo apt-get install python3-tk -y
sudo apt-get install cmake -y
sudo apt-get install gfortran -y
sudo apt-get install libeigen3-dev -y

# "-O" means otuput to a file with the same name
# "-C -": see https://stackoverflow.com/questions/11856351/how-to-skip-already-existing-files-when-downloading-with-curl
mkdir -p data
echo "Downloading physics model data files (may take a few minutes) ..."
curl -O -C - --silent https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp --output-dir data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc --output-dir data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc --output-dir data
curl -O -C - --silent https://www.ngdc.noaa.gov/IAGA/vmod/igrf13.f --output-dir data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls --output-dir data

mkdir -p build
cd build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j10