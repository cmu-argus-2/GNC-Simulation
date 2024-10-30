#!/bin/bash
set -e # exit when any command fails

# "-O" means otuput to a file with the same name
# "-C -": see https://stackoverflow.com/questions/11856351/how-to-skip-already-existing-files-when-downloading-with-curl
mkdir -p data
echo "Downloading physics model data files (may take a few minutes) ..."
curl -O -C - --silent https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp --output data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc --output data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc --output data
curl -O -C - --silent https://www.ngdc.noaa.gov/IAGA/vmod/igrf13.f --output data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls --output data

mkdir -p build
cd build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j10