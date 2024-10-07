#!/bin/bash
set -e # exit when any command fails

# "-O" means otuput to a file with the same name
# "-C -": see https://stackoverflow.com/questions/11856351/how-to-skip-already-existing-files-when-downloading-with-curl
curl -O -C - --silent https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp --output-dir data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc --output-dir data

mkdir -p build
cd build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j10