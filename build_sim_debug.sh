#!/bin/bash
set -e # exit when any command fails
mkdir -p build
cd build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4