#!/bin/bash

set -e

# Build C++ simulation
./install.sh

# Run C++ simulations
num_trials=1
if [ -z "$1" ] ; then
    num_trials=1;
else
    num_trials=$1;
fi

./run.sh $num_trials