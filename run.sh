#!/bin/bash

num_trials=1
if [ -z "$1" ] ; then
    num_trials=1;
else
    num_trials=$1;
fi

echo "Number of Trials : $num_trials"

# Run Simulation
python3 argusim/run_job.py $num_trials