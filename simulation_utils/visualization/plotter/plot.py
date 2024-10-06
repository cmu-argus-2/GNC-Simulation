#!/usr/bin/python3
import inspect
import traceback
import numpy as np
import os
import sys
from plots import MontecarloPlots
from plot_menu import get_user_selections
from matplotlib import pyplot as plt
import argparse

PERCENTAGE_OF_DATA_TO_PLOT = 1

# Create a dictionary to hold method names and their corresponding functions
all_plotting_task_names = []
for name, func in inspect.getmembers(MontecarloPlots, predicate=inspect.isfunction):
    if not name.startswith("_"):
        all_plotting_task_names.append(name)
all_plotting_task_names = sorted(all_plotting_task_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="This program generates plots from data produced by a montecarlo job",
        epilog="=" * 80,
    )
    parser.add_argument("job_name", metavar="job_name")
    parser.add_argument("-i", "--interactive", action="store_true")

    args = parser.parse_args()

    job_directory = os.path.realpath(
        os.path.join("../../../montecarlo/results", args.job_name)
    )
    plot_directory = os.path.join(job_directory, "plots")

    assert os.path.exists(job_directory), job_directory

    plotting_task_names = all_plotting_task_names
    if args.interactive:
        plotting_task_names = get_user_selections(all_plotting_task_names)

    os.system(f"mkdir -p {plot_directory}")
    assert os.path.exists(plot_directory), plot_directory

    if os.path.exists(plot_directory) and os.listdir(plot_directory) != []:
        np.random.seed()
        passkey = np.random.randint(100, 1000)
        user_input = input(
            f'****WARNING****: Found exisitng plot folder for job "{args.job_name}" at "{plot_directory}". Are you sure you want to risk overwriting existing plots? [Type {passkey} to continue] '
        )
        if user_input != str(passkey):
            print("exiting without plotting")
            exit(0)

    show_plots = args.interactive and input("View plots? [y/n]") == "y"
    mcp = MontecarloPlots(
        job_directory,
        plot_directory,
        PERCENTAGE_OF_DATA_TO_PLOT=PERCENTAGE_OF_DATA_TO_PLOT,
        close_after_saving=not show_plots,
    )

    for plotting_task_name in plotting_task_names:
        try:
            plotting_task = getattr(mcp, plotting_task_name)
            plotting_task()
        except:
            traceback.print_exc()
        print()
    if show_plots:
        plt.show()
