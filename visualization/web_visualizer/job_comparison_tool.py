#!/usr/bin/python3

import os
from flask import Flask, render_template, abort, send_file, request

app = Flask(__name__)

RESULTS_FOLDER = "../../results"


def get_number_of_trials(job_name):
    try:
        trials_directory = os.path.realpath(os.path.join(RESULTS_FOLDER, job_name, "trials/"))
        return sum(
            1
            for name in os.listdir(trials_directory)
            if os.path.isdir(os.path.join(trials_directory, name)) and name.startswith("trial")
        )
    except FileNotFoundError:
        return "???"


def get_job_description(job_name):
    description_file = os.path.realpath(os.path.join(RESULTS_FOLDER, job_name, "description.txt"))
    try:
        with open(description_file, "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def get_job_size(job_name):
    total_size = 0
    for dirpath, _, filenames in os.walk(os.path.join(RESULTS_FOLDER, job_name)):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def stylize_size_in_byte(number_of_bytes):
    if number_of_bytes < 1024:
        return str(number_of_bytes)
    elif number_of_bytes < 1024**2:
        kB = int((number_of_bytes / 1024.0) + 0.5)  # add half to round
        return f"{kB} kB"
    elif number_of_bytes < 1024**3:
        MB = int((number_of_bytes / (1024.0**2)) + 0.5)  # add half to round
        return f"{MB} MB"
    elif number_of_bytes < 1024**4:
        GB = int((number_of_bytes / (1024.0**3)) + 0.5)  # add half to round
        return f"{GB} GB"


def get_job_list():
    job_list = []
    for job_directory in sorted(os.listdir(RESULTS_FOLDER), reverse=True):
        if os.path.exists(os.path.join(RESULTS_FOLDER, job_directory, "plots")):  # ensure the job directory has plots
            job_list.append(
                {
                    "name": job_directory,
                    "trial_count": get_number_of_trials(job_directory),
                    "description": get_job_description(job_directory),
                    "size": stylize_size_in_byte(get_job_size(job_directory)),
                }
            )
    return job_list


@app.route("/")
def home_page():
    return render_template(
        "home_page.html",
        job_list=get_job_list(),
    )


@app.route("/compare")
def compare():
    print("inside compare")
    jobs_to_compare = request.args.get("jobs_to_compare").split(",")

    data_for_all_jobs = [
        {
            "name": job,
            "description": get_job_description(job),
            "files": set(os.listdir(os.path.join(RESULTS_FOLDER, job, "plots"))),
        }
        for job in jobs_to_compare
    ]

    # find images common to all jobs so they can be displayed together, at the top of the page
    sets_of_image_files = [data_for_job["files"] for data_for_job in data_for_all_jobs]
    common_images = set.intersection(*sets_of_image_files)

    # recombine the common images and other images alphabetically
    for data_for_job in data_for_all_jobs:
        other_images = data_for_job["files"] - common_images
        data_for_job["files"] = sorted(list(common_images)) + sorted(list(other_images))

    return render_template(
        "compare.html",
        jobs_to_compare=data_for_all_jobs,
    )


@app.route("/<job_dir>/plots/<file>")
def get_img(job_dir, file):
    print("trying to get img")
    return send_file(os.path.join(RESULTS_FOLDER, job_dir, "plots", file))


if __name__ == "__main__":
    app.run(debug=True)
