"""
Landmark Detection Module

This module defines the LandmarkDetector class, which utilizes a pretrained YOLO (You Only Look Once) model
to detect and process landmarks within images. The detector extracts landmarks as bounding boxes along with
their associated class IDs and confidence scores. The main functionality revolves around the detection of
landmarks in given images and the extraction of useful information such as centroids and class/confidence data.

Dependencies:
- numpy: Used for array manipulations and handling numerical operations.
- cv2 (OpenCV): Required for image processing tasks.
- ultralytics YOLO: The YOLO model implementation from Ultralytics, used for object detection tasks. (Large package warning)

Author: Eddie, Haochen
Date: [Creation or Last Update Date]
"""

import numpy as np
from ultralytics import YOLO
import os
import csv
import cv2
import time
from PIL import Image
from flight import Logger

LD_MODEL_SUF = "_nadir.pt"

# Define error and info messages
error_messages = {
    "CONFIGURATION_ERROR": "Configuration error.",
    "LOADING_FAILED": "Failed to load necessary data.",
    "DETECTION_FAILED": "Detection process failed.",
    "INVALID_DIMENSIONS": "Invalid bounding box dimensions detected.",
    "LOW_CONFIDENCE": "Skipping low confidence landmark.",
    "EMPTY_DETECTION": "No landmark detected.",
}

info_messages = {
    "INITIALIZATION_START": "Initializing LandmarkDetector.",
    "DETECTION_START": "Starting the landmark detection process.",
    "DETECTION_COMPLETE": "Landmark detection completed successfully.",
}


class LandmarkDetector:

    def __init__(self, region_id, model_path="models/ld"):
        """
        Initialize the LandmarkDetector with a specific region ID and model path
        The YOLO object is created with the path to a specific pretrained model
        """
        Logger.log("INFO", f"Initializing LandmarkDetector for region {region_id}.")

        self.region_id = region_id
        try:
            self.model = YOLO(os.path.join(model_path, region_id, f"{region_id}_nadir.pt"))
            self.ground_truth = self.load_ground_truth(
                os.path.join(model_path, region_id, f"{region_id}_top_salient.csv")
            )
        except Exception as e:
            Logger.log("ERROR", f"{error_messages['LOADING_FAILED']}: {e}")
            raise

    def load_ground_truth(self, ground_truth_path):
        """
        Loads ground truth bounding box coordinates from a CSV file.

        Args:
            ground_truth_path (str): Path to the ground truth CSV file.

        Returns:
            list of tuples: Each tuple contains (top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat)
        """
        ground_truth = []
        try:
            with open(ground_truth_path, "r") as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip the header row
                for row in csvreader:
                    ground_truth.append(
                        (
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                        )
                    )
        except Exception as e:
            Logger.log("ERROR", f"{error_messages['CONFIGURATION_ERROR']}: {e}")
            raise
        return ground_truth

    def calculate_bounding_boxes(self, centroid_xy, landmark_wh):
        """
        Calculate the top-left and bottom-right coordinates of bounding boxes from centroids and dimensions.

        Args:
            centroid_xy (np.ndarray): Centroid coordinates of the landmarks as [x, y].
            landmark_wh (np.ndarray): Dimensions of the landmarks as [width, height].

        Returns:
            np.ndarray: An array of shape (n, 4), with each row containing the top-left and bottom-right coordinates of the bounding boxes.
        """
        # Calculate the top-left and bottom-right coordinates of the bounding boxes
        top_left = centroid_xy - landmark_wh / 2
        bottom_right = centroid_xy + landmark_wh / 2

        # Combine top-left and bottom-right into a single numpy array of shape (n, 4)
        bounding_boxes = np.hstack((top_left, bottom_right))

        return bounding_boxes

    def get_latlons(self, bbox_indexes):
        """
        Get the latitude and longitude for each detected bounding box based on class index.

        Args:
            bbox_indexes (list of int): Indexes of bounding boxes in the ground truth data.

        Returns:
            np.ndarray: Array of bounding box latitudes and longitudes.
        """
        centroids = []
        corners = []
        for index in bbox_indexes:
            (
                centroid_lon,
                centroid_lat,
                top_left_lon,
                top_left_lat,
                bottom_right_lon,
                bottom_right_lat,
            ) = self.ground_truth[index]
            corners.append([top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat])
            centroids.append([centroid_lon, centroid_lat])
        return np.array(centroids), np.array(corners)

    def detect_landmarks(self, frame_obj):
            """
            Detects landmarks in an input image using a pretrained YOLO model and extracts relevant information.

            Args:
                img (np.ndarray): The input image array on which to perform landmark detection.

            Returns:
                tuple: A tuple containing several numpy arrays:
                    - centroid_xy (np.ndarray): Array of [x, y] coordinates for the centroids of detected landmarks.
                    - centroid_latlons (np.ndarray): Array of geographical coordinates [latitude, longitude] for each detected landmark's centroid, based on class ID.
                    - landmark_class (np.ndarray): Array of class IDs for each detected landmark.
                    - confidence_scores

            The detection process filters out landmarks with low confidence scores (below 0.5) and invalid bounding box dimensions. It aims to provide a comprehensive set of data for each detected landmark, facilitating further analysis or processing.
            """
            Logger.log(
                "INFO",
                f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {info_messages['DETECTION_START']}",
            )

            centroid_xy, centroid_latlons, landmark_class, confidence_scores = [], [], [], []
            try:
                # Detect landmarks using the YOLO model
                img = Image.fromarray(cv2.cvtColor(frame_obj.frame, cv2.COLOR_BGR2RGB))
                start_time = time.time()
                results = self.model.predict(img, conf=0.5, imgsz=(1088, 1920), verbose=False)
                end_time = time.time()
                inference_time = end_time - start_time

                landmark_list = []

                for result in results:
                    landmarks = result.boxes

                    if landmarks:  # Sanity Check
                        # Iterate over each detected bounding box (landmark)
                        for landmark in landmarks:
                            x, y, w, h = landmark.xywh[0]
                            cls = landmark.cls[0].item()
                            conf = landmark.conf[0].item()

                            if w < 0 or h < 0:
                                Logger.log("INFO", "Skipping landmark with invalid bounding box dimensions.")
                                continue

                            landmark_list.append([x, y, cls, w, h, conf])

                if not landmark_list:
                    Logger.log("INFO", f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] No landmarks detected in Region {self.region_id}.")
                    return None, None, None, None

                landmark_arr = np.array(landmark_list)

                centroid_xy = landmark_arr[:, :2]
                landmark_class = landmark_arr[:, 2].astype(int)
                confidence_scores = landmark_arr[:, 5]  # Confidence scores

                # Additional processing to calculate bounding box corners and lat/lon coordinates
                centroid_latlons, _ = self.get_latlons(landmark_class)

                Logger.log(
                    "INFO",
                    f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {len(landmark_list)} landmarks detected.",
                )
                Logger.log(
                    "INFO",
                    f"Inference completed in {inference_time:.2f} seconds."
                )
                
                # Logging details for each detected landmark
                if landmark_arr.size > 0:
                    Logger.log(
                        "INFO",
                        f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] class\tcentroid_xy\tcentroid_latlons\tconfidence",
                    )
                    for i in range(len(landmark_list)):
                        # Class ID, convert to int for cleaner logging
                        cls = int(landmark_arr[i, 2])
                        x, y = int(landmark_arr[i, 0]), int(
                            landmark_arr[i, 1]
                        )  # Centroid coordinates, convert to int for cleaner logging
                        lat, lon = centroid_latlons[i]  
                        conf = confidence_scores[i]
                        # Format lat and lon to two decimal places
                        formatted_lat = f"{lat:.2f}"
                        formatted_lon = f"{lon:.2f}"
                        formatted_conf = f"{conf:.2f}"
                        Logger.log(
                            "INFO",
                            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {cls}\t({x}, {y})\t({formatted_lat}, {formatted_lon})\t{formatted_conf}",
                        )

                return centroid_xy, centroid_latlons, landmark_class, confidence_scores

            except Exception as e:
                Logger.log("ERROR", f"Detection failed: {str(e)}")
                raise
