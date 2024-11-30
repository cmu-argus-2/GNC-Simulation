import cv2
import os
import yaml
import time
import hashlib
from datetime import datetime
import logging
from typing import List
import numpy as np
import threading
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import Logger


class CameraErrorCodes:
    CAMERA_INITIALIZATION_FAILED = 1001
    CAPTURE_FAILED = 1002
    NO_IMAGES_FOUND = 1003
    READ_FRAME_ERROR = 1004
    CAMERA_NOT_OPERATIONAL = 1005
    CONFIGURATION_ERROR = 1006
    SUN_BLIND = 1007


error_messages = {
    CameraErrorCodes.CAMERA_INITIALIZATION_FAILED: "Camera initialization failed.",
    CameraErrorCodes.CAPTURE_FAILED: "Failed to capture image.",
    CameraErrorCodes.NO_IMAGES_FOUND: "No images found.",
    CameraErrorCodes.READ_FRAME_ERROR: "Error reading frame.",
    CameraErrorCodes.SUN_BLIND: "Image blinded by the sun",
    CameraErrorCodes.CAMERA_NOT_OPERATIONAL: "Camera is not operational.",
    CameraErrorCodes.CONFIGURATION_ERROR: "Configuration error.",
    CameraErrorCodes.CONFIGURATION_ERROR: "Configuration error.",
}


class Frame:
    def __init__(self, frame, camera_id, timestamp):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp
        # Generate ID by hashing the timestamp
        self.frame_id = self.generate_frame_id(timestamp)
        self.landmarks = []

    def generate_frame_id(self, timestamp):
        """
        Generates a unique frame ID using the hash of the timestamp.

        Args:
            timestamp (datetime): The timestamp associated with the frame.

        Returns:
            str: A hexadecimal string representing the hash of the timestamp.
        """
        # Convert the timestamp to string and encode it to bytes, then hash it
        timestamp_str = str(timestamp)
        hash_object = hashlib.sha1(timestamp_str.encode())  # Using SHA-1
        frame_id = hash_object.hexdigest()
        return frame_id[:16]  # Optionally still shorten if needed

    def update_landmarks(self, new_landmarks):
        """Update the frame with new landmark data."""
        self.landmarks = new_landmarks
        Logger.log("INFO", f"[Camera {self.camera_id} frame {self.frame_id}] Landmarks updated on Frame object.")

    def save(self):
        pass

    @classmethod
    def resize(cls, img, width=640, height=480):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class Camera:
    def __init__(self, camera_id, config_path):
        try:
            config = self.load_config(config_path)
        except Exception as e:
            Logger.log("ERROR", f"{error_messages[CameraErrorCodes.CONFIGURATION_ERROR]}: {e}")
            raise ValueError(error_messages[CameraErrorCodes.CONFIGURATION_ERROR])

        self.stop_event = False
        self.camera_id = camera_id
        self.image_folder = f"data/camera_{camera_id}"
        os.makedirs(self.image_folder, exist_ok=True)
        self.max_startup_time = config["max_startup_time"]
        self.camera_settings = config["cameras"].get(camera_id, {})
        if self.camera_settings != {}:

            self.resolution = (
                self.camera_settings["resolution"]["width"],
                self.camera_settings["resolution"]["height"],
            )
            self.zoom = self.camera_settings.get("zoom")
            self.focus = self.camera_settings.get("focus")
            self.exposure = self.camera_settings.get("exposure")

            self.camera_status = self.initialize_camera()
            
            Logger.log(
                "INFO",
                f"Camera {camera_id}: {self.camera_status}",
            )

            self._current_frame = None
            self.all_frames = []

            Logger.log(
                "INFO",
                f"Camera {camera_id}: Initialized with settings {self.camera_settings}",
            )
        else:
            self.camera_status = 0
            Logger.log("ERROR", f"Camera {camera_id}: Configuration not found.")

    def initialize_camera(self):
        start_time = time.time()
        self.cap = cv2.VideoCapture(self.camera_id)
        status = 0
        if self.cap.isOpened():
            elapsed_time = (
                time.time() - start_time
            ) * 1000  # Calculate elapsed time in milliseconds

            if elapsed_time <= self.max_startup_time:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                Logger.log(
                    "INFO",
                    f"Camera {self.camera_id}: Successfully initialized within {self.max_startup_time} ms",
                )
                status = 1
                return status
            else:
                Logger.log(
                    "ERROR",
                    f"Camera {self.camera_id} initialization exceeded {self.max_startup_time} milliseconds.",
                )
                self.log_error(CameraErrorCodes.CAMERA_INITIALIZATION_FAILED)
                return status
        else:
            return status

    def check_operational_status(self):
        if not hasattr(self, "cap") or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(
                self.camera_id
            )  ## This line shouldn't be there as it takes too much time to create a new video capture object
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.camera_status = 1
                return self.camera_status
            else:
                self.camera_status = 0
                return self.camera_status
        return self.camera_status

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def log_error(self, error_code):
        message = error_messages.get(error_code, "Unknown error.")
        Logger.log("ERROR", f"Camera {self.camera_id}: {message}")

    def capture_frame(self):
        if self.camera_status:
            try:
                ret, frame = self.cap.read()
                if ret:
                    timestamp = datetime.now()
                    # Logger.log("INFO", f"Camera {self.camera_id}: Frame captured at {timestamp}")

                    self.current_frame = Frame(frame, self.camera_id, timestamp)
                    # self.save_image(self.current_frame)
                    # self.all_frames.append(self.current_frame)
                    return self.current_frame

                else:
                    Logger.log("ERROR", f"Camera {self.camera_id}: Failed to capture image")
                    self.log_error(CameraErrorCodes.READ_FRAME_ERROR)
                    self.log_error(CameraErrorCodes.CAPTURE_FAILED)
                    self.camera_status = 0
                    return None
            except Exception as e:
                Logger.log("ERROR", f"Camera {self.camera_id}: Failed to capture image: {e}")
                self.log_error(CameraErrorCodes.CAPTURE_FAILED)
                self.camera_status = 0
                return None
        else:
            Logger.log("ERROR", f"Camera {self.camera_id}: Not operational.")
            self.log_error(CameraErrorCodes.CAMERA_NOT_OPERATIONAL)

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value

    def get_latest_image(self):
        image_files = os.listdir(self.image_folder)
        if not image_files:
            Logger.log("ERROR", f"Camera {self.camera_id}: No images found.")
            self.log_error(CameraErrorCodes.NO_IMAGES_FOUND)
            return None
        latest_image_path = max(
            [os.path.join(self.image_folder, filename) for filename in image_files],
            key=os.path.getctime,
        )
        return cv2.imread(latest_image_path)

    def set_zoom(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_ZOOM, self.zoom)

    def set_focus(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)

    def set_exposure(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def save_image(self, target_frame):
        frame = target_frame.frame
        ts = target_frame.timestamp
        image_name = f"{self.image_folder}/{ts}.jpg"
        cv2.imwrite(image_name, frame)
        Logger.log("INFO", f"Camera {self.camera_id}: Image saved as {image_name}")

        self._maintain_image_limit(self.image_folder, 50)

    def _maintain_image_limit(self, directory_path, limit=50):
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]
        # Sort files by creation time (oldest first)
        files.sort(key=os.path.getctime)

        # If more than `limit` files, remove the oldest ones
        while len(files) > limit:
            os.remove(files[0])
            Logger.log("INFO", f"Camera {self.camera_id}: Removed old image {files[0]} to maintain limit")
            files.pop(0)

    # DEBUG only
    def get_live_feed(self):
        if self.check_operational_status():
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now()
                curr_frame = Frame(frame, self.camera_id, timestamp)
                self.all_frames.append(curr_frame)
                self.save_image(curr_frame)
                cv2.imshow(f"Live Feed from Camera {self.camera_id}", frame)
        else:
            Logger.log("ERROR", f"Camera {self.camera_id} is not operational.")
            self.log_error(CameraErrorCodes.CAMERA_NOT_OPERATIONAL)

    def stop_live_feed(self):
        self.stop_event = True


class CameraManager:

    def __init__(self, camera_ids, config_path="configuration/camera_configuration.yml"):
        self.cameras = {}
        for camera_id in camera_ids:
            cam_obj = Camera(camera_id, config_path=config_path)
            if cam_obj is not None:
                self.cameras[camera_id] = cam_obj
                Logger.log("INFO", f"Camera {camera_id} added to the camera manager.")
                Logger.log("INFO", f"Camera {camera_id} operational status: {cam_obj.camera_status}")

        number_of_cameras = len(self.cameras)
        self.camera_frames = []
        self.stop_event = False
        self.inf_flag = False
        self.ML_image_path = "data/inference_output/frames_w_landmarks.jpg"
        Logger.log("INFO", f"Camera Manager initialized.")

    def get_status(self):
        status = []
        for camera_id, camera in self.cameras.items():
            status.append(camera.camera_status)
        return status

    def capture_frames(self):
        """
        capture stores images for all cameras given in the list
        """
        for camera_id, camera in self.cameras.items():
            camera.capture_frame()

    def set_exposure(self):
        for camera_id, camera in self.cameras.items():
            camera.set_exposure()

    # def enable_default_exposure(self):
    #     for camera_id, camera in self.cameras.items():
    #         camera.enable_default_exposure()

    def turn_on_cameras(self):
        """
        re-initialises cameras
        Returns:
            Bool status list of camera
        """
        status_list = []
        for camera_id, camera in self.cameras.items():
            status = camera.initialize_camera()
            status_list.append(status == 1)
        return status_list

    def set_flag(self):
        self.inf_flag = True

    def kill_flag(self):
        self.inf_flag = False

    def turn_off_cameras(self):
        """
        Release cameras of given IDs
        """
        for camera_id, camera in self.cameras.items():
            if hasattr(camera, "cap") and camera.cap.isOpened():
                camera.cap.release()
                Logger.log("INFO", f"Camera {camera_id} turned off.")

    def get_camera(self, camera_id: int) -> Camera:
        """
        takes in camera ID
        Returns:
            camera object of specified ID
        """
        return self.cameras.get(camera_id)

    def run_live(self, save_frequency=10):
        """
        Run the camera manager to capture frames from all cameras.
        """
        start_time = time.time()

        while not self.stop_event:
            frame_list = []
            for camera_id, camera in self.cameras.items():
                if camera.camera_status:
                    resulting_frame = camera.capture_frame()
                    if resulting_frame == None:
                        continue
                    cv2.imshow(f"Camera {camera.camera_id}", resulting_frame.frame)
                    frame_list.append(resulting_frame)
            # if self.inf_flag:
            #     img = cv2.imread(self.ML_image_path)
            #     cv2.imshow(f"ML result ",img)
            # else:
            #     cv2.destroyWindow("ML result")

            if self.inf_flag:
                img = cv2.imread(self.ML_image_path)
                if img is not None:
                    cv2.imshow("Landmark detection result", img)
                else:
                    Logger.log("WARNING", f"Failed to load image from {self.ML_image_path}")
            else:
                # Display an empty black image to 'hide' the window content
                empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow("Landmark detection result", empty_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_event = True

            if (time.time() - start_time) > save_frequency:
                start_time = time.time()
                for fr in frame_list:
                    self.save_image(fr, f"data/camera_{fr.camera_id}/{fr.timestamp}.png")

            # if self.new_landmarked_data:
            #     # update the display of the landmarked frame from its specific path
            #     pass

        for camera_id, camera in self.cameras.items():
            camera.cap.release()

        cv2.destroyAllWindows()

    def save_image(self, frame_obj, img_path):
        cv2.imwrite(img_path, frame_obj.frame)
        Logger.log("INFO", f"Camera {frame_obj.camera_id}: Image saved at {img_path}")

    def stop_live(self):
        self.stop_event = True

    def get_latest_images(self):
        latest_imgs = {}
        for camera_id, camera in self.cameras.items():
            img = camera.get_latest_image()
            if img is not None:
                latest_imgs[camera_id] = img
        return latest_imgs

    def get_latest_frames(self):
        """
        Get the latest available image frame for each camera.
        Returns:
            A dictionary with camera IDs as keys and the latest frame object as values.
        """
        latest_frames = {}
        for camera_id, camera in self.cameras.items():
            if camera.camera_status:
                resulting_frame = camera.capture_frame()
                if resulting_frame != None:
                    latest_frames[camera_id] = resulting_frame
            else:
                Logger.log("ERROR", f"No frames found for camera {camera_id}.")
                camera.log_error(CameraErrorCodes.NO_IMAGES_FOUND)
                latest_frames[camera_id] = None
        return latest_frames

    def get_available_frames(self):
        """
        Get all available image frames for each camera.
        Returns:
            A dictionary with camera IDs as keys and lists of image paths as values.
        """
        camera_frames = {}
        for camera_id, camera in self.cameras.items():
            try:
                # camera_frames.append(camera.current_frame)
                camera_frames[camera_id] = camera.all_frames
            except:
                camera.log_error(CameraErrorCodes.NO_IMAGES_FOUND)
                camera_frames[camera_id] = []
        return camera_frames

    # def return_status(self):
    #     pass

    @classmethod
    def show(self, frame: Frame):
        cv2.imshow(
            f"Camera {frame.camera_id} Frame at {frame.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            frame.frame,
        )

    @classmethod
    def close_windows(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":

    cm = CameraManager([0])
    cm.run_live()
