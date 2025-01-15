import os
import cv2
import datetime
from itertools import cycle
from payload.vision.camera import Frame


class DemoFrames:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".png")
        ]
        self.image_files.sort()  # Ensure the files are in a consistent order
        self.image_cycle = cycle(
            self.image_files
        )  # Create an endless iterator to cycle through images

    def get_next_image_path(self):
        return next(self.image_cycle)

    def get_latest_frame(self):
        image_path = self.get_next_image_path()
        image = cv2.imread(image_path)
        if image is not None:
            timestamp = datetime.datetime.now()
            return Frame(frame=image, camera_id=0, timestamp=timestamp)
        else:
            return None


# Create an instance of DemoFrames with the specified directory
relative_path = "data/inference_input"
img_path = os.path.join(os.getcwd(), relative_path.strip("/"))
demo_frames = DemoFrames(img_path)
