"""
Machine Learning Pipeline for Region Classification and Landmark Detection

This script defines a machine learning pipeline that processes a series of frames from camera feeds, 
performs region classification to identify geographic regions within the frames, and subsequently 
detects landmarks within those regions. The script is designed to handle varying lighting conditions 
by discarding frames that are deemed too dark for reliable classification or detection.

Author: Eddie
Date: [Creation or Last Update Date]
"""

# import necessary modules
from PIL import Image
import cv2
from flight.vision.rc import RegionClassifier
from flight.vision.ld import LandmarkDetector
from flight import Logger
import os


class Landmark:
    """
    A class to store landmark info including centroid coordinates, geographic coordinates, and classes.

    Attributes:
        centroid_xy (list of tuples): The centroid coordinates (x, y) of detected landmarks.
        centroid_latlons (list of tuples): The geographic coordinates (latitude, longitude) of detected landmarks.
        landmark_classes (list): The classes of the detected landmarks.
    """

    def __init__(self, centroid_xy, centroid_latlons, landmark_classes, confidence_scores):
        """
        Initializes the Landmark

        Args:
            centroid_xy (list of tuples): Centroid coordinates (x, y) of detected landmarks.
            centroid_latlons (list of tuples): Geographic coordinates (latitude, longitude) of detected landmarks.
            landmark_classes (list): Classes of detected landmarks.
        """
        self.centroid_xy = centroid_xy
        self.centroid_latlons = centroid_latlons
        self.landmark_classes = landmark_classes
        self.confidence_scores = confidence_scores

    def __repr__(self):
        return f"Landmark(centroid_xy={self.centroid_xy}, centroid_latlons={self.centroid_latlons}, landmark_classes={self.landmark_classes}, confidence_scores={self.confidence_scores})"


class MLPipeline:
    """
    A class representing a machine learning pipeline for processing camera feed frames for
    region classification and landmark detection.

    Attributes:
        region_classifier (RegionClassifier): An instance of RegionClassifier for classifying geographic regions in frames.
    """

    def __init__(self):
        """
        Initializes the MLPipeline class, setting up any necessary components for the machine learning tasks.
        """
        self.region_classifier = RegionClassifier()
        self.region_to_location = {
            '10S': 'California',
            '10T': 'Washington / Oregon',
            '11R': 'Baja California, Mexico',
            '12R': 'Sonora, Mexico',
            '16T': 'Minnesota / Wisconsin / Iowa / Illinois',
            '17R': 'Florida',
            '17T': 'Toronto, Canada / Michigan / OH / PA',
            '18S': 'New Jersey / Washington DC',
            '32S': 'Tunisia (North Africa near Tyrrhenian Sea)',
            '32T': 'Switzerland / Italy / Tyrrhenian Sea',
            '33S': 'Sicilia, Italy',
            '33T': 'Italy / Adriatic Sea',
            '52S': 'Korea / Kumamoto, Japan',
            '53S': 'Hiroshima to Nagoya, Japan',
            '54S': 'Tokyo to Hachinohe, Japan',
            '54T': 'Sapporo, Japan'
        }


    def classify_frame(self, frame_obj):
        """
        Classifies a frame to identify geographic regions using the region classifier.

        Args:
            frame_obj (Frame): The Frame object to classify.

        Returns:
            list: A list of predicted region IDs classified from the frame.
        """
        predicted_list = self.region_classifier.classify_region(frame_obj)
        return predicted_list

    def run_ml_pipeline_on_batch(self, frames):
        """
        Processes a series of frames, classifying each for geographic regions and detecting landmarks,
        and returns the detection results along with camera IDs.

        Args:
            frames (list of Frame): A list of Frame objects.

        Returns:
            list of tuples: Each tuple consists of the camera ID and the landmark detection results for that frame.
        """
        results = []
        for frame_obj in frames:
            pred_regions = self.classify_frame(frame_obj)
            frame_results = []
            for region in pred_regions:
                detector = LandmarkDetector(region_id=region)
                centroid_xy, centroid_latlons, landmark_classes, confidence_scores = detector.detect_landmarks(
                    frame_obj.frame
                )
                landmark = Landmark(centroid_xy, centroid_latlons, landmark_classes, confidence_scores)
                frame_results.append((region, landmark))
            results.append((frame_obj.camera_id, frame_results))
        return results

    def run_ml_pipeline_on_single(self, frame_obj):
        """
        Processes a single frame, classifying it for geographic regions and detecting landmarks,
        and returns the detection result along with the camera ID.

        Args:
            frame_obj (Frame): The Frame object to process.

        Returns:
            tuple: The camera ID and the landmark detection results for the frame.
        """
        Logger.log(
            "INFO",
            "------------------------------Inference---------------------------------",
        )
        pred_regions = self.classify_frame(frame_obj)
        if len(pred_regions) == 0:
            Logger.log(
                "INFO",
                f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] No landmarks detected. ",
            )
            return None
        frame_results = []
        for region in pred_regions:
            detector = LandmarkDetector(region_id=region)
            centroid_xy, centroid_latlons, landmark_classes, confidence_scores = detector.detect_landmarks(frame_obj)
            if (
                centroid_xy is not None
                and centroid_latlons is not None
                and landmark_classes is not None
            ):
                landmark = Landmark(centroid_xy, centroid_latlons, landmark_classes, confidence_scores)
                frame_results.append((region, landmark))
            else:
                continue
        # Use the class method to update landmarks
        frame_obj.update_landmarks(frame_results)
        return frame_results

    def adjust_color(self, color, confidence):
        # Option 1: Exponential scaling
        # scale_factor = (confidence ** 2)  # Square the confidence to exaggerate differences
        
        # Option 2: Offset and scaling adjustment
        # This ensures that even low confidence values have a noticeable color intensity
        #min_factor = 0.5  # Ensure that even the lowest confidence gives us at least half the color intensity
        #scale_factor = min_factor + (1 - min_factor) * confidence
        
        # Option 3: Squared scaling (chosen for demonstration)
        # More dramatic effect as confidence increases
        scale_factor = confidence ** 2
        
        # Apply the scale factor to the color components
        adjusted_color = tuple(int(c * scale_factor) for c in color)
        
        return adjusted_color

    def visualize_landmarks(self, frame_obj, regions_and_landmarks, save_dir):
        """
        Draws larger centroids of landmarks on the frame, adds a larger legend for region colors with semi-transparent boxes,
        and saves the image. Also displays camera metadata on the image.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = frame_obj.frame.copy()

        colors = [
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (0, 165, 255),    # Orange
            (180, 105, 255),  # Pink
            (255, 0, 0),      # Blue
        ]

        # ============================== landmark display ==================================
        region_color_map = {}
        circle_radius = 15
        circle_thickness = -1
        
        top_landmarks = []  # List to store top landmarks for display

        for idx, (region, detection_result) in enumerate(regions_and_landmarks):
            base_color = colors[idx % len(colors)]
            region_color_map[region] = base_color

            for (x, y), confidence, cls in zip(detection_result.centroid_xy, detection_result.confidence_scores, detection_result.landmark_classes):
                adjusted_color = self.adjust_color(base_color, confidence)
                cv2.circle(image, (int(x), int(y)), circle_radius, adjusted_color, circle_thickness)
                
                # Collect data for top landmarks
                top_landmarks.append((region, confidence, (x, y), detection_result.centroid_latlons))

        # Sort landmarks by confidence, descending, and keep the top 5
        top_landmarks.sort(key=lambda x: x[1], reverse=True)
        top_landmarks = top_landmarks[:5]
        
        # ========================== Metadata displaying ========================================
        # Metadata drawing first to determine right edge for alignment
        metadata_info = f"Camera ID: {frame_obj.camera_id} | Time: {frame_obj.timestamp} | Frame: {frame_obj.frame_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        metadata_font_scale = 1
        text_thickness = 2
        metadata_text_size = cv2.getTextSize(metadata_info, font, metadata_font_scale, text_thickness)[0]
        metadata_text_x = image.shape[1] - metadata_text_size[0] - 10  # Right align
        metadata_text_y = 30
        metadata_box_height = metadata_text_size[1] + 20  # Some padding

        # Draw semi-transparent rectangle for metadata
        overlay = image.copy()
        cv2.rectangle(overlay, (metadata_text_x, metadata_text_y - metadata_text_size[1] - 10), (metadata_text_x + metadata_text_size[0] + 10, metadata_text_y + 10), (50, 50, 50), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        # Place metadata text
        cv2.putText(image, metadata_info, (metadata_text_x, metadata_text_y), font, metadata_font_scale, (255, 255, 255), text_thickness)

        # Prepare for Top Landmarks box
        top_legend_x = metadata_text_x  # Align with the left edge of metadata text
        top_legend_y = metadata_text_y + metadata_box_height + 20  # Spacing below the metadata box

        # Top landmarks settings
        top_font_scale = 1  # Three times bigger
        entry_height = int(cv2.getTextSize("Test", font, top_font_scale, 1)[0][1] * 1.5)  # Adjusted entry height
        max_width = 0
        total_height = 0

        text_entries = []
        for i, (cls, confidence, (x, y), latlons) in enumerate(top_landmarks):
            latitude = latlons[0].item(0)
            longitude = latlons[1].item(0)
            text = f"Top {i+1}: Region {region}, Conf: {confidence:.2f}, XY: ({int(x)}, {int(y)}), LatLon: ({latitude:.2f}, {longitude:.2f})"
            text_size = cv2.getTextSize(text, font, top_font_scale, 1)[0]
            max_width = max(max_width, text_size[0] + 20)  # Update max width
            total_height += entry_height
            text_entries.append((text, top_legend_y + total_height))

        # Draw semi-transparent rectangle for top landmarks
        overlay = image.copy()
        cv2.rectangle(overlay, (top_legend_x, top_legend_y), (top_legend_x + max_width, top_legend_y + total_height + 10), (50, 50, 50), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        # Place each text entry
        for text, y_position in text_entries:
            cv2.putText(image, text, (top_legend_x + 10, y_position), font, top_font_scale, (255, 255, 255), 2)
        # ==================== Region Legend =======================
        legend_x = 10
        legend_y = 30
        font_scale_legend = 1.5
        text_thickness_legend = 3
        for region, color in region_color_map.items():
            location = self.region_to_location.get(region, 'Unknown Location')  # Get the location name or default to 'Unknown Location'
            text = f"Region {region}: {location}"
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale_legend, text_thickness_legend)
            overlay = image.copy()
            # Draw a semi-transparent rectangle
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + text_width, legend_y + text_height + 10), color, -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            # Put the text on the image
            cv2.putText(image, text, (legend_x, legend_y + text_height), font, font_scale_legend, (255, 255, 255), text_thickness_legend)
            # Move down for the next entry
            legend_y += text_height + 10

        landmark_save_path = os.path.join(save_dir, f"frame_w_landmarks_{frame_obj.camera_id}.png")
        cv2.imwrite(landmark_save_path, image)

        img_save_path = os.path.join(save_dir, "frame.png")
        cv2.imwrite(img_save_path, frame_obj.frame)

        metadata_path = os.path.join(save_dir, "frame_metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Camera ID: {frame_obj.camera_id}\n")
            f.write(f"Timestamp: {frame_obj.timestamp}\n")
            f.write(f"Frame ID: {frame_obj.frame_id}\n")

        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] Landmark visualization saved to data/inference_output",
        )

