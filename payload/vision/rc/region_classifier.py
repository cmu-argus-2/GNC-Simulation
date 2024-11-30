"""
Region Classification Module

This module defines the RegionClassifier class, which leverages a pretrained EfficientNet model to classify 
images based on geographic regions. The classifier is tailored to recognize specific regions by adjusting the 
final layer to match the number of target classes and loading custom model weights. Main functionalities 
include image preprocessing and the execution of classification, providing class probabilities for each 
recognized region.


Author: Eddie
Date: [Creation or Last Update Date]
"""


import os
import yaml
import cv2
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from flight import Logger

LD_MODEL_SUF = ".pth"
NUM_CLASS = 16

# Define error and info messages
error_messages = {
    "CONFIGURATION_ERROR": "Configuration error.",
    "MODEL_LOADING_FAILED": "Failed to load model.",
    "CLASSIFICATION_FAILED": "Classification process failed.",
}

info_messages = {
    "INITIALIZATION_START": "Initializing RegionClassifier.",
    "MODEL_LOADED": "Model loaded successfully.",
    "CLASSIFICATION_START": "Starting the classification process.",
}


class ClassifierEfficient(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierEfficient, self).__init__()
        # Using new weights system
        # This uses the most up-to-date weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        for param in self.efficientnet.features[:3].parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x


class RegionClassifier:
    def __init__(self):
        Logger.log("INFO", info_messages["INITIALIZATION_START"])

        model_path, config_path = self.construct_paths()

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ClassifierEfficient(NUM_CLASS).to(self.device)

            # Load Custom model weights
            model_weights_path = os.path.join(model_path, "model_effnet_0.997_acc" + LD_MODEL_SUF)
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            self.model.eval()
            Logger.log("INFO", info_messages["MODEL_LOADED"])

        except Exception as e:
            Logger.log("ERROR", f"{error_messages['MODEL_LOADING_FAILED']}: {e}")
            raise

        # Define the preprocessing
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.region_ids = self.load_region_ids(config_path)

    def construct_paths(self):
        model_path = os.path.join("models", "rc")
        config_path = os.path.join("configuration", "inference_config.yml")
        return model_path, config_path

    def load_region_ids(self, config_path):
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            return config.get("region_ids", [])
        except Exception as e:
            Logger.log("ERROR", f"{error_messages['CONFIGURATION_ERROR']}: {e}")
            raise

    def classify_region(self, frame_obj):
        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {info_messages['CLASSIFICATION_START']}",
        )
        predicted_region_ids = []
        inference_time = 0
        try:
            img = Image.fromarray(cv2.cvtColor(frame_obj.frame, cv2.COLOR_BGR2RGB))
            img = self.transforms(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(img)
                end_time = time.time()
                inference_time = end_time - start_time

                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.55).float()
                predicted_indices = predicted.nonzero(as_tuple=True)[1]
                predicted_region_ids = [self.region_ids[idx] for idx in predicted_indices]

        except Exception as e:
            Logger.log("ERROR", f"{error_messages['CLASSIFICATION_FAILED']}: {e}")
            raise

        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {predicted_region_ids} region(s) identified.",
        )
        Logger.log(
            "INFO",
            f"Inference completed in {inference_time:.2f} seconds."
        )
        return predicted_region_ids
