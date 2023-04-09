from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import cv2
import numpy as np


class ImageEnhancer(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.enhancement

    def process(self, ip_img_array):
        # Get the minimum and maximum pixel values in the image
        min_val, max_val, _, _ = cv2.minMaxLoc(ip_img_array)
        # Compute the output image using contrast stretching
        out_img = np.uint8((ip_img_array - min_val) * (255 / (max_val - min_val)))
        return out_img
