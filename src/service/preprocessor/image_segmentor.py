from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import cv2


class ImageSegmentor(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.segmentation

    def process(self, ip_img_array):
        _, segmented_img = cv2.threshold(ip_img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return segmented_img
