import cv2

from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor


class ImageFilterer(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.filtration

    def process(self, ip_img_array):
        filtered_img = cv2.medianBlur(ip_img_array, ksize=3)
        return filtered_img
