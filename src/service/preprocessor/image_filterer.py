import cv2

from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
from PIL import Image, ImageFilter
import numpy as np


class ImageFilterer(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.filtration

    def process(self, ip_img_from) -> str:
        img = cv2.imread(ip_img_from)
        # Convert the input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Median Filter to the input image
        filtered_img = cv2.medianBlur(gray_img, ksize=3)
        op_img_at = self.get_path_to_save_intermediate_img_output(ip_img_from)
        is_saved = cv2.imwrite(op_img_at, np.array(filtered_img))
        if is_saved:
            print(
                f'IMAGE-FILTERER:: successfully saved output img at:  {op_img_at}')
            return op_img_at
        else:
            print(
                f'IMAGE-FILTERER:: failed to save output img at:  {op_img_at}')
            raise Exception("Failed to save output image")
