from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import numpy as np
import cv2


class ImageSegmentor(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.segmentation

    def process(self, ip_img_from) -> str:
        img = cv2.imread(ip_img_from)
        # Convert the input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding to the grayscale image
        _, segmented_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        op_img_at = self.get_path_to_save_intermediate_img_output(ip_img_from)
        is_saved = cv2.imwrite(op_img_at, segmented_img)
        if is_saved:
            print(
                f'IMAGE-SEGMENTOR:: successfully saved output img at:  {op_img_at}')
            return op_img_at
        else:
            print(
                f'IMAGE-SEGMENTOR:: failed to save output img at:  {op_img_at}')
            raise Exception("Failed to save output image")
