from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import cv2
import numpy as np


class ImageEnhancer(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.enhancement

    def process(self, ip_img_from) -> str:
        img = cv2.imread(ip_img_from)
        # Convert the input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Get the minimum and maximum pixel values in the image
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_img)
        # Compute the output image using contrast stretching
        out_img = np.uint8((gray_img - min_val) * (255 / (max_val - min_val)))
        op_img_at = self.get_path_to_save_intermediate_img_output(ip_img_from)
        is_saved = cv2.imwrite(op_img_at, out_img)
        if is_saved:
            print(
                f'IMAGE-ENHANCER:: successfully saved output img at:  {op_img_at}')
            return op_img_at
        else:
            print(
                f'IMAGE-ENHANCER:: failed to save output img at:  {op_img_at}')
            raise Exception("Failed to save output image")
