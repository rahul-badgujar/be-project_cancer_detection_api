from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import numpy as np
import cv2


class ImageSegmentor(Preprocessor):
    def get_preprocessing_stage(self) -> PreprocessingStage:
        return PreprocessingStage.segmentation

    def process(self, ip_img_from) ->str:
        print(f'IMAGE-SEGMENTOR:: request to process img:  {ip_img_from}')

        print(f'IMAGE-SEGMENTOR:: loading img:  {ip_img_from}')
        img = cv2.imread(ip_img_from, 0)
        print(f'IMAGE-FILTERER:: successfully loaded img:  {ip_img_from}')

        bins_num = 256
        hist, bin_edges = np.histogram(img, bins=bins_num)

        is_normalized = True
        if is_normalized:
            hist = np.divide(hist.ravel(), hist.max())

        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

        inter_class_variance = weight1[:-1] * weight2[1:] * (
                mean1[:-1] - mean2[1:]) ** 2

        index_of_max_val = np.argmax(inter_class_variance)

        threshold = bin_mids[:-1][index_of_max_val]

        otsu_threshold, segmented_img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        print(f'IMAGE-SEGMENTOR: successfully generated output img')
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
