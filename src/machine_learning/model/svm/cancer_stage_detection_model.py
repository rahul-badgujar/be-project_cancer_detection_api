import cv2
import numpy as np

from src.machine_learning.ml_utils import quantify_image, fd_hu_moments
from src.machine_learning.model.svm.svm_model_base import SvmModelBase
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils


class CancerStageDetectionModel(SvmModelBase):
    preprocess_before_training = True
    model_save_path = "../../../../model_saved/cancer_stage_detection_model.py"

    def __int__(self):
        super().__int__(model_save_path=CancerStageDetectionModel.model_save_path,
                        preprocess_before_training=CancerStageDetectionModel.preprocess_before_training)

    def get_input_feature(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        if CancerStageDetectionModel.preprocess_before_training:
            image = PreprocessingUtils.apply_all_preprocessors(image)
        ft1 = quantify_image(image)
        ft2 = fd_hu_moments(image)
        return np.hstack([ft1, ft2])


if __name__ == '__main__':
    model = CancerStageDetectionModel()
    training_result = model.train()
    print(training_result)
