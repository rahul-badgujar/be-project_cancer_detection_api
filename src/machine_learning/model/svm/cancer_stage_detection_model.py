import cv2
import numpy as np
from sklearn import svm

from src.machine_learning.ml_utils import quantify_image, fd_hu_moments
from src.machine_learning.model.svm.svm_model_base import SvmModelBase
from src.machine_learning.model.svm_model_training_config import SvmModelTrainingConfig
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils


class CancerStageDetectionModel(SvmModelBase):
    def __init__(self):
        super().__init__(
            model_save_path="/home/rahul/rahul/be-project/cancer-detection-api/model_saved/cancer_stage_detection_model.pkl")

    def get_input_feature(self, image_path, training_config: SvmModelTrainingConfig):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        if training_config.pretraining_preprocessing_enabled:
            image = PreprocessingUtils.apply_all_preprocessors(image)
        ft1 = quantify_image(image)
        ft2 = fd_hu_moments(image)
        return np.hstack([ft1, ft2])

    def get_model_skeleton(self):
        c = 1.0
        kernel = 'linear'
        return svm.SVC(kernel=kernel, C=c, gamma='scale')


if __name__ == '__main__':
    model = CancerStageDetectionModel()
    # result = model.predict("/home/rahul/rahul/be-project/cancer-detection-api/data/testing_set/0/1 (1).jpg")
    result = model.train()
    print(result)
