import cv2

from src.machine_learning.model.svm.svm_model_base import SvmModelBase
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils


class CancerStageDetectionModelV2(SvmModelBase):

    def __init__(self):
        super().__init__(model_save_path="../../../../model_saved/cancer_stage_detection_model_v2.py",
                         preprocess_before_training=True)

    def get_input_feature(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image=cv2.resize(image,(200,200))
        if self.preprocess_before_training:
            image = PreprocessingUtils.apply_all_preprocessors(image)
        return image.flatten()


if __name__ == '__main__':
    model = CancerStageDetectionModelV2()
    result = model.predict("/home/rahul/rahul/be-project/cancer-detection-api/data/testing_set/0/1 (1).jpg")
    print(result)
