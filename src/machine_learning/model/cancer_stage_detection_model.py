import cv2
import numpy as np
from sklearn import svm
from imutils import paths
from sklearn.metrics import accuracy_score

from src.constant.app_constants import AppConstants
from src.machine_learning.ml_utils import quantify_image, fd_hu_moments
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils
from src.util.file_system_utils import FileSystemUtils


class CancerStageDetectionModel:
    preprocess_before_training = True

    @staticmethod
    def get_input_feature(image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (200, 200))
        features1 = quantify_image(image)
        features2 = fd_hu_moments(image)
        return np.hstack([features1, features2])

    @staticmethod
    def split_data(dataset_path) -> tuple:
        training_images_paths = list(paths.list_images(dataset_path))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        if CancerStageDetectionModel.preprocess_before_training:
            training_images_paths = [PreprocessingUtils.apply_all_preprocessors(img_path) for img_path in
                                     training_images_paths]
        input_features = [CancerStageDetectionModel.get_input_feature(img) for img in training_images_paths]
        return np.array(input_features), np.array(labels)

    @staticmethod
    def train() -> dict:
        x_train, y_train = CancerStageDetectionModel.split_data(AppConstants.training_dataset_directory)
        x_test, y_test = CancerStageDetectionModel.split_data(AppConstants.testing_dataset_directory)

        result = dict()
        result['configuration'] = dict()
        result['output'] = dict()

        result['configuration']['training-sample-length'] = len(x_train)
        result['configuration']['testing-sample-length'] = len(x_test)

        c = 1.0
        kernel = 'linear'
        model = svm.SVC(kernel=kernel, C=c, gamma='scale')

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(predictions, y_test) * 100

        result['output']['accuracy'] = accuracy

        return result


if __name__ == '__main__':
    print(CancerStageDetectionModel.train())
