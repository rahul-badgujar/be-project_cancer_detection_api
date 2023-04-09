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
    @staticmethod
    def get_input_feature(image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (200, 200))
        # # threshold the image such that the drawing appears as white
        # # on a black background
        # image = cv2.threshold(image, 0, 255,
        #                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features1 = quantify_image(image)
        features2 = fd_hu_moments(image)
        return np.hstack([features1, features2])

    @staticmethod
    def split_data(dataset_path) -> tuple:
        training_images_paths = list(paths.list_images(dataset_path))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        training_images_preprocessed = [PreprocessingUtils.apply_all_preprocessors(img_path) for img_path in
                                        training_images_paths]
        # training_images_preprocessed = [img_path for img_path in
        #                                 training_images_paths]
        input_features = [CancerStageDetectionModel.get_input_feature(img) for img in training_images_preprocessed]
        return input_features, labels

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
        model = svm.SVC(kernel=kernel, C=c)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(predictions, y_test) * 100

        result['output']['accuracy'] = accuracy

        return result


if __name__ == '__main__':
    print(CancerStageDetectionModel.train())
