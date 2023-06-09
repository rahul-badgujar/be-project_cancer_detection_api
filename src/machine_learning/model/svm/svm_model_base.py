import time

import joblib
import numpy as np
from imutils import paths
from sklearn.metrics import accuracy_score, confusion_matrix

from src.constant.app_constants import AppConstants
from src.machine_learning.ml_utils import calculate_metrics_from_3x3_confusion_matrix
from src.machine_learning.model.svm_model_training_config import SvmModelTrainingConfig
from src.util.file_system_utils import FileSystemUtils


class SvmModelBase:
    def __init__(self, model_save_path):
        self.model = None
        self.model_save_path = model_save_path

    def get_input_feature(self, image_path, training_config: SvmModelTrainingConfig):
        raise NotImplemented("Svm Model must implement get_input_feature()")

    def split_data(self, dataset_path, training_config: SvmModelTrainingConfig) -> tuple:
        training_images_paths = list(paths.list_images(dataset_path))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        input_features = [self.get_input_feature(img, training_config) for img in training_images_paths]
        return np.array(input_features), np.array(labels)

    def get_model_skeleton(self):
        raise NotImplemented("Svm Model must implement get_model_skeleton()")

    def train(self, training_config: SvmModelTrainingConfig) -> dict:
        result = dict()
        result['internal_training_specifications'] = dict()
        result['output'] = dict()

        # model training
        x_train, y_train = self.split_data(FileSystemUtils.get_training_dataset_directory(), training_config)
        result['internal_training_specifications']['training_sample_length'] = len(x_train)

        model = self.get_model_skeleton()

        training_start_time = time.time_ns()
        model.fit(x_train, y_train)
        training_end_time = time.time_ns()
        result['output']['training_time'] = f'{training_end_time - training_start_time} ns'

        # model evaluation
        x_test, y_test = self.split_data(FileSystemUtils.get_testing_dataset_directory(), training_config)
        result['internal_training_specifications']['testing_sample_length'] = len(x_test)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Model trained with accuracy: {accuracy}")
        result['output']['accuracy'] = accuracy

        conf_matrix_result = dict()
        cm = confusion_matrix(y_test, y_pred)
        tp, tn, fp, fn = calculate_metrics_from_3x3_confusion_matrix(cm)
        conf_matrix_result['true_negative'] = tn
        conf_matrix_result['false_positive'] = fp
        conf_matrix_result['false_negative'] = fn
        conf_matrix_result['true_positive'] = tp
        result['output']['confusion_matrix'] = conf_matrix_result

        # saving model if applicable
        if training_config.update_stored_model:
            joblib.dump(model, self.model_save_path)
            print(f'Model saved at: {self.model_save_path}')

        return result

    def predict(self, img_path):
        if self.model is None:
            try:
                self.model = joblib.load(self.model_save_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Model pickle file not found at {self.model_save_path}. Please generate one.")
        ip_feature = self.get_input_feature(img_path, SvmModelTrainingConfig(pretraining_preprocessing_enabled=True))
        prediction = self.model.predict([ip_feature])[0]
        return AppConstants.cancer_stage_encodings.get(prediction)
