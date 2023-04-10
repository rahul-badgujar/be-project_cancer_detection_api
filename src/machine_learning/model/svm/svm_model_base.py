import joblib
import numpy as np
from imutils import paths
from sklearn.metrics import accuracy_score

from src.constant.app_constants import AppConstants
from src.util.file_system_utils import FileSystemUtils


class SvmModelBase:
    def __init__(self, model_save_path, preprocess_before_training=True, preprocess_before_prediction=True):
        self.model = None
        self.model_save_path = model_save_path
        self.preprocess_before_training = preprocess_before_training
        self.preprocess_before_prediction = preprocess_before_prediction

    def get_input_feature(self, image_path):
        raise NotImplemented("Svm Model must implement get_input_feature()")

    def split_data(self, dataset_path) -> tuple:
        training_images_paths = list(paths.list_images(dataset_path))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        input_features = [self.get_input_feature(img) for img in training_images_paths]
        return np.array(input_features), np.array(labels)

    def get_model_skeleton(self):
        raise NotImplemented("Svm Model must implement get_model_skeleton()")

    def train(self) -> dict:
        x_train, y_train = self.split_data(AppConstants.training_dataset_directory)
        x_test, y_test = self.split_data(AppConstants.testing_dataset_directory)

        result = dict()
        result['configuration'] = dict()
        result['output'] = dict()

        result['configuration']['training-sample-length'] = len(x_train)
        result['configuration']['testing-sample-length'] = len(x_test)

        model=self.get_model_skeleton()

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(predictions, y_test) * 100
        print(f"Model trained with accuracy: {accuracy}")

        result['output']['accuracy'] = accuracy

        joblib.dump(model, self.model_save_path)
        print(f'Model saved at: {self.model_save_path}')

        return result

    def predict(self, img_path):
        if self.model is None:
            try:
                self.model = joblib.load(self.model_save_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Model pickle file not found at {self.model_save_path}. Please generate one.")
        ip_feature = self.get_input_feature(img_path)
        prediction = self.model.predict([ip_feature])[0]
        return AppConstants.cancer_stage_encodings.get(prediction)
