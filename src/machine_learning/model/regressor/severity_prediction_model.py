import cv2
import joblib
from imutils import paths
import numpy as np
from sklearn.svm import SVR
from src.util.file_system_utils import FileSystemUtils
from src.machine_learning.ml_utils import quantify_image, fd_hu_moments


class SeverityPredictionModel:

    def __init__(self):
        self.label_to_proposed_severity_value = {
            '0': 5, '1': 10, '2': 1
        }
        self.model_save_path = FileSystemUtils.join_all(
            [FileSystemUtils.get_model_save_path(), 'severity_prediction_model.pkl'])
        self.model = None

    def extract_feature(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        ft1 = quantify_image(image)
        ft2 = fd_hu_moments(image)
        return np.hstack([ft1, ft2])

    def split_data(self, dataset_path) -> tuple:
        training_images_paths = list(paths.list_images(dataset_path))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        labels = [self.label_to_proposed_severity_value[e] for e in labels]
        input_features = [self.extract_feature(img) for img in training_images_paths]
        return np.array(input_features), np.array(labels)

    def train(self):
        x_train, y_train = self.split_data(FileSystemUtils.get_training_dataset_directory())
        model = SVR()
        model.fit(x_train, y_train)
        joblib.dump(model, self.model_save_path)
        print(f'Model saved at: {self.model_save_path}')

    def predict(self, img_path):
        try:
            self.model = joblib.load(self.model_save_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model pickle file not found at {self.model_save_path}. Please generate one.")
        ip_feature = self.extract_feature(img_path)
        return self.model.predict([ip_feature])[0]
