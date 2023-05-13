import cv2
import joblib
import numpy as np
import pandas as pd
from imutils import paths
from skimage.transform import resize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.constant.app_constants import AppConstants
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils
from src.util.file_system_utils import FileSystemUtils


class CancerStageDetectionModelV3:
    @staticmethod
    def train(preprocess=True):
        resize_dimension = (160, 160)

        training_images_paths = list(paths.list_images(FileSystemUtils.get_training_dataset_directory()))
        labels = [img_path.split(FileSystemUtils.get_os_path_separator())[-2] for img_path in training_images_paths]
        input_features = []
        for img in training_images_paths:
            img_array = cv2.imread(img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = resize(img_array, resize_dimension)
            if preprocess:
                img_array = PreprocessingUtils.apply_all_preprocessors(img_array)
            input_features.append(img_array.flatten())

        flat_data = np.array(input_features)
        target = np.array(labels)
        df = pd.DataFrame(flat_data)
        df['Target'] = target
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # model creation
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, param_grid)
        # model training
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
        print('Splitted Successfully')
        model.fit(x_train, y_train)
        print('The Model is trained well with the given images')
        joblib.dump(model, f"saved-model_preprocess-{preprocess}.pkl")

        # Model testing
        y_pred = model.predict(x_test)
        print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")


if __name__ == '__main__':
    CancerStageDetectionModelV3.train(preprocess=True)
