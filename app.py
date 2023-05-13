import base64
import json
import time

import cv2
from flask import Flask, request

from flask_cors import CORS, cross_origin

from src.constant.app_constants import AppConstants
from src.enum.preprocessing_stage import PreprocessingStage
from src.machine_learning.ml_utils import opencv_img_from_base64, opencv_img_to_base64
from src.machine_learning.model.regressor.severity_prediction_model import SeverityPredictionModel
from src.machine_learning.model.svm.cancer_stage_detection_model import CancerClassificationModel
from src.machine_learning.model.svm.cancer_stage_detection_model_v2 import CancerStageDetectionModelV2
from src.machine_learning.model.svm_model_training_config import SvmModelTrainingConfig
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils
from src.util.common_utils import parse_bool
from src.util.file_system_utils import FileSystemUtils

app = Flask(__name__)
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = FileSystemUtils.get_temp_file_upload_directory()

cancer_classification_model = CancerClassificationModel()
severity_predictor_model = SeverityPredictionModel()


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@app.route('/api/models/cancer_detection_model/train', methods=['POST'])
def train_classification_model():
    training_config = SvmModelTrainingConfig(
        pretraining_preprocessing_enabled=parse_bool(request.form.get("pretraining_preprocessing_enabled"),
                                                     default_to=True),
        update_stored_model=parse_bool(request.form.get("update_stored_model"),
                                       default_to=False))
    training_result = cancer_classification_model.train(training_config)
    training_result['input_training_configuration'] = request.form
    return training_result


@app.route('/api/models/severity_prediction_model/train', methods=['POST'])
def train_severity_prediction_model():
    severity_predictor_model.train()
    return {'result': 'success'}


@app.route('/api/core/predict', methods=['POST'])
@cross_origin()
def predict_cancer_stage():
    ip_img_base64 = request.form.get("image_base64")
    assert ip_img_base64 is not None, "No input image provided"

    temp_file_path = FileSystemUtils.get_path_to_store_intermediate_file(f"temp_{time.time() * 1000}.jpg")
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(base64.b64decode(ip_img_base64))

    response_body = dict()
    response_body['input_image_base64'] = ip_img_base64
    # preprocessed images
    response_body['preprocessing_output'] = dict()
    img_arr = opencv_img_from_base64(ip_img_base64)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    enhanced_arr = PreprocessingUtils.image_enhancer.process(img_arr)
    response_body['preprocessing_output'][PreprocessingStage.enhancement.name] = opencv_img_to_base64(
        enhanced_arr).decode()
    filtered_arr = PreprocessingUtils.image_filterer.process(enhanced_arr)
    response_body['preprocessing_output'][PreprocessingStage.filtration.name] = opencv_img_to_base64(
        filtered_arr).decode()
    segmented_arr = PreprocessingUtils.image_segmentor.process(filtered_arr)
    response_body['preprocessing_output'][PreprocessingStage.segmentation.name] = opencv_img_to_base64(
        segmented_arr).decode()
    # prediction
    predicted_stage = cancer_classification_model.predict(temp_file_path)
    response_body['predicted_stage'] = predicted_stage
    response_body['predicted_scale'] = severity_predictor_model.predict(temp_file_path)
    response_body['scale_pointers']={
        'normal':3,
        'benign':5,
        'malignant':8
    }
    return response_body


if __name__ == '__main__':
    app.run(debug=True)
