from flask import Flask, request, send_file

from src.constant.app_constants import AppConstants
from src.enum.preprocessing_stage import PreprocessingStage
from src.machine_learning.model.svm.cancer_stage_detection_model import CancerStageDetectionModel
from src.machine_learning.model.svm.cancer_stage_detection_model_v2 import CancerStageDetectionModelV2
from src.machine_learning.model.svm.cancer_stage_detection_model_v3 import CancerStageDetectionModelV3
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils
from src.util.file_system_utils import FileSystemUtils

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = AppConstants.temp_file_upload_directory

cancer_stage_detection_model = CancerStageDetectionModel()
cancer_stage_detection_model_v2 = CancerStageDetectionModelV2()
cancer_stage_detection_model_version_wise_predictor = {
    "v1": lambda img_path: cancer_stage_detection_model.predict(img_path),
    "v2": lambda img_path: cancer_stage_detection_model_v2.predict(img_path),
}
cancer_stage_detection_model_version_wise_trainer = {
    "v1": lambda: cancer_stage_detection_model.train(),
    "v2": lambda: cancer_stage_detection_model_v2.train(),
}


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@app.route('/api/models/cancer_detection_model/<version>/train', methods=['GET'])
def preprocess_image(version):
    assert version == request.view_args['version']
    assert version in ('v1', 'v2'), "Invalid Model Version"
    training_result = cancer_stage_detection_model_version_wise_trainer[version]()
    return training_result


#
# @app.route('/api/preprocess/<preprocessing_stage>', methods=['POST'])
# def preprocess_image(preprocessing_stage):
#     assert preprocessing_stage == request.view_args['preprocessing_stage']
#     assert PreprocessingStage.is_valid_name(preprocessing_stage), "Invalid Preprocessor"
#     preprocessor = PreprocessingUtils.get_preprocessor_from_stage_name(preprocessing_stage)
#     file = request.files['file']
#     input_file_saved_path = FileSystemUtils.save_temp_file(file)
#     preprocessed_img_path = preprocessor.process(input_file_saved_path)
#     return send_file(preprocessed_img_path)


if __name__ == '__main__':
    app.run(debug=True)
