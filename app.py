from flask import Flask, request, send_file

from src.constant.app_constants import AppConstants
from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessing_utils import PreprocessingUtils
from src.util.file_system_utils import FileSystemUtils

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = AppConstants.temp_file_upload_directory


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, World!'

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
