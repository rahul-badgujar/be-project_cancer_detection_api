from flask import Flask, request, send_file

from src.constant.app_constants import AppConstants
from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.image_enhancer import ImageEnhancer
from src.service.preprocessor.image_filterer import ImageFilterer
from src.service.preprocessor.image_segmentor import ImageSegmentor
from src.util.file_system_utils import FileSystemUtils

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = AppConstants.temp_file_upload_directory

preprocessor_from_stage_name: dict = {
    PreprocessingStage.enhancement.name: ImageEnhancer(),
    PreprocessingStage.filtration.name: ImageFilterer(),
    PreprocessingStage.segmentation.name: ImageSegmentor()
}


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@app.route('/api/preprocess/<preprocessor>', methods=['POST'])
def preprocess_image(preprocessor):
    assert preprocessor == request.view_args['preprocessor']
    assert PreprocessingStage.is_valid_name(preprocessor), "Invalid Preprocessor"
    preprocessor = preprocessor_from_stage_name.get(preprocessor)
    file = request.files['file']
    input_file_saved_path = FileSystemUtils.save_temp_file(file)
    preprocessed_file_save_path = FileSystemUtils.join(AppConstants.temp_preprocessed_file_save_directory,
                                                       FileSystemUtils.append_filename_with(
                                                           FileSystemUtils.get_filename(file), f'_{preprocessor}'))
    preprocessor.process(input_file_saved_path, preprocessed_file_save_path)
    return send_file(preprocessed_file_save_path)


if __name__ == '__main__':
    app.run(debug=True)
