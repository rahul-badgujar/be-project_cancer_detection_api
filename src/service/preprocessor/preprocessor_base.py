from src.enum.preprocessing_stage import PreprocessingStage
from src.util.file_system_utils import FileSystemUtils


class Preprocessor:

    def get_preprocessing_stage(self) -> PreprocessingStage:
        raise NotImplemented("Preprocessor must implement processing stage")

    def process(self, ip_img_array):
        raise NotImplemented("Preprocessor must implement process")
