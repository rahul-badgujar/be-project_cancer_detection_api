from src.enum.preprocessing_stage import PreprocessingStage
from src.util.file_system_utils import FileSystemUtils


class Preprocessor:

    def get_preprocessing_stage(self) -> PreprocessingStage:
        raise NotImplemented("Preprocessor must implement processing stage")

    def process(self, ip_img_from) -> str:
        raise NotImplemented("Preprocessor must implement process")

    def get_path_to_save_intermediate_img_output(self, ip_filepath):
        ip_filename = FileSystemUtils.get_filename_from_path(ip_filepath)
        intermediate_filename = FileSystemUtils.append_filename_with(ip_filename,
                                                                     f"_{self.get_preprocessing_stage().name}")
        return FileSystemUtils.get_path_to_store_intermediate_file(intermediate_filename)
