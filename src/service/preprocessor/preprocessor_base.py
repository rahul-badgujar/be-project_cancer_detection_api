from src.enum.preprocessing_stage import PreprocessingStage


class Preprocessor:

    def get_preprocessing_stage(self)->PreprocessingStage:
        raise NotImplemented("Preprocessor must implement processing stage")

    def process(self,ip_img_from,op_img_at)->None:
        raise NotImplemented("Preprocessor must implement process")

