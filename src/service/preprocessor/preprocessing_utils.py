from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.image_enhancer import ImageEnhancer
from src.service.preprocessor.image_filterer import ImageFilterer
from src.service.preprocessor.image_segmentor import ImageSegmentor
from src.service.preprocessor.preprocessor_base import Preprocessor


class PreprocessingUtils:
    image_enhancer = ImageEnhancer()
    image_segmentor = ImageSegmentor()
    image_filterer = ImageFilterer()

    preprocessor_from_stage_name: dict = {
        PreprocessingStage.enhancement.name: image_enhancer,
        PreprocessingStage.filtration.name: image_filterer,
        PreprocessingStage.segmentation.name: image_segmentor
    }

    @staticmethod
    def get_preprocessor_from_stage_name(stage_name: str) -> Preprocessor:
        return PreprocessingUtils.preprocessor_from_stage_name.get(stage_name)

    @staticmethod
    def apply_all_preprocessors(img_array) -> str:
        enhanced = PreprocessingUtils.image_enhancer.process(img_array)
        filtered = PreprocessingUtils.image_filterer.process(enhanced)
        segmented = PreprocessingUtils.image_segmentor.process(filtered)
        return segmented
