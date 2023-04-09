from enum import Enum


class PreprocessingStage(Enum):
    enhancement = 1,
    filtration = 2,
    segmentation = 3

    @staticmethod
    def names():
        return [stage.name for stage in PreprocessingStage]

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return name in PreprocessingStage.names()
