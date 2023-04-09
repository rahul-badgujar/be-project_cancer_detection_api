from enum import Enum


class CancerStage(Enum):
    normal = 0
    adenocarcinoma = 1
    large_cell_carcinoma = 2
    squamous_cell_carcinoma = 3

    @staticmethod
    def parse_from_name(name):
        for stage in CancerStage:
            if name == stage.name:
                return stage
        return None
