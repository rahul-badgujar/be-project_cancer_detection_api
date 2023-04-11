class SvmModelTrainingConfig:
    def __init__(self, pretraining_preprocessing_enabled=True, update_stored_model=False):
        self.pretraining_preprocessing_enabled = pretraining_preprocessing_enabled
        self.update_stored_model = update_stored_model
