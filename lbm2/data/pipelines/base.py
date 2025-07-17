class BaseWebDatasetPipeline:
    def __init__(self, modality, data_configs, batch_size):
        self.modality = modality
        self.data_configs = data_configs
        self.batch_size = batch_size

    def create_pipeline(self, datastring, checkpoint_num):
        raise NotImplementedError("Implemented in individual classes.")
