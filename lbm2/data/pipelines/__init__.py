import webdataset as wds
from itertools import islice

from lbm2.data.pipelines.text import TextPipeline
from lbm2.data.pipelines.text_untokenized import TextUntokenizedPipeline
from lbm2.data.pipelines.image_caption import ImageCaptionPipeline


class FiniteDataPipeline(wds.DataPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Iterate through up to self.nsamples steps.

        Note: wds.DataPipeline.__iter__ inexplicably only limits the number of samples with self.nsamples if
        self.repetitions != 1. Here, we always slice using self.nsamples, if self.nsamples > 0.
        """
        if self.nsamples > 0:
            return islice(self.iterator(), self.nsamples)
        else:
            return self.iterator()


def create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, data_configs):
    if modality == "text":
        pipeline = TextPipeline(modality, data_configs, batch_size)
    elif modality == "text_untokenized":
        pipeline = TextUntokenizedPipeline(modality, data_configs, batch_size)
    elif modality == "image_caption":
        pipeline = ImageCaptionPipeline(modality, data_configs, batch_size)
    else:
        raise ValueError(f"{modality} webdataset pipeline not supported")

    return FiniteDataPipeline(*pipeline.create_pipeline(datastring, checkpoint_num))