import webdataset as wds

from lbm2.data.pipelines.base import BaseWebDatasetPipeline
from lbm2.data.processor import get_processor
from lbm2.data.utils import deterministic_shuffle, log_and_continue
from lbm2.params.base_data_params import DataParams


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


class ImageCaptionPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality: str, data_params: DataParams, batch_size: int):
        super().__init__(modality, data_params, batch_size)
        self.processor = get_processor(data_params)

    def create_pipeline(self, datastring: str, checkpoint_num: int):
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_params.shuffle_buffer_size,
                initial=self.data_params.shuffle_initial,
                seed=self.data_params.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map(lambda sample: {**sample, "text": "<image> " + sample["text"]}),
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda sample: self.processor(
                    images=sample["image"],
                    text=sample["text"],
                    return_tensors="pt",
                    padding="max_length",
                    padding_side="right",
                    max_length=self.data_params.seq_len + 1,
                ),
                handler=log_and_continue,
            ),
            wds.map(
                lambda sample: {
                    "input_ids": sample["input_ids"],
                    "attention_mask": sample["attention_mask"],
                    "pixel_values": sample["pixel_values"],
                }
            ),
        ]
        return pipeline
