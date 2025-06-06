import webdataset as wds
from data.pipelines.base import BaseWebDatasetPipeline
from data.utils import deterministic_shuffle, log_and_continue
from data.processor import get_processor


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


class ImageCaptionPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality, data_configs, batch_size, vit_configs):
        super().__init__(modality, data_configs, batch_size)
        self.vit_configs = vit_configs
        self.processor = get_processor(data_configs.processor, vit_configs)

    def create_pipeline(self, datastring, checkpoint_num):
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=0,
                initial=0,
                seed=self.data_configs.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            # wds.map(lambda sample: {
            #     **sample,
            #     "text": self.processor.processor.apply_chat_template([
            #         {
            #             "role": "user",
            #             "content": [
            #                 {"type": "image"},
            #                 {"type": "text", "text": sample["text"]}
            #             ]
            #         }
            #     ])
            # }),
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda sample: self.processor(sample['image'], sample['text'])
            ),
            wds.map(lambda sample: {
                "input_ids": sample["input_ids"],
                "pixel_values": sample["pixel_values"],
            }),
            
        ]
        return pipeline

