import webdataset as wds

from vla_foundry.data.augmentations.base import Augmentations
from vla_foundry.data.pipelines.base import BaseWebDatasetPipeline
from vla_foundry.data.processor import get_processor
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue, tarfile_to_samples_closing
from vla_foundry.params.base_data_params import DataParams


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


def _apply_chat_template(processor, text, num_images=1):
    """Format text with image placeholders using the processor's chat template.

    Uses the chat template (from processor or tokenizer) when available so that
    model-specific image tokens (e.g. Qwen's <|vision_start|>/<|image_pad|>)
    are inserted correctly.  Falls back to a plain ``<image>`` prefix for
    processors without a chat template (e.g. PaliGemma).
    """
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]

    if getattr(processor, "chat_template", None):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    elif hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "chat_template", None):
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return "<image> " + text


class ImageCaptionPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality: str, data_params: DataParams, batch_size: int, profile_train_steps=False):
        super().__init__(modality, data_params, batch_size)
        self.processor = get_processor(data_params)
        self.augmentations = Augmentations(data_params.augmentation)
        self.profile_train_steps = profile_train_steps


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
            tarfile_to_samples_closing(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.map(
                lambda sample: self.augmentations.apply_transforms(sample),
                handler=log_and_continue,
            ),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map(lambda sample: {
                **sample,
                "text": _apply_chat_template(self.processor, sample["text"]),
            }),
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
            # BatchFeature is preserved so batch_handlers can use .to(device)
            # to forward all VLM-specific tensor keys automatically.
            wds.map(lambda sample: (sample.pop("text", None), sample)[1]),
        ]
        return pipeline
