import logging
from types import SimpleNamespace

import torch
from transformers import AutoProcessor

from lbm2.data.processor.stable_diffusion_processor import StableDiffusionProcessor
from lbm2.data.utils import text_to_seed
from lbm2.params.base_data_params import DataParams


class DebugProcessor:
    def __init__(self):
        self.image_token_id = 0
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def __call__(self, images, text, return_tensors="pt", padding="max_length", padding_side="right", max_length=2048):
        batch_size = len(images)
        seed = text_to_seed(text[0])
        return {
            "input_ids": torch.randint(0, 100, (batch_size, max_length), generator=torch.Generator().manual_seed(seed)),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224, generator=torch.Generator().manual_seed(seed)),
        }


def get_processor(data_params: DataParams):
    if data_params.processor == "stable_diffusion":
        return StableDiffusionProcessor(
            image_size=data_params.image_size,
            max_length=data_params.seq_len,
        )
    elif data_params.processor == "debug":
        return DebugProcessor()
    elif data_params.processor is not None:
        processor = AutoProcessor.from_pretrained(data_params.processor)
        processor.image_seq_length = data_params.img_num_tokens

        # Set image size for processors if specified in config
        processor_name = str(data_params.processor).lower() if hasattr(data_params, "processor") else ""
        image_size = data_params.get("image_size")
        if image_size and hasattr(processor, "image_processor"):
            # Different processors expect different size formats
            if "paligemma" in processor_name or "pali-gemma" in processor_name:
                # PaliGemma expects height and width
                processor.image_processor.size = {"height": int(image_size), "width": int(image_size)}
                logging.debug(
                    f"Set processor image_processor.size to {{'height': {image_size}, 'width': {image_size}}}"
                )
            else:
                # SmolVLM and others expect longest_edge
                processor.image_processor.size = {"longest_edge": int(image_size)}
                logging.debug(f"Set processor image_processor.size to {{'longest_edge': {image_size}}}")

        return processor
    else:
        raise ValueError(f"{data_params.processor} not yet supported.")
