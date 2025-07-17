from transformers import AutoProcessor
from lbm2.data.processor.stable_diffusion_processor import StableDiffusionProcessor


def get_processor(data_configs):
    if data_configs.processor == "stable_diffusion":
        return StableDiffusionProcessor(
            image_size=data_configs.image_size, 
            max_length=data_configs.seq_len,
        )
    elif data_configs.processor is not None:
        processor = AutoProcessor.from_pretrained(data_configs.processor)
        processor.image_seq_length = data_configs.img_num_tokens
        return processor
    else:
        raise ValueError(f"{data_configs.processor} not yet supported.")