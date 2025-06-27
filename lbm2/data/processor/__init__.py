from transformers import AutoProcessor
from .stable_diffusion_processor import StableDiffusionProcessor


def get_processor(processor, vit_configs, **kwargs):
    if processor == "stable_diffusion":
        image_size = vit_configs.vit_img_size
        max_length = kwargs.get('max_length', 64)
        return StableDiffusionProcessor(image_size=image_size, max_length=max_length)
    elif processor is not None:
        processor = AutoProcessor.from_pretrained(processor)
        processor.image_seq_length = vit_configs.vit_img_num_tokens
        if not hasattr(processor, "image_token_id"):
            processor.image_token_id = None
        return processor
    else:
        raise ValueError(f"CustomTransform not yet supported. Use an existing HF transform.")