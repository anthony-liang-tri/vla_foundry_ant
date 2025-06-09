from transformers import AutoProcessor


class CustomTransform:
    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, image, text):
        raise NotImplementedError

def get_processor(processor, vit_configs):
    if processor is not None:
        processor = AutoProcessor.from_pretrained(processor, use_fast=True)
        if vit_configs.vit_img_num_tokens is not None:
            processor.image_seq_length = vit_configs.vit_img_num_tokens
        return processor
    else:
        raise ValueError(f"CustomTransform not yet supported. Use an existing HF transform.")