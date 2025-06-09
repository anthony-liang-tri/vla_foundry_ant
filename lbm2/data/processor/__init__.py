from transformers import AutoProcessor


class CustomTransform:
    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, image, text):
        raise NotImplementedError

def get_processor(processor, vit_configs):
    if processor is not None:
        return AutoProcessor.from_pretrained(processor, use_fast=True)
    else:
        raise ValueError(f"CustomTransform not yet supported. Use an existing HF transform.")