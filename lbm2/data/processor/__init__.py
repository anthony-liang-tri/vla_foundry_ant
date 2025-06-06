from data.processor.image_transforms import *
from transformers import AutoProcessor


class MyTransform:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", use_fast=True)
    
    def __call__(self, image, text):
        # print("image input : ", image)
        # out = self.processor(image, text, return_tensors='pt', padding='max_length', max_length=2049)
        # for i in out:
        #     print("I: ", i, "i shape: ", out[i].shape)
        return self.processor(image, text, return_tensors='pt', padding='max_length', max_length=2049)


def get_processor(processor, vit_configs):
    if processor is not None:
        return AutoProcessor.from_pretrained(processor, use_fast=True)

    return MyTransform()