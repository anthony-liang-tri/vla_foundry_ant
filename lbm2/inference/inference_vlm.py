import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from params.params import load_params_from_json
from models import create_model
from file_utils import load_model_checkpoint


cfg = load_params_from_json("s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/config.json")
object.__setattr__(cfg.model, 'processor', 'google/paligemma-3b-pt-224')
model = create_model(cfg.model)

ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/checkpoints/checkpoint_2.pt"
load_model_checkpoint(model, ckpt, cfg.experiment.seed, cfg.distributed)


import requests
from PIL import Image
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "<image>"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(image, prompt, return_tensors="pt", padding='max_length', padding_side='right', max_length=2048)
inputs = processor(image, prompt, return_tensors="pt")
print(inputs)

out = model.generate(input_ids=inputs['input_ids'], image=inputs['pixel_values'], attention_mask=inputs['attention_mask'])
print(out)
print(processor.decode(out[0], skip_special_tokens=True))