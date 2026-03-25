import requests
from PIL import Image

from vla_foundry.data.processor import get_processor
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml

BASE_PATH = "s3://tri-ml-datasets/vla_foundry_scratch/models/vlm_smolvlm_fromllm_samples50m/2026_03_05-00_34_13-model_vlm-lr_0.0001-bsz_512"
train_params = load_params_from_yaml(TrainExperimentParams, f"{BASE_PATH}/config.yaml")
model = create_model(train_params.model)
print("model: ", model)
load_model_checkpoint(model, f"{BASE_PATH}/checkpoints/checkpoint_14.pt")

processor = get_processor(train_params.data)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image_token = processor.image_token if hasattr(processor, "image_token") else "<image>"
inputs = processor(image, image_token, return_tensors="pt")
print("inputs: ", inputs)
if inputs["pixel_values"].dim() == 5:
    inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
