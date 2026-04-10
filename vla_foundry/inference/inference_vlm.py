import sys

import requests
import torch
from PIL import Image

from vla_foundry.data.processor import get_processor
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml

if len(sys.argv) < 2:
    print("Usage: python inference_vlm.py <base_path> [checkpoint_name]")
    print("  base_path: directory containing config.yaml and checkpoints/")
    print("  checkpoint_name: (optional) checkpoint file, defaults to checkpoint_1.pt")
    sys.exit(1)

BASE_PATH = sys.argv[1]
CHECKPOINT_NAME = sys.argv[2] if len(sys.argv) > 2 else "checkpoint_1.pt"
CHECKPOINT = f"{BASE_PATH}/checkpoints/{CHECKPOINT_NAME}"

print(f"Loading config from {BASE_PATH}...")
train_params = load_params_from_yaml(TrainExperimentParams, f"{BASE_PATH}/config.yaml")

print("Creating model...")
model = create_model(train_params.model)
model = model.cuda()

print(f"Loading checkpoint from {CHECKPOINT}...")
load_model_checkpoint(model, CHECKPOINT)
model.eval()

print("Loading processor...")
processor = get_processor(train_params.data)
processor_kwargs = getattr(train_params.data, "processor_kwargs", {})

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
print(f"Fetching image from {url}...")
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(
    images=[[image]],
    text=["Describe this image."],
    return_tensors="pt",
    padding=True,
    **processor_kwargs,
)
inputs = inputs.to("cuda")

print("Running inference...")
with torch.autocast(device_type="cuda", dtype=model.dtype), torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100)

print("\n=== OUTPUT ===")
print(processor.decode(out[0], skip_special_tokens=True))
