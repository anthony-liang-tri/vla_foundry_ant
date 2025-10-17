import argparse
import os

import torch
from PIL import Image

from lbm2.file_utils import load_model_checkpoint
from lbm2.models import create_model
from lbm2.params.model_params import ModelParams
from lbm2.params.train_experiment_params import load_params_from_yaml

"""
This script generates images using a Stable Diffusion model.
Example usage:
    With prompt (text conditioning): uv run lbm2/inference/inference_sd.py --prompt="A picture of a cat"
    Without prompt: uv run lbm2/inference/inference_sd.py
"""

args = argparse.ArgumentParser()
args.add_argument("--prompt", type=str, default="tree")
prompt = args.parse_args().prompt


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# Load config and model
model_params = load_params_from_yaml(
    ModelParams,
    "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/stable_diffusion_cfg/2025_10_16-04_40_27-model_stable_diffusion-lr_0.001-bsz_1024/config.yaml",
)
model = create_model(model_params)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/stable_diffusion_cfg/2025_10_16-04_40_27-model_stable_diffusion-lr_0.001-bsz_1024/checkpoints/checkpoint_12.pt"
load_model_checkpoint(model, ckpt)

model = model.to("cuda")
images = model.generate(batch_size=16, device=torch.device("cuda"), prompt=prompt)  # numpy [16, 224, 224, 3]

# Make a grid and save the images
image_grid = make_grid(images, rows=4, cols=4)
output_dir = "experiments/"
os.makedirs(output_dir, exist_ok=True)
image_grid.save(f"{output_dir}/stable_diffusion_output_{prompt}.png")
