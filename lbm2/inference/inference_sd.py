import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from params.params import load_params_from_json
from models import create_model
from file_utils import load_model_checkpoint
from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# Load config and model
cfg = load_params_from_json("s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/stable_diffusion/2025_06_18-19_02_29-model_stable_diffusion-lr_0.0001-bsz_1024/config.json")
model = create_model(cfg.model)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/stable_diffusion/2025_06_18-19_02_29-model_stable_diffusion-lr_0.0001-bsz_1024/checkpoints/checkpoint_6.pt"
load_model_checkpoint(model, ckpt, cfg.experiment.seed, cfg.distributed)

model = model.to("cuda")
images = model.generate(batch_size=16, device=torch.device("cuda"))     # numpy [16, 224, 224, 3]

# Make a grid and save the images
image_grid = make_grid(images, rows=4, cols=4)
image_grid.save(f"stable_diffusion_output.png")