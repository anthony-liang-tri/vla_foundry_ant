import requests
from PIL import Image
from transformers import AutoProcessor

from lbm2.file_utils import load_model_checkpoint
from lbm2.models import create_model
from lbm2.params.model_params import ModelParams
from lbm2.params.train_experiment_params import load_params_from_yaml

model_params = load_params_from_yaml(
    ModelParams,
    "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_07_04-01_38_18-model_vlm-lr_0.0001-bsz_128/config_model.yaml",
)
model = create_model(model_params)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_07_04-01_38_18-model_vlm-lr_0.0001-bsz_128/checkpoints/checkpoint_2.pt"
load_model_checkpoint(model, ckpt)

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "<image>"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(image, prompt, return_tensors="pt", padding='max_length', padding_side='right', max_length=2048)
inputs = processor(image, prompt, return_tensors="pt")
print(inputs)

out = model.generate(
    input_ids=inputs["input_ids"], image=inputs["pixel_values"], attention_mask=inputs["attention_mask"]
)
print(out)
print(processor.decode(out[0], skip_special_tokens=True))
