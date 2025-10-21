from transformers import AutoTokenizer

from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

model_params = load_params_from_yaml(
    ModelParams,
    "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/llm_11m/2025_07_29-20_26_02-model_transformer-lr_0.0001-bsz_2048/config_model.yaml",
)
model = create_model(model_params)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/llm_11m/2025_07_29-20_26_02-model_transformer-lr_0.0001-bsz_2048/checkpoints/checkpoint_1.pt"
load_model_checkpoint(model, ckpt)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
ins = tokenizer(["hi", "This is a batch"], return_tensors="pt", padding=True)
out = model.generate(ins["input_ids"], ins["attention_mask"])
print(tokenizer.batch_decode(out))
