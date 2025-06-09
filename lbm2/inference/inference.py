import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from params.params import load_params_from_json
from models import create_model
from file_utils import load_model_checkpoint


cfg = load_params_from_json("s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/transformer_11m_sagemaker/2025_05_26-21_03_08-model_transformer-lr_0.0001-bsz_128/config.json")
model = create_model(cfg.model)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/transformer_11m_sagemaker/2025_05_26-21_03_08-model_transformer-lr_0.0001-bsz_128/checkpoints/checkpoint_4.pt"
load_model_checkpoint(model, ckpt, cfg.experiment.seed, cfg.distributed)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
ins = tokenizer(["hi", "This is a batch"], return_tensors='pt', padding=True)
out = model.generate(ins['input_ids'], ins['attention_mask'])
print(tokenizer.batch_decode(out))