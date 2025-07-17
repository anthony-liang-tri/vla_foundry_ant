import torch
from lbm2.params.train_experiment_params import load_params_from_yaml
from lbm2.models import create_model
from lbm2.file_utils import load_model_checkpoint


cfg = load_params_from_yaml("s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/llm_11m/2025_07_04-05_57_53-model_transformer-lr_0.0001-bsz_2048/config.yaml")
model = create_model(cfg.model)
ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/llm_11m/2025_07_04-05_57_53-model_transformer-lr_0.0001-bsz_2048/checkpoints/checkpoint_6.pt"
load_model_checkpoint(model, ckpt, cfg.hparams.seed, cfg.distributed)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
ins = tokenizer(["hi", "This is a batch"], return_tensors='pt', padding=True)
out = model.generate(ins['input_ids'], ins['attention_mask'])
print(tokenizer.batch_decode(out))