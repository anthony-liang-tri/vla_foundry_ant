from lbm2.file_utils import load_model_checkpoint
from lbm2.models import create_model
from lbm2.params.train_experiment_params import load_params_from_yaml

EXPECTED_OUTPUT_TEXT = ["hi[PAD][PAD][PAD]....................", "This is a batch...................."]


def test_inference_text():
    cfg = load_params_from_yaml("tests/shared/tiny_model/config.yaml")
    model = create_model(cfg.model)
    ckpt = "tests/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt, cfg.hparams.seed, cfg.distributed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    ins = tokenizer(["hi", "This is a batch"], return_tensors="pt", padding=True)
    out = model.generate(ins["input_ids"], ins["attention_mask"])
    assert tokenizer.batch_decode(out) == EXPECTED_OUTPUT_TEXT


### Need to find a way to make this faster. Probably load a smaller model.
# def test_inference_vlm():
#     cfg = load_params_from_yaml("s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/config.json")
#     object.__setattr__(cfg.model, 'processor', 'google/paligemma-3b-pt-224')
#     model = create_model(cfg.model)

#     ckpt = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/checkpoints/checkpoint_1.pt"
#     load_model_checkpoint(model, ckpt, cfg.experiment.seed, cfg.distributed)


#     import requests
#     from PIL import Image
#     from transformers import AutoProcessor
#     processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

#     prompt = "<image>"
#     url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
#     image = Image.open(requests.get(url, stream=True).raw)
#     inputs = processor(image, prompt, return_tensors="pt")
#     print(inputs)

#     out = model.generate(input_ids=inputs['input_ids'],
#                           image=inputs['pixel_values'], attention_mask=inputs['attention_mask'])
#     print(out)
#     print(processor.decode(out[0], skip_special_tokens=True))
