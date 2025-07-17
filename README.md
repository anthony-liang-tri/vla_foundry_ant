# lbm2

This was put together from some combination of [MBM](https://github.com/TRI-ML/mbm) and [nanoVLM](https://github.com/huggingface/nanoVLM/tree/main).

## Installation
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. Please follow the uv documentation for installation. Once uv is installed, create a Python 3.10 virtual environment using uv and install the project dependencies with the command below:
```bash
uv sync
uv pip install -e .
```
The recommended workflow is to run scripts directly with `uv` via `uv run <script> <args>`.
Alternatively, to activate the virtual env you can then run `source .venv/bin/activate` then proceed as usual (though should still use `uv` for package and depedendency management).

## Quickstart
The main entrypoint is `lbm2/main.py`.

An example command is something like this:
```bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 lbm2/main.py \
--model.type vlm \
--model.transformer.load_path lbm2/config_presets/models/vlm_3b.yaml \
--model.vit.load_path lbm2/config_presets/models/vit_paligemma.yaml \
--data.type image_caption \
--data.processor google/paligemma-3b-pt-224 \
--data.dataset_manifest ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"] \
--data.dataset_modality ["image_caption"] \
--data.dataset_weighting [1.0] \
--data.img_num_tokens 256 \
--total_train_samples 14_000_000 \
--num_checkpoints 5 \
--hparams.per_gpu_batch_size 2 \
--hparams.global_batch_size 64 \
--remote_sync s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b
```

See `./examples` for more examples. [llm_11m.sh](examples/llm_11m.sh) is a good place to start. 

### Running on SageMaker
Create a `secrets.env` file in the root directory:
```bash
WANDB_API_KEY=<your wandb key>
HF_TOKEN=<your hf token>
```

To launch something on SageMaker, the launch file is [sagemaker/launch_training.py](sagemaker/launch_training.py).
Pass arguments in a similar way as you would for a local run.
Note that you have to run this with `uv run --group sagemaker`. This creates a temporary venv used in running the script
where sagemaker is installed. The reason to do this is so that the local and sagemaker environments match
(when sagemaker is installed it changes other dependencies).

```bash
uv run --group sagemaker sagemaker/launch_training.py \
--user your.user.name \
--instance_count 1 \
--instance_type p4de \
--insert_your_arguments_here
```

## Repo Structure and Implementation
The sections below highlight several key design choices and functionalities of the repo.

### 1. Param/Argument Structure
We use [draccus](https://github.com/dlwh/draccus) for argument parsing. Params are defined in the [lbm2/params](lbm2/params) folder. We use nested parameters. There is a high level `cfg` dataclass object in [lbm2/main.py](lbm2/main.py). This dataclass has attributes which are dataclasses themselves, namely `cfg.model`, `cfg.hparams`, `cfg.data`, and `cfg.distributed`, which themselves contain attributes like `cfg.model.hidden_dim`.

#### 1.1 Argument Parsing Usage
Below we show an example of how we supply arguments (see [examples](lbm2/examples) folder for more):
```bash
--model.type transformer \
--model.load_path lbm2/config_presets/models/transformer_11m.yaml \
--distributed.fsdp True \
--distributed.fsdp_use_orig_params True \
--distributed.fsdp_limit_all_gathers True \
--data.type text \
--data.dataset_manifest ["s3://tri-ml-datasets/openlm/dcnlp/datasets/tri-hero-run1_cc_v4_resiliparse_rw_v2_bff_minngram13_10shards_all_fasttext_OH_eli5_vs_rw_v2_bigram_200k_train_0.11-starcoder-math_datasets/manifest.jsonl"] \
--data.dataset_modality ["text"] \
--data.dataset_weighting [1.0] \
--data.seq_len 2048 \
--total_train_samples 14_000_000 \
--num_checkpoints 5 \
--hparams.per_gpu_batch_size 8 \
--hparams.global_batch_size 512
```

A few usage notes:
- We pass arguments by prepending the subclass, separated by a period, for example `--model.hidden_dim`. We can also nest multiple layers deep, for example `--model.vit.vit_n_layers`
- For `model` and `data`, we are **required** to set `--model.type` and `--data.type`, which will indicate which specific subclass of `ModelParams` or `DataParams` we will instantiate. For example, `--model.type=transformer_hf` will instantiate `cfg.model` as a `TransformerHFParams` object.
- As seen in the example above, we can use the `--model.load_path` argument to recycle presets that we want to use repeatedly. Command line arguments still take precedence (i.e., if an overlapping argument is supplied in the command line, it will overwrite the value from the preset yaml). This `load_path` can be used for any parameter class, so we can conceivably have `--data.load_path` for datasets we want to recycle, or even something like `--model.unet.load_path`.

#### 1.2 Design Choices
- Arguments are immutable by design, and we recommend developing around this. If really necessary, `object.__setattr__` can be used to modify an immutable argument.
- Shared arguments
    - Sometimes attributes may need to be accessed in multiple param classes. For example, we may want to have both `cfg.experiment.seed` and `cfg.data.seed`. 
    - To prevent the user needing to supply the same argument twice, we pick an "owner" class for the attribute, then for the non-owner class, we list the attribute under the `init_shared_attributes()` function which is automatically called after initialization and populates all shared attributes. 
- We can also have arguments with the same name but are not shared. For instance, `cfg.data.seq_len` and `cfg.model.seq_len` are defined separately. The one in `cfg.data` controls the padding/truncation during dataloading, while the one in `cfg.model` is used for the rotary embedding. 
    - (Note: Now updated to `cfg.model.max_seq_len` instead of just `cfg.model.seq_len`, but point still holds.)

#### 1.3 Dynamic Selection
Consider the following definition of the `VLMParams`
```python
@register_model_params("vlm")
@dataclass(frozen=True)
class VLMParams(ModelParams):
    vit: Union[ViTParams, ViTHFParams] = field(default_factory=ViTParams)
    transformer: Union[TransformerParams, TransformerHFParams] = field(default_factory=TransformerParams)
```

Here, the ViT can either be `ViTParams` or `ViTHFParams`. We can dynamically pick between the two by directly supplying the necessary arguments. For example, indicating `--model.vit.hf_pretrained=vit_base_patch16_siglip_224` will automatically instantiate `cfg.model.vit` as a `ViTHFParams` object, while `--model.vit.vit_hidden_dim=1152` will automatically instantiate `cfg.model.vit` as a `ViTParams` object. No need to indicate `--model.vit.type` in this case.


### 2. Data
Data are stored in shards. Each shard is a tar file. Within each tar file, each sample is distinguished by its unique prefix. 
The structure of the directory is as follows: 

```
dataset_name/
├── manifest.jsonl
├── shard_00000000.tar
│   ├── unique_name_or_hash_1_image1.jpg
│   ├── unique_name_or_hash_1_image2.jpg
│   ├── unique_name_or_hash_1_image3.jpg
│   ├── unique_name_or_hash_1_meta.json
│   ├── unique_name_or_hash_1_caption.json
│   ├── unique_name_or_hash_1_actions.npz
│   ├── unique_name_or_hash_1_otherstuff.json
│   ├── unique_name_or_hash_2_...json
│   ├── unique_name_or_hash_3_...json
├── shard_00000001.tar
│   ├── unique_name_or_hash_100_...json
├── shard_00000002.tar
├── shard_00000003.tar
└── ...
```

In the directory above, the `unique_name_or_hash_1_...` files make up the first sample, the `unique_name_or_hash_2_...` files make up the second sample, and so on. Each tar file can have hundreds or thousands of samples. 

The `manifest.jsonl` provides an overview of the tar files as follows:
```
{"shard": "00000000", "num_sequences": 4518}
{"shard": "00000001", "num_sequences": 4617}
{"shard": "00000002", "num_sequences": 4625}
{"shard": "00000003", "num_sequences": 4701}
```

The dataset can be either local (not recommended) or on S3 (recommended). An example is `s3://tri-ml-datasets/datasets/datacompdr_1b/`.

During dataloading, the code will read `manifest.jsonl`, shuffle the rows, then select the appropriate number of tar files for the given number of training steps. 

#### 2.1 Multiple Datasets
Use the `--data.dataset_manifest` argument to indicate which dataset to use for training. To use more than one dataset, you can supply multiple comma-separated manifests. For example, `--data.dataset_manifest ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl","s3://some-other-dataset/manifest.jsonl"]`. 

Webdatasets also supports different dataset ratios. This is done through the `--data.dataset_weighting` argument. For example, `--data.dataset_weighting [0.4,0.6]`.


### 3. Dataloading Pipeline
We use [webdatasets](https://github.com/webdataset/webdataset) to load the data. Each modality (e.g., image+caption, interleaved, image+actions) has its own pipeline where all the processing steps are defined at a high-level. This involves steps like untarring, shuffling, batching, etc. An example is [lbm2/data/pipelines/image_caption.py](lbm2/data/pipelines/image_caption.py). 

You wil notice that in that file, there is a `self.processor` class that is invoked as a step within the pipeline. This is where all the lower-level processing operations (e.g., normalization, tokenization, padding) are abstracted to. An example is [lbm2/data/processor/stable_diffusion_processor.py](https://github.com/TRI-ML/lbm2/blob/sedrick/diffusion/lbm2/data/processor/stable_diffusion_processor.py).

### 4. Model Saving / Loading
Models checkpoints are saved locally to the path in `cfg.save_path`. If `cfg.remote_sync` is set, then it will save to that path on s3 as well. Save frequency is per checkpoint. The number of checkpoints is determined by the `--num_checkpoints` argument, and the size of a checkpoint is equal to `--total_train_samples` divided by `--num_checkpoints`.

To load checkpoints, (1) Load the params, (2) Create the model (no weights yet), (3) Load the model weights into the model. An example is shown below. More examples can be found in [lbm2/inference](lbm2/inference).
```python
cfg = load_params_from_yaml("s3://(path-here)/config.yaml")
model = create_model(cfg.model)
ckpt = "s3://(path-here)/checkpoints/checkpoint_5.pt"
load_model_checkpoint(model, ckpt, cfg.hparams.seed, cfg.distributed)
```

### 5. Training
At a very high level, training logic is as follows:
```python
model = create_model(cfg)

for ckpt in range(num_checkpoints):
    datastring = get_datastring(cfg)
    dataloader = get_dataloader(datastring)
    train_one_checkpoint(model, dataloader)
    save_checkpoint(model)
```
- `create_model()` -- The [create_model](lbm2/models/__init__.py) function creates the appropriate model based on the `--model.type` model selector and the other `cfg.model` arguments. 
- `datastring` -- This is a string containing a list of the tar files to be loaded for the current checkpoint. A new datastring is created at the beginning of every checkpoint. If using multiple datasets, this is a list of comma-separated strings. A sample datastring is shown below.
```bash
['pipe:aws s3 cp s3://tri-ml-datasets/datasets/datacompdr_1b/{00000037,00000078,00000005,00000099,00000015,00000007,00000063}.tar -']
```
- `train_one_checkpoint()` -- This is defined in [lbm2/train.py](lbm2/train.py). Operations such as model forward, model backward, and loss calculation happen in here.

#### 5.1 Batch Size / Accumulation
- Global batch size is important -- it's a key training hyperparameter.
- Per gpu batch size is important -- it affects training speed.
- Accumulation in itself is less important -- its key role is to make sure the math adds up when your per gpu batch size is not consistent with your global batch size.

Given these, we support setting both the `--hparams.per_gpu_batch_size` (try as high as possible), as well as the `--hparams.global_batch_size`. Accumulation is computed automatically.

### 6. Logging
Logging is done automatically to [wandb](wandb.ai). We use `samples_per_sec_per_gpu` as the main measure of speed. To disable logging, set the `--wandb=False` flag.  

### 7. Linting
We use [ruff](https://github.com/astral-sh/ruff) for formatting and linting. Ruff runs these in separate steps:
```bash
uv run ruff format
uv run ruff check --fix
```

### 8. Tests
Tests are implemented with [pytest](https://docs.pytest.org/en/stable/). To run tests, you can call
```
uv run pytest
```
To run more verbose tests, you can add `-v` for detailed per-test breakdowns and `-s` to display print statement outputs. 

Please add tests for things you implement. To make it clearer on where to add new tests, we organize the `tests` folder in similar structure to the main `lbm2` folder (with subfolders `data`, `models`, etc.) You can run tests in a specific folder by calling something like
```
uv run pytest tests/data
```

#### 8.1 Credentials and Tiny Datasets
API keys and secrets are stored in Github secrets and can be accessed like `${{ secrets.HF_TOKEN }}`. This is already set up properly for Hugging Face, so HF tokenizers and models can now be loaded on tests with no issue.

For AWS S3, this is currently not set up and is generally not recommended (we want tests to be as simple and self-contained as possible, and this adds unnecessary complexity.) For tests that require loading data, we recommend creating tiny WebDataset shards in [tests/shared/tiny_dataset](tests/shared/tiny_dataset). More examples can be found in that folder.