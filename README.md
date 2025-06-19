# lbm2

This was put together from some combination of [MBM](https://github.com/TRI-ML/mbm) and [nanoVLM](https://github.com/huggingface/nanoVLM/tree/main).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. Please follow the uv documentation for installation. Once uv is installed, create a Python 3.10 virtual environment using uv and install the project dependencies with the command below:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quickstart
The main entrypoint is `lbm2/main.py`.

An example command is something like this:
```bash
torchrun --nproc_per_node=8 --nnodes=1 lbm2/main.py \
--model vlm_3b \
--model-type vlm \
--processor google/paligemma-3b-pt-224 \
--fsdp \
--fsdp-use-orig-params \
--fsdp-limit-all-gathers \
--dataset-type webdataset \
--dataset-manifest s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl \
--dataset-modality image_caption \
--total-train-samples 14_000_000 \
--num-checkpoints 5 \
--per-gpu-batch-size 2 \
--global-batch-size 64 \
--vit-img-size 224 \
--vit-hidden-dim 1152 \
--vit-inter-dim 4304 \
--vit-n-heads 16 \
--vit-n-layers 27 \
--vit-patch-size 14 \
--projector-pixel-shuffle-factor 1 \
--seq-len 2048 \
--remote-sync s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b \
--disable-wandb
```

See `./examples` for more examples.

### Running on SageMaker
To install SageMaker,
```
pip install install/sagemaker-2.240.1.dev0.tar
```

To launch something on SageMaker, the launch file is [sagemaker/launch_training.py](sagemaker/launch_training.py). Pass arguments in a similar way as you would for a local run. 

```bash
python sagemaker/launch_training.py --user your.user.name --instance-count 4 --instance-type p4de --insert-your-arguments-here
```

## Repo Structure and Implementation
The sections below highlight several key design choices and functionalities of the repo.

### 1. Param/Argument Structure
Params are defined in the [lbm2/params](lbm2/params) folder. We use nested parameters with pure argparse. There is a high level `cfg` dataclass object in [lbm2/main.py](lbm2/main.py). This dataclass has attributes which are dataclasses themselves (e.g., `cfg.model`, `cfg.experiment`). Within `cfg.model` are the actual arguments like `cfg.model.hidden_dim`.

All arguments are supplied using the standard argparse argument passing (i.e., `--argument-name`). The `parser.add_argument` for any given argument is defined only once, but the argument can be shared among multiple dataclasses by defining it as a class attribute. For example, we can have both `cfg.model.vocab_size` and `cfg.data.vocab_size`. To add a new argument, first select the appropriate dataclass, then (1) add the `parser.add_argument` lines, and (2) add the argument as an attribute to the class. 


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
Use the `--dataset-manifest` argument to indicate which dataset to use for training. To use more than one dataset, you can supply multiple comma-separated manifests. For example, `--dataset-manifest s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl,s3://some-other-dataset/manifest.jsonl`. 

Webdatasets also supports different dataset ratios. This is done through the `--dataset-weighting` argument. For example, `--dataset-weighting 0.4,0.6`.


### 3. Dataloading Pipeline
We use [webdatasets](https://github.com/webdataset/webdataset) to load the data. Each modality (e.g., image+caption, interleaved, image+actions) has its own pipeline where all the processing steps are defined at a high-level. This involves steps like untarring, shuffling, batching, etc. An example is [lbm2/data/pipelines/image_caption.py](lbm2/data/pipelines/image_caption.py). 

You wil notice that in that file, there is a `self.processor` class that is invoked as a step within the pipeline. This is where all the lower-level processing operations (e.g., normalization, tokenization, padding) are abstracted to. An example is [lbm2/data/processor/stable_diffusion_processor.py](https://github.com/TRI-ML/lbm2/blob/sedrick/diffusion/lbm2/data/processor/stable_diffusion_processor.py).

### 4. Model Saving / Loading
Models checkpoints are saved locally to the path in `cfg.experiment.save_path`. If `cfg.experiment.remote_sync` is set, then it will save to that path on s3 as well. Save frequency is per checkpoint. The number of checkpoints is determined by the `--num-checkpoints` argument, and the size of a checkpoint is equal to `--total-train-samples` divided by `--num-checkpoints`.

To load checkpoints, (1) Load the params, (2) Create the model (no weights yet), (3) Load the model weights into the model. An example is shown below. More examples can be found in [lbm2/inference](lbm2/inference).
```python
cfg = load_params_from_json("s3://(path-here)/config.json")
model = create_model(cfg.model)
ckpt = "s3://(path-here)/checkpoints/checkpoint_5.pt"
load_model_checkpoint(model, ckpt, cfg.experiment.seed, cfg.distributed)
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
- `create_model()` -- The [create_model](lbm2/models/__init__.py) function creates the appropriate model based on the `--model-type` argument and the other `cfg.model` arguments. 
- `datastring` -- This is a string containing a list of the tar files to be loaded for the current checkpoint. A new datastring is created at the beginning of every checkpoint. If using multiple datasets, this is a list of comma-separated strings. A sample datastring is shown below.
```bash
['pipe:aws s3 cp s3://tri-ml-datasets/datasets/datacompdr_1b/{00000037,00000078,00000005,00000099,00000015,00000007,00000063}.tar -']
```
- `train_one_checkpoint()` -- This is defined in [lbm2/train.py](lbm2/train.py). Operations such as model forward, model backward, and loss calculation happen in here.

#### 5.1 Batch Size / Accumulation
- Global batch size is important -- it's a key training hyperparameter.
- Per gpu batch size is important -- it affects training speed.
- Accumulation in itself is less important -- its key role is to make sure the math adds up when your per gpu batch size is not consistent with your global batch size.

Given these, we support setting both the `--per-gpu-batch-size` (try as high as possible), as well as the `--global-batch-size`. Accumulation is computed automatically.

### 6. Logging
Logging is done automatically to [wandb](wandb.ai). We use `samples_per_sec_per_gpu` as the main measure of speed. To disable logging, set the `--disable-wandb` flag.  

### 7. Tests
(todo)
