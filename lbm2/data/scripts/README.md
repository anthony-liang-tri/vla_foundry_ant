# Using Ray
Many scripts use Ray for parallelization.

The general flow is as follows: (1) Create your script that's compatible with Ray. (2) Start a Ray instance on AWS clusters with `ray_cluster_configs.yaml`. This will start a head node with several worker instances. (3) From within the Ray cluster, run your script.

Alternatively, Ray also works on local instances, so Step 3 can be skipped. 

1. Start ray (if not already running)
```bash
ray start --head
```

2. Create the ray cluster
```bash
# Edit ray_cluster_configs.yaml as needed
ray up lbm2/data/scripts/ray_cluster_configs.yaml
```

3. Attach the ray cluster. This will take you "inside" the cluster.
```bash
ray attach lbm2/data/scripts/ray_cluster_configs.yaml
```

4. Run your script inside the cluster.
```bash
# [optional] Start a persistent terminal like tmux
cd lbm2
python (some-script-here)
```

5. When finished, exit the cluster. Then, from your own machine, shut down the ray cluster with `ray down`.
```bash
ray down lbm2/data/scripts/ray_cluster_configs.yaml
```

# Downloading a Hugging Face dataset to S3
```bash
python lbm2/data/scripts/hf_dataset_downloader.py --dataset IPEC-COMMUNITY/droid_lerobot --mode s3 --s3-output-path s3://tri-ml-datasets/hf_datasets/droid_lerobot --local-output-dir /datasets/hf_datasets/droid_lerobot --preserve-structure
```

# Sharding a VLM Hugging Face dataset

We use [img2dataset](https://github.com/rom1504/img2dataset) to handle image downloading and webdataset shard creation. 

This assumes that the HF dataset is already downloaded to S3 (see above section).

```bash
python lbm2/data/scripts/img_shards_img2dataset.py --cluster ray --input_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads/ --output_path s3://tri-ml-datasets/scratch/sedrick.keh/downloads2/ --url_col images --caption_col texts --save_additional_columns metadata
```

# Sharding a LeRobot dataset
This assumes that the HF dataset is already downloaded to S3.

```bash
python lbm2/data/scripts/convert_lerobot_to_tar.py --dataset_path s3://tri-ml-datasets/hf_datasets/oxe_lerobot/droid_lerobot --s3_output_path s3://tri-ml-datasets/lbm2_datasets/droid_lerobot --tmp_dir /tmp --shard_size 2048
```
