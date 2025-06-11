import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lbm2.data.dataloader import get_datastring_input

datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = get_datastring_input(
    num_samples = 350_000,
    curr_shard_idx_per_dataset = [0],
    shard_shuffle_seed_per_dataset = [42],
    manifest_paths = ["s3://tri-ml-datasets/scratch/sedrick.keh/syntheticdatacomop/manifest.jsonl"],
    dataset_weighting = None,
    allow_multiple_epochs = True,
    num_workers_per_gpu = 4,
    world_size = 8,
)