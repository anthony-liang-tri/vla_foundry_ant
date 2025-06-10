import copy
import logging
import random
import numpy as np
import torch
from typing import List, Optional
from dataclasses import dataclass
import webdataset as wds

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.pipelines import create_wds_pipeline
from data.utils import SharedCheckpointCounter
from file_utils import get_metadata_file


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_checkpoint_counter: SharedCheckpointCounter = None

    pad_token_id: int = None
    image_token_id: int = None

    def set_checkpoint_num(self, checkpoint_num):
        if self.shared_checkpoint_counter is not None:
            self.shared_checkpoint_counter.set_value(checkpoint_num)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_checkpoint_num(checkpoint_num)

    def fill_token_ids_if_available(self, processor, vit_configs):
        if processor is not None:
            from data.processor import get_processor
            processor = get_processor(processor, vit_configs)
            self.pad_token_id = processor.tokenizer.pad_token_id
            self.image_token_id = processor.image_token_id


def get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, cfg):
    shared_checkpoint_counter = SharedCheckpointCounter(checkpoint_num=checkpoint_num)
    batch_size = cfg.data.global_batch_size // cfg.distributed.world_size

    datasets = []
    for datastring, modality in zip(datastrings, cfg.data.dataset_modality):
        datasets.append(create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, cfg))
    dataset = wds.mix.RandomMix(datasets, probs=num_samples_per_dataset, longest=True)

    # Start a generator to have control over reproducibility.
    if cfg.data.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(cfg.data.seed + shared_checkpoint_counter.get_value() * cfg.distributed.world_size + cfg.distributed.rank)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
        
    num_workers_per_gpu = max(1, cfg.data.num_workers)
    num_worker_batches = sum(num_samples_per_dataset) // (cfg.data.global_batch_size * num_workers_per_gpu)
    if num_worker_batches == 0:
        raise ValueError(f"The dataloader for has received zero batches.")

    num_batches = num_worker_batches * num_workers_per_gpu
    num_samples = num_batches * cfg.data.global_batch_size
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    dataloader = DataInfo(dataloader=dataloader, shared_checkpoint_counter=shared_checkpoint_counter)
    dataloader.fill_token_ids_if_available(cfg.data.processor, cfg.vit)
    return dataloader


def get_datastring_input(
    num_samples: int,
    curr_shard_idx_per_dataset: int,
    manifest_paths: str,
    dataset_weighting: str,
    num_workers_per_gpu: int,
    world_size: int,
    shard_shuffle_seed: Optional[int],
):
    manifests = [get_metadata_file(path, shard_shuffle_seed=shard_shuffle_seed) for path in manifest_paths]
    if dataset_weighting is None:
        dataset_weighting = [1 for i in range(len(manifests))]
    
    needed_samples_per_dataset = [int(np.ceil(dataset_weighting[i] * num_samples / sum(dataset_weighting))) for i in range(len(manifests))]
    next_shard_idx_per_dataset = copy.deepcopy(curr_shard_idx_per_dataset)
    shard_list_per_dataset = [[] for i in range(len(manifests))]
    num_samples_list_per_dataset = [[] for i in range(len(manifests))]
    total_num_workers = num_workers_per_gpu * world_size

    for i in range(len(manifests)):
        while len(shard_list_per_dataset[i]) < total_num_workers or sum(num_samples_list_per_dataset[i]) < needed_samples_per_dataset[i]:
            if sum(num_samples_list_per_dataset[i]) >= needed_samples_per_dataset[i]:
                logging.warning("num_samples requirement satisfied but not all workers have shards. Adding data to ensure each worker has a shard.")
            try:
                # Add shards incrementally
                shard_idx = curr_shard_idx_per_dataset[i]
                shard_list_per_dataset[i].append(manifests[i][shard_idx]["shard"])
                num_samples_list_per_dataset[i].append(manifests[i][shard_idx]["num_sequences"])
                curr_shard_idx_per_dataset[i] += 1
            except IndexError as e:
                logging.error("Number of shards requested for a single epoch is more than the number of shards available.")
                raise e

    for i in range(len(manifests)):
        # Ensure number of shards is a multiple of number of workers, so each worker has same number of shards.
        idx_div = (len(shard_list_per_dataset[i]) // total_num_workers) * total_num_workers
        shard_list_per_dataset[i] = shard_list_per_dataset[i][:idx_div]
        num_samples_list_per_dataset[i] = num_samples_list_per_dataset[i][:idx_div]

        # Put back unused shards.
        next_shard_idx_per_dataset[i] += len(shard_list_per_dataset[i])

    datastrings = []
    for i, manifest_path in enumerate(manifest_paths):
        shard_root_source = "/".join(manifest_path.split("/")[:-1]) + "/"
        curr_datastring = shard_root_source + "{" + ",".join(shard_list_per_dataset[i]) + "}.tar"
        if manifest_path.startswith("s3"):
            curr_datastring = f"pipe:aws s3 cp {curr_datastring} -"
        datastrings.append(curr_datastring)

    num_samples_list_per_dataset = [sum(i) for i in num_samples_list_per_dataset]
    return datastrings, num_samples_list_per_dataset, next_shard_idx_per_dataset
