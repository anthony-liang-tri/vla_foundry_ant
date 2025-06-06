import random
import numpy as np
import torch
from dataclasses import dataclass
import webdataset as wds

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.pipelines import create_wds_pipeline
from data.utils import SharedCheckpointCounter


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_checkpoint_counter: SharedCheckpointCounter = None

    def set_checkpoint_num(self, checkpoint_num):
        if self.shared_checkpoint_counter is not None:
            self.shared_checkpoint_counter.set_value(checkpoint_num)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_checkpoint_num(checkpoint_num)


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

    return DataInfo(dataloader=dataloader, shared_checkpoint_counter=shared_checkpoint_counter)


def get_synthetic_dataset(data_configs, distributed_configs, checkpoint_num, tokenizer, floor):
    from data.datasets import SyntheticDataset
    dataset = SyntheticDataset(seq_len=data_configs.seq_len, vocab_size=data_configs.vocab_size, dataset_size=data_configs.total_train_samples)
    sampler = DistributedSampler(dataset) if distributed_configs.use_distributed else None
    shuffle = sampler is None

    per_gpu_batch_size = data_configs.global_batch_size // distributed_configs.world_size
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=shuffle,
        num_workers=data_configs.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
