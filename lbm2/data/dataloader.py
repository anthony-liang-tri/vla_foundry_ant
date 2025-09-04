import copy
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lbm2.data.pipelines import create_wds_pipeline
from lbm2.data.utils import SharedCheckpointCounter
from lbm2.file_utils import load_dataset_manifest


def seed_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs inside a dataloader worker process.

    Args:
        worker_id: The worker id provided by PyTorch's DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataInfo:
    # The `webdataset.WebLoader` (behaves like an iterator of batches).
    dataloader: DataLoader
    # Optional distributed sampler (if used upstream) that can receive
    # checkpoint notifications.
    sampler: DistributedSampler = None
    # Cross-worker/rank observable counter used by some dataset pipelines
    shared_checkpoint_counter: SharedCheckpointCounter = None

    # Optional token id to treat as padding.
    pad_token_id: int = None
    # Optional token id representing image positions (for VLM masking).
    image_token_id: int = None

    def set_checkpoint_num(self, checkpoint_num: int) -> None:
        """
        Propagate the current checkpoint window number to helpers.
        """
        if self.shared_checkpoint_counter is not None:
            self.shared_checkpoint_counter.set_value(checkpoint_num)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_checkpoint_num(checkpoint_num)


def get_wds_dataloader(
    datastrings: Sequence[str],
    num_samples_per_dataset: Sequence[int],
    checkpoint_num: int,
    cfg: object,
) -> DataInfo:
    """
    Build a mixed WebDataset dataloader for a single checkpoint window.
    Args:
        datastrings: Per-dataset WebDataset input strings.
        num_samples_per_dataset: The sample budget to draw from each dataset for this window.
            These are used as mixing probabilities in `wds.mix.RandomMix`.
        checkpoint_num: Current checkpoint window index.
        cfg: Training configuration object.

    Returns:
        DataInfo: A wrapper containing the `WebLoader` and helper objects.
    """
    shared_checkpoint_counter = SharedCheckpointCounter(checkpoint_num=checkpoint_num)
    
    # Per-rank batch size (global batch is split evenly across ranks).
    if cfg.hparams.global_batch_size // cfg.distributed.world_size == 0:
        logging.error(
            f"Global batch size {cfg.hparams.global_batch_size} is smaller than world size "
            f"{cfg.distributed.world_size}, setting it to world size"
        )
    batch_size = max(cfg.hparams.global_batch_size // cfg.distributed.world_size, 1)

    # Build one pipeline per dataset, then mix them by target sample counts.
    datasets = []
    for datastring, modality in zip(datastrings, cfg.data.dataset_modality, strict=False):
        datasets.append(create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, cfg.data))
    dataset = wds.mix.RandomMix(datasets, probs=num_samples_per_dataset, longest=True)

    # Start a generator to have control over reproducibility.
    if cfg.data.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(
            cfg.data.seed + shared_checkpoint_counter.get_value() * cfg.distributed.world_size + cfg.distributed.rank
        )
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    # WebDataset's DataLoader replacement; yields already-batched samples.
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching handled in the pipeline
        shuffle=False,  # mixing is handled by RandomMix
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    # Compute total batches/samples this loader will emit in this window.
    # We want each worker to process the same number of shard-groups.
    if cfg.data.num_workers == 0:
        logging.warning("num_workers is <= 0, setting to 1 per GPU")
    num_workers_per_gpu = max(1, cfg.data.num_workers)
    num_worker_batches = sum(num_samples_per_dataset) // (cfg.hparams.global_batch_size * num_workers_per_gpu)
    if num_worker_batches == 0:
        raise ValueError("The dataloader for has received zero batches.")

    num_batches = num_worker_batches * num_workers_per_gpu
    num_samples = num_batches * cfg.hparams.global_batch_size

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    dataloader = DataInfo(dataloader=dataloader, shared_checkpoint_counter=shared_checkpoint_counter)
    return dataloader


def get_datastring_input(
    num_samples: int,
    curr_shard_idx_per_dataset: int,
    shard_shuffle_seed_per_dataset: int,
    manifest_paths: str,
    dataset_weighting: str,
    allow_multiple_epochs: str,
    num_workers_per_gpu: int,
    world_size: int,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Select shards for the next checkpoint window and build datastrings.

    Given one or more dataset manifests, this function determines how many
    samples to draw from each dataset (according to `dataset_weighting`), then
    selects enough shards so that every worker across all ranks receives the
    same count of shards. It returns WebDataset datastrings suitable for
    `create_wds_pipeline`.

    Args:
        num_samples: Total number of samples to fetch across all
            datasets for this window.
        curr_shard_idx_per_dataset: Current shard cursor per dataset.
        shard_shuffle_seed_per_dataset: Current shuffle seed per dataset.
        manifest_paths: Paths/URIs to per-dataset manifest JSON files.
        dataset_weighting: Optional per-dataset weights; if `None`,
            uses uniform weighting.
        allow_multiple_epochs: Whether to reshuffle and wrap around when
            shards are exhausted.
        num_workers_per_gpu: Number of dataloader workers per rank.
        world_size: Total number of ranks in the distributed job.

    Returns:
        datastrings: Per-dataset WebDataset input strings (local or S3 pipe).
        num_samples_list_per_dataset: Per-dataset total samples scheduled
            for this window (after shard selection).
        next_shard_idx_per_dataset: Updated shard cursors after accounting
            for selected shards.
        next_shard_shuffle_seed_per_dataset: Updated shuffle seeds.
    """
    # Load/reshuffle manifests per dataset with the provided seeds.
    manifests = [
        load_dataset_manifest(path, shard_shuffle_seed=seed)
        for path, seed in zip(manifest_paths, shard_shuffle_seed_per_dataset, strict=False)
    ]

    # Default to uniform weighting if not provided.
    if dataset_weighting is None:
        dataset_weighting = [1 for i in range(len(manifests))]

    if num_samples > 0:
        needed_samples_per_dataset = [
            int(np.ceil(dataset_weighting[i] * num_samples / sum(dataset_weighting))) for i in range(len(manifests))
        ]
    else:
        needed_samples_per_dataset = [-1 for i in range(len(manifests))]
        # Avoid infinite loop when num_samples is -1
        assert not allow_multiple_epochs, "allow_multiple_epochs must be False when num_samples is -1"

    next_shard_idx_per_dataset = copy.deepcopy(curr_shard_idx_per_dataset)
    next_shard_shuffle_seed_per_dataset = copy.deepcopy(shard_shuffle_seed_per_dataset)

    # Build lists of shard names and their sample counts selected for this window.
    shard_list_per_dataset = [[] for i in range(len(manifests))]
    num_samples_list_per_dataset = [[] for i in range(len(manifests))]
    total_num_workers = num_workers_per_gpu * world_size

    # Greedily add shards until we satisfy both:
    # (a) enough samples for the weighting target, and
    # (b) at least one shard per worker (to balance work).
    for i in range(len(manifests)):
        while (
            len(shard_list_per_dataset[i]) < total_num_workers
            or sum(num_samples_list_per_dataset[i]) < needed_samples_per_dataset[i]
            or needed_samples_per_dataset[i] == -1
        ):
            if sum(num_samples_list_per_dataset[i]) >= needed_samples_per_dataset[i]:
                logging.warning(
                    "num_samples requirement satisfied but not all workers have shards. "
                    "Adding data to ensure each worker has a shard."
                )
            try:
                # Take the next shard from the manifest and advance the cursor.
                shard_idx = curr_shard_idx_per_dataset[i]
                shard_list_per_dataset[i].append(manifests[i][shard_idx]["shard"])
                num_samples_list_per_dataset[i].append(manifests[i][shard_idx]["num_sequences"])
                curr_shard_idx_per_dataset[i] += 1
            except IndexError as e:
                if allow_multiple_epochs:
                    # Reshuffle and set index back to 0
                    shard_shuffle_seed_per_dataset[i] += 1
                    manifests[i] = load_dataset_manifest(
                        manifest_paths[i], shard_shuffle_seed=shard_shuffle_seed_per_dataset[i]
                    )
                    curr_shard_idx_per_dataset[i] = 0
                    continue
                else:
                    # If num_samples is -1, we don't need to raise an error,
                    # just break the loop to continue with the next dataset
                    if needed_samples_per_dataset[i] == -1:
                        break
                    logging.error(
                        "Number of shards requested for a single epoch is more than the number of shards available. "
                        "Consider using --allow-multiple-epochs."
                    )
                    raise e

    # Normalize shard lists: ensure each dataset's shard count is divisible by total workers.
    for i in range(len(manifests)):
        # Ensure number of shards is a multiple of number of workers, so each worker has same number of shards.
        idx_div = (
            (len(shard_list_per_dataset[i]) // total_num_workers) * total_num_workers
            if total_num_workers > 0
            else len(shard_list_per_dataset[i])
        )
        shard_list_per_dataset[i] = shard_list_per_dataset[i][:idx_div]
        num_samples_list_per_dataset[i] = num_samples_list_per_dataset[i][:idx_div]

        # Only add used shards. Put back unused shards.
        next_shard_idx_per_dataset[i] += len(shard_list_per_dataset[i])
        next_shard_shuffle_seed_per_dataset[i] += next_shard_idx_per_dataset[i] // len(manifests[i])
        next_shard_idx_per_dataset[i] = next_shard_idx_per_dataset[i] % len(manifests[i])

    # Build WebDataset datastrings per dataset from selected shard names.
    datastrings = []
    for i, manifest_path in enumerate(manifest_paths):
        shard_root_source = "/".join(manifest_path.split("/")[:-1]) + "/"
        curr_datastring = shard_root_source + "{" + ",".join(shard_list_per_dataset[i]) + "}.tar"
        if manifest_path.startswith("s3"):
            # Stream from S3 via pipe so WebDataset can read tar files from stdin.
            curr_datastring = f"pipe:aws s3 cp {curr_datastring} -"
        datastrings.append(curr_datastring)

    # Collapse per-shard sample counts into per-dataset totals.
    total_num_samples_per_dataset = [sum(i) for i in num_samples_list_per_dataset]

    return datastrings, total_num_samples_per_dataset, next_shard_idx_per_dataset, next_shard_shuffle_seed_per_dataset
