import hashlib
import logging
import random
import traceback
from multiprocessing import Value
from typing import Iterable, Sequence

import webdataset as wds
from torch.utils.data import get_worker_info

from vla_foundry.file_utils import load_dataset_manifest, pt_load


class SharedCheckpointCounter:
    """
    A process-safe counter that can be shared across dataloader workers.
    """

    def __init__(self, checkpoint_num: int = 0):
        """
        Args:
            checkpoint_num: Initial value for the counter.
        """
        self.shared_checkpoint_num = Value("i", checkpoint_num)

    def set_value(self, checkpoint_num: int):
        """Set the shared counter to a specific value."""
        self.shared_checkpoint_num.value = checkpoint_num

    def get_value(self):
        """Get the current value of the shared counter."""
        return self.shared_checkpoint_num.value


def log_and_continue(exn: BaseException) -> bool:
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    tb_str = "".join(traceback.format_tb(exn.__traceback__))
    logging.warning(f"Handling webdataset error ({repr(exn)}):\n{tb_str}Ignoring.")
    return True


def pytorch_worker_seed(increment: int = 0) -> int:
    """Get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # Favor using the worker's seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class deterministic_shuffle(wds.PipelineStage):
    """
    Deterministic shuffling stage for WebDataset pipelines.
    If `epoch` is an int, it is incremented locally each time
    `run()` is invoked, which may diverge across workers in multi-process
    settings. To keep workers aligned, pass a `SharedCheckpointCounter`.
    """

    def __init__(
        self,
        bufsize: int = 1000,
        initial: int = 100,
        seed: int = 0,
        epoch: int | SharedCheckpointCounter = -1,
    ) -> None:
        """
        Args:
            bufsize (int): Buffer size for shuffling.
            initial (int): Initial buffer size before yielding.
            seed: Seed for the random number generator.
            epoch: Epoch number.
        """
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src: Iterable) -> Iterable:
        """Yield items from `src` in a deterministic, buffered-shuffled order."""
        if isinstance(self.epoch, SharedCheckpointCounter):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
        # Otherwise, we use the seed + epoch to be deterministic AND the same across all nodes/workers in each epoch
        seed = pytorch_worker_seed(epoch) if self.seed < 0 else self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def load_data_chunks(resume_from_checkpoint: str) -> tuple[list[int], int]:
    """
    Load dataloader cursor state from a checkpoint path.

    Args:
        resume_from_checkpoint: Path to a checkpoint file created by the
        training loop.
    """
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")
    return checkpoint["curr_shard_idx_per_dataset"], checkpoint["samples_seen"]


def epochs_to_samples(manifest_paths: Sequence[str], num_epochs: int) -> int:
    """
    Compute total samples as `num_epochs * sum(num_sequences in manifests)`.

    Args:
        manifest_paths: Sequence of manifest file paths/URIs.
        num_epochs: Number of epochs to iterate over the combined dataset.
    """
    manifests = [load_dataset_manifest(path) for path in manifest_paths]
    num_samples = 0
    for m in manifests:
        num_samples += sum(i["num_sequences"] for i in m)
    return num_samples * num_epochs


def text_to_seed(text: str) -> int:
    """Convert an arbitrary string to a stable 32-bit seed via SHA-256."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32 - 1)
