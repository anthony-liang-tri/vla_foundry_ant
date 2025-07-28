import hashlib
import logging
import random
from multiprocessing import Value

import webdataset as wds
from torch.utils.data import get_worker_info

from lbm2.file_utils import get_metadata_file, pt_load


class SharedCheckpointCounter:
    def __init__(self, checkpoint_num: int = 0):
        self.shared_checkpoint_num = Value("i", checkpoint_num)

    def set_value(self, checkpoint_num):
        self.shared_checkpoint_num.value = checkpoint_num

    def get_value(self):
        return self.shared_checkpoint_num.value


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class deterministic_shuffle(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedCheckpointCounter):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        seed = pytorch_worker_seed(epoch) if self.seed < 0 else self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def load_data_chunks(resume_from_checkpoint):
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")
    return checkpoint["curr_shard_idx_per_dataset"], checkpoint["samples_seen"]


def epochs_to_samples(manifest_paths, num_epochs):
    manifests = [get_metadata_file(path) for path in manifest_paths]
    num_samples = 0
    for m in manifests:
        num_samples += sum(i["num_sequences"] for i in m)
    return num_samples * num_epochs


def text_to_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32 - 1)
