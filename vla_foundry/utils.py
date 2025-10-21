import random
from datetime import datetime

import numpy as np
import torch

from vla_foundry.distributed import broadcast_object


def set_random_seed(seed: int = 42, rank: int = 0) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Base seed.
        rank: Rank-specific offset to decorrelate RNG streams across processes.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_experiment_name(cfg):
    if cfg.name is not None:
        name = cfg.name
    elif cfg.model.resume_from_checkpoint is not None:
        # Save in the same directory as the existing checkpoint
        name = cfg.model.resume_from_checkpoint.split("/checkpoints/")[0].split("/")[-1]
    else:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if cfg.distributed.use_distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(cfg, date_str)
        name = "-".join(
            [
                date_str,
                f"model_{cfg.model.type}",
                f"lr_{cfg.hparams.lr}",
                f"bsz_{cfg.hparams.global_batch_size}",
            ]
        )

    # sanitize model name for filesystem / uri use
    name.replace("/", "-")
    object.__setattr__(cfg, "name", name)
    return name
