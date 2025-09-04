import random

import numpy as np
import torch

from datetime import datetime

from lbm2.distributed import broadcast_object


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
    """
    Resolve and return a canonical experiment name, updating ``cfg.name``
    in-place.

    The naming scheme is as follows:
      * If ``cfg.name`` is ``None``, constructs a name of the form
        ``YYYY_MM_DD-HH_MM_SS-model_<type>-lr_<lr>-bsz_<global_batch_size>``.
        In distributed runs, the timestamp is broadcast from the master rank so
        all processes use the exact same name.
      * If ``cfg.name`` is set, it is used as-is.
      * The resulting name is sanitized by replacing forward
        slashes (``/``) with hyphens (``-``).
      * The resolved name is written back to the cfg.

    Args:
        cfg: A TrainExperimentParams config.
    """
    if cfg.name is None:
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
    else:
        name = cfg.name

    # sanitize model name for filesystem / uri use
    name.replace("/", "-")
    object.__setattr__(cfg, "name", name)
    return name
