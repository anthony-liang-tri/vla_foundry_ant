import random
import re
from collections import Counter, defaultdict
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
    elif cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
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


def summarize_datastrings(datastrings: list[str]) -> str:
    """
    Sometimes datastrings can be very long (e.g., many epochs). This helper function
    summarize them to avoid polluting logging.
    """
    datastring_pattern = re.compile(r"^(?P<prefix>.*?){(?P<items>[^}]*)}(?P<suffix>.*)$")

    # (prefix, suffix) -> Counter[str, count]
    counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    # Data strings without braces
    passthrough = Counter()

    for ds in datastrings:
        match = datastring_pattern.match(ds)
        if match:
            prefix, items, suffix = match.group("prefix", "items", "suffix")
            key = (prefix, suffix)
            vals = [item.strip() for item in items.split(",")]
            for val in vals:
                counter[key][val] += 1
        else:
            passthrough[ds] += 1

    parts = []
    for (prefix, suffix), item_counter in sorted(counter.items()):
        items = sorted(item_counter.keys())
        counts = [item_counter[item] for item in items]
        uniform = all(c == counts[0] for c in counts)
        if uniform:
            part = f"{prefix}{{{', '.join(items)}}}{suffix}"
            if counts[0] > 1:
                part += f" (x{counts[0]})"
        else:
            item_parts = [f"{item} (x{item_counter[item]})" for item in items]
            part = f"{prefix}{{{', '.join(item_parts)}}}{suffix}"
        parts.append(part)

    for ds, count in sorted(passthrough.items()):
        part = ds
        if count > 1:
            part += f" (x{count})"
        parts.append(part)

    return "[" + ",".join(parts) + "]"
