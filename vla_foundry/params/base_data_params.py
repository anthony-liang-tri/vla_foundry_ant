from dataclasses import dataclass, field
from typing import List

import draccus

from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class DataParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)
    dataset_manifest: List[str] = field(default_factory=list)
    dataset_weighting: List[float] = field(default_factory=list)
    dataset_modality: List[str] = field(default_factory=list)
    allow_multiple_epochs: bool = False
    num_workers: int = field(default=1)
    seq_len: int = field(default=2048)
    shuffle: bool = field(default=True)
    shuffle_buffer_size: int = field(default=2000)
    shuffle_initial: int = field(default=500)

    # Shared attributes. Overwritten in init_shared_attributes.
    seed: int = field(default=42)

    def __init__(self):
        raise NotImplementedError("DataParams should not be instantiated directly. Use a subclass with data.type=...")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "dataset_weighting", [float(i) for i in self.dataset_weighting])
        if not self.shuffle:
            object.__setattr__(self, "shuffle_buffer_size", 0)
            object.__setattr__(self, "shuffle_initial", 0)
        if self.type is None:
            object.__setattr__(self, "type", getattr(self.__class__, "_type", None))

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        object.__setattr__(self, "seed", cfg.hparams.seed)
