from dataclasses import dataclass, field
from lbm2.params.base_params import BaseParams

@dataclass(frozen=True)
class HyperParams(BaseParams):
    precision: str = field(default="amp_bfloat16")
    global_batch_size: int = field(default=512)
    per_gpu_batch_size: int = field(default=8)

    seed: int = field(default=42)
    lr: float = field(default=1e-4)
    lr_scheduler: str = field(default="cosine")
    warmup: str = field(default='0.1')
    lr_cooldown_end: float = field(default=0.0)
    force_min_lr: float = field(default=0.0)
    optimizer: str = field(default="adamw")
    wd: float = field(default=0.2)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1.0e-8)
    loss_function: str = field(default="cross_entropy")
    z_loss_coefficient: float = field(default=0.0)
    grad_clip_norm: float = field(default=None)
    grad_checkpointing: bool = field(default=False)
    torchcompile: bool = field(default=False)

    # Shared attributes. Overwritten in init_shared_attributes.
    world_size: int = field(default=1)

    @property
    def accum_freq(self):
        combined_batch_size = self.world_size * self.per_gpu_batch_size
        assert self.global_batch_size % combined_batch_size == 0
        return self.global_batch_size // combined_batch_size

    def init_shared_attributes(self, cfg):
        object.__setattr__(self, 'world_size', cfg.distributed.world_size)