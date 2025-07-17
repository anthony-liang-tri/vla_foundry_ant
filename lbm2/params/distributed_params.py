from dataclasses import dataclass, field

from lbm2.distributed import init_distributed_device
from lbm2.params.base_params import BaseParams


@dataclass(frozen=True)
class DistributedParams(BaseParams):
    dist_url: str = field(default="env://")
    dist_backend: str = field(default="nccl")
    fsdp: bool = field(default=False)
    fsdp_amp: bool = field(default=False)
    fsdp_pure_bf16: bool = field(default=False)
    fsdp_backward_prefetch: bool = field(default=False)
    fsdp_hybrid: bool = field(default=False)
    fsdp_hybrid_o2: bool = field(default=False)
    fsdp_cpu_offload: bool = field(default=False)
    fsdp_use_orig_params: bool = field(default=False)
    fsdp_limit_all_gathers: bool = field(default=False)
    ddp_static_graph: bool = field(default=False)

    # The following should not be initialized by the user.
    # These will be initialized automatically in init_distributed_device()
    use_distributed: bool = field(default=False)
    world_size: int = field(default=1)
    rank: int = field(default=0)
    local_rank: int = field(default=0)
    device: str = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        init_distributed_device(self)
