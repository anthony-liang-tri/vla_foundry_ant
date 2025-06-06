import torch
from dataclasses import dataclass, field, fields


def add_distributed_params(parser):
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-amp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-pure-bf16",
        default=False,
        action="store_true",
        help="Use pure bf16 FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-backward-prefetch",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-hybrid",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-hybrid-o2",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        default=False,
        action="store_true",
        help="CPU offloading for FSDP and checkpoint saving. This does not work with gradient accumulation.",
    )
    parser.add_argument(
        "--fsdp-use-orig-params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. This does not work for OPT models. Enables param_groups for weight_decay.",
    )
    parser.add_argument(
        "--fsdp-limit-all-gathers",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )


@dataclass(frozen=True)
class DistributedParams:
    dist_url: str
    dist_backend: str
    fsdp: bool
    fsdp_amp: bool
    fsdp_pure_bf16: bool
    fsdp_backward_prefetch: bool
    fsdp_hybrid: bool
    fsdp_hybrid_o2: bool
    fsdp_cpu_offload: bool
    fsdp_use_orig_params: bool
    fsdp_limit_all_gathers: bool
    ddp_static_graph: bool

    # The following should not be initialized by the user. 
    # These will be initialized automatically in init_distributed_device()
    use_distributed: bool = field(default=False)
    world_size: int = field(default=1)
    rank: int = field(default=0)
    local_rank: int = field(default=0)
    device: torch.device = field(default=None)

    @classmethod
    def from_args(cls, args):
        init_kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(cls)
            if hasattr(args, f.name)
        }
        return cls(**init_kwargs)
