# This is from open_clip.
import os
import logging
import torch
import torch.distributed as dist


def is_global_master(cfg):
    return cfg.distributed.rank == 0


def is_local_master(cfg):
    return cfg.distributed.local_rank == 0


def is_master(cfg, local=False):
    return is_local_master(cfg) if local else is_global_master(cfg)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def init_distributed_device(distributed_configs):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    object.__setattr__(distributed_configs, 'use_distributed', False)       # bypass Frozen=True
    object.__setattr__(distributed_configs, 'world_size', 1)
    object.__setattr__(distributed_configs, 'rank', 0)
    object.__setattr__(distributed_configs, 'local_rank', 0)
    if is_using_distributed():
        # DDP via torchrun, torch.distributed.launch
        # Note that this currently assumes that the world size is all gpus in a node.
        local_rank, _, _ = world_info_from_env()
        object.__setattr__(distributed_configs, 'local_rank', local_rank)
        torch.distributed.init_process_group(
            backend=distributed_configs.dist_backend, 
            init_method=distributed_configs.dist_url
        )
        object.__setattr__(distributed_configs, 'world_size', torch.distributed.get_world_size())
        object.__setattr__(distributed_configs, 'rank', torch.distributed.get_rank())
        object.__setattr__(distributed_configs, 'use_distributed', True)

    if torch.cuda.is_available():
        if distributed_configs.use_distributed:
            device = "cuda:%d" % distributed_configs.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    object.__setattr__(distributed_configs, 'device', device)
    device = torch.device(device)
    return device


def broadcast_object(cfg, obj, src=0):
    if cfg.distributed.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(cfg, obj, dst=0):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(cfg.distributed.world_size)]
    dist.all_gather_object(objects, obj)
    return objects


# Add fsdp here:
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lbm2.models import get_model_block
import numpy as np
import random

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def get_model_precision(cfg):
    """
    Determine the appropriate model precision based on distributed configuration.
    Returns the dtype that should be used for model parameters.
    """
    if cfg.distributed.fsdp:
        # FSDP handles precision through MixedPrecision policy
        if cfg.distributed.fsdp_amp or cfg.distributed.fsdp_pure_bf16:
            return torch.bfloat16
        else:
            return torch.float32  # Default FSDP precision
    else:
        # For DDP and single GPU, use bfloat16 by default to match FSDP behavior
        return torch.bfloat16

def wrap_fsdp_ddp(model, device, cfg):
    if cfg.distributed.fsdp:
        # from https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=get_model_block(cfg.model.type, cfg.model),
        )
        # tries to follow gopher...
        mp_policy = None
        if cfg.distributed.fsdp_amp:
            print("=> using bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        elif cfg.distributed.fsdp_pure_bf16:
            print("=> using pure bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        if cfg.distributed.rank == 0:
            print(f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Before FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

        fsdp_kwargs = {}
        assert not (
            cfg.distributed.fsdp_hybrid and cfg.distributed.fsdp_hybrid_o2
        ), "Only --fsdp-hybrid or --fsdp-hybrid-o2 should be set."
        if cfg.distributed.fsdp_backward_prefetch:
            fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
        if cfg.distributed.fsdp_hybrid:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
        if cfg.distributed.fsdp_hybrid_o2:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy._HYBRID_SHARD_ZERO2
        print("=> FSDP kwargs: ", fsdp_kwargs)

        # Initialize FSDP. Use the same seed across workers to ensure reset_parameters is the same across workers.
        random_seed(cfg.hparams.seed, rank=0)
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrapper_policy,
            device_id=device,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=cfg.distributed.fsdp_cpu_offload),
            use_orig_params=cfg.distributed.fsdp_use_orig_params,
            limit_all_gathers=cfg.distributed.fsdp_limit_all_gathers,
            **fsdp_kwargs,
        )

        print(f"After FSDP parameter num: {sum(p.numel() for p in model.parameters()):,} on rank {cfg.distributed.rank}")
        print(f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {cfg.distributed.rank}")
    else:
        ddp_args = {}
        if cfg.distributed.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    return model