import io
import json
import logging
import os
import subprocess
import fsspec
import numpy as np
import torch

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)


def _pt_load_s3_cp(file_path, map_location=None):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Failed to fetch model from s3. stderr: {stderr.decode()}")
    return torch.load(io.BytesIO(stdout), map_location=map_location, weights_only=False)

def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
        return _pt_load_s3_cp(file_path, map_location)
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location, weights_only=False)
    return out

def _json_load_s3_cp(file_path):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch JSON from S3: {stderr.decode().strip()}")
    
    return json.load(io.BytesIO(stdout))

def json_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote json.")
        return _json_load_s3_cp(file_path)
    with open(file_path, 'r') as f:
        out = json.load(f)
    return out


def get_metadata_file(path, shard_shuffle_seed=None):
    of = fsspec.open(path, "rb")
    with of as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]
    if shard_shuffle_seed is not None:
        rng_gen = np.random.default_rng(shard_shuffle_seed)
        rng_gen.shuffle(out)
    return out



def save_checkpoint(
    cfg,
    checkpoint_num, 
    checkpoint_path, 
    model, 
    optimizer, 
    datastrings, 
    curr_shard_idx_per_dataset, 
    samples_seen, 
    global_step, 
    shard_shuffle_seed,
):
    if cfg.distributed.fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    
    checkpoint_dict = {
        "checkpoint_num": checkpoint_num,
        "state_dict": cpu_state if cfg.distributed.fsdp else model.state_dict(),
        "datastrings": datastrings,
        "curr_shard_idx_per_dataset": curr_shard_idx_per_dataset,
        "samples_seen": samples_seen,
        "global_step": global_step,
        "shard_shuffle_seed": shard_shuffle_seed,
    }
    optimizer_dict = {
        "checkpoint_num": checkpoint_num,
        "optimizer": optim_state if cfg.distributed.fsdp else optimizer.state_dict(),
    }
    prefixes = {
        "checkpoint_": checkpoint_dict,
        "optimizer_": optimizer_dict,
    }
    for prefix in prefixes:
        path = os.path.join(checkpoint_path, f"{prefix}{checkpoint_num}.pt")
        print(f"Saving {prefix}{checkpoint_num} in {path}...")
        torch.save(prefixes[prefix], path)


def remote_sync(local_dir, remote_dir):
    logging.info("Starting remote sync.")
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir, remote_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False

    logging.info(f"Successfully synced with S3 bucket")
    return True


def load_model_checkpoint(model, resume_from_checkpoint, seed, distributed_configs):
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")
    if "shard_shuffle_seed" in checkpoint:
        pretrained_seed = checkpoint["shard_shuffle_seed"]
        assert (
            pretrained_seed == seed
        ), f"This checkpoint was trained with a random seed of {pretrained_seed}. Since this seed affects shard shuffling, resuming training must use the same seed."
    else:
        message = "Resuming a checkpoint that does not have a seed saved. This means that the shards were not shuffled, so they will remain unshuffled."
        logging.info(message)
        pretrained_seed = None

    # resuming a train checkpoint w/ epoch and optimizer state
    start_checkpoint_num = checkpoint["checkpoint_num"]
    sd = checkpoint["state_dict"]
    global_step = checkpoint["global_step"]
    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    if "_orig_mod" in next(iter(sd.items()))[0]:
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if distributed_configs.fsdp:
        model.load_state_dict(sd)
    elif distributed_configs.use_distributed:
        model.module.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    logging.info(f"=> resuming checkpoint '{resume_from_checkpoint}' (checkpoint {start_checkpoint_num})")
    return start_checkpoint_num, global_step, pretrained_seed

