import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager

import fsspec
import numpy as np
import torch
import torch.distributed
import yaml
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
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
    with open(file_path, "r") as f:
        out = json.load(f)
    return out


def _yaml_load_s3_cp(file_path):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch YAML from S3: {stderr.decode().strip()}")

    return yaml.safe_load(io.BytesIO(stdout))


def yaml_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote yaml.")
        return _yaml_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        out = yaml.safe_load(f)
    return out


@contextmanager
def copy_to_temp_file(file_path):
    """
    Copy a file to a temporary file and clean it up when done.
    If the file is on s3, use aws s3 cp to copy it to a temporary file.
    If the file is on the local filesystem, use shutil.copy to copy it to a temporary file.
    """
    extension = os.path.splitext(file_path)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_path = temp_file.name

    try:
        if file_path.startswith("s3"):
            cmd = f"aws s3 cp {file_path} {temp_path}"
            subprocess.run(cmd, shell=True, check=True)
        else:
            shutil.copy(file_path, temp_path)

        yield temp_path
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_dataset_manifest(path, shard_shuffle_seed=None):
    # Check if we're using distributed training
    is_distributed = False
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank = torch.distributed.get_rank()

    # Only rank 0 reads the file
    if not is_distributed or rank == 0:
        max_retry = 3
        for i in range(max_retry):
            try:
                of = fsspec.open(path, "rb")
                with of as f:
                    out = f.read()
                out_split = out.decode("utf-8").split("\n")
                if len(out_split[-1]) == 0:
                    out_split = out_split[:-1]
                out = [json.loads(o) for o in out_split]
                break
            except Exception as e:
                logging.error(f"Error loading dataset manifest: {e}, retry {i}/{max_retry}")
                time.sleep(1)
                if i == max_retry - 1:
                    out = None
                    logging.error(f"Failed to load dataset manifest from {path}")

        # Apply shuffling if needed
        if out is not None and shard_shuffle_seed is not None:
            rng_gen = np.random.default_rng(shard_shuffle_seed)
            rng_gen.shuffle(out)
    else:
        # Non-master processes initialize with None
        out = None

    # Broadcast the result from rank 0 to all other processes
    if is_distributed:
        object_list = [out]
        torch.distributed.broadcast_object_list(object_list, src=0)
        out = object_list[0]

    if out is None:
        raise Exception(f"Failed to load dataset manifest from {path}")

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
    shard_shuffle_seed_per_dataset,
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
        "shard_shuffle_seed_per_dataset": shard_shuffle_seed_per_dataset,
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

    logging.info("Successfully synced with S3 bucket")
    return True


def load_model_checkpoint(model, resume_from_checkpoint, distributed_configs):
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")

    # resuming a train checkpoint w/ epoch and optimizer state
    start_checkpoint_num = checkpoint["checkpoint_num"]
    sd = checkpoint["state_dict"]
    global_step = checkpoint["global_step"]
    shard_shuffle_seed_per_dataset = checkpoint.get("shard_shuffle_seed_per_dataset", None)
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
    return start_checkpoint_num, global_step, shard_shuffle_seed_per_dataset
