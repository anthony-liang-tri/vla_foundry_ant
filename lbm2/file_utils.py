import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import fsspec
import numpy as np
import torch
import yaml
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor


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


def get_metadata_file(path, shard_shuffle_seed=None):
    of = fsspec.open(path, "rb")
    with of as f:
        out = f.read()
    out_split = out.decode("utf-8").split("\n")
    if len(out_split[-1]) == 0:
        out_split = out_split[:-1]
    out = [json.loads(o) for o in out_split]
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
    shard_shuffle_seed_per_dataset,
):
    if cfg.distributed.fsdp:
        # FSDP get model state dict (load all params to CPU)
        cpu_state = {}
        for param_name, sharded_param in model.state_dict().items():
            full_param = sharded_param.full_tensor() if isinstance(sharded_param, DTensor) else sharded_param
            if torch.distributed.get_rank() == 0:
                cpu_state[param_name] = full_param.cpu()
            else:
                del full_param

        # FSDP get optimizer state dict
        full_state = {}
        for group_id, sharded_group in optimizer.state_dict()["state"].items():
            group_state = {}
            for param_name, sharded_param in sharded_group.items():
                full_tensor = sharded_param.full_tensor() if isinstance(sharded_param, DTensor) else sharded_param
                if torch.distributed.get_rank() == 0:
                    group_state[param_name] = full_tensor.cpu()
                else:
                    del full_tensor
            if torch.distributed.get_rank() == 0:
                full_state[group_id] = group_state
            else:
                del group_state
        optim_state = {
            "param_groups": optimizer.state_dict()["param_groups"],
            "state": full_state,
        }

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
    if "_orig_mod" in next(iter(sd.items()))[0]:
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if distributed_configs.fsdp:
        if isinstance(model, FSDPModule):
            sharded_sd = {}
            model_sd = model.state_dict()
            for param_name, full_tensor in sd.items():
                sharded_meta_param = model_sd.get(param_name)
                if isinstance(sharded_meta_param, DTensor):
                    # shard weights from cpu to their target device
                    sharded_tensor = distribute_tensor(
                        full_tensor,
                        sharded_meta_param.device_mesh,
                        sharded_meta_param.placements,
                    )
                    sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
                else:
                    # FSDP2 doesn't shard buffers.
                    assert torch.allclose(
                        full_tensor.to(sharded_meta_param.device), sharded_meta_param, rtol=1e-5, atol=1e-8
                    )
                    sharded_sd[param_name] = sharded_meta_param
            model.load_state_dict(sharded_sd, assign=True)
        else:
            # Inference
            model.load_state_dict(sd)
    elif distributed_configs.use_distributed:
        model.module.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    logging.info(f"=> resuming checkpoint '{resume_from_checkpoint}' (checkpoint {start_checkpoint_num})")
    return start_checkpoint_num, global_step, shard_shuffle_seed_per_dataset
