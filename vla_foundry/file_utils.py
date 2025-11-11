import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from typing import Tuple

import boto3
import fsspec
import numpy as np
import yaml

try:
    import torch
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:
    # Avoid needing to download torch for Ray clusters
    logging.info("Skipping torch imports in file_utils.py")
    pass


def _pt_load_s3_cp(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        return torch.load(f, map_location=map_location, weights_only=False)


def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
        return _pt_load_s3_cp(file_path, map_location)
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location, weights_only=False)
    return out


def _json_load_s3_cp(file_path):
    of = fsspec.open(file_path, "rb")
    with of as f:
        return json.load(f)


def json_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote json.")
        return _json_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        out = json.load(f)
    return out


def _jsonl_load_s3_cp(file_path):
    of = fsspec.open(file_path, "rb")
    with of as f:
        content = f.read().decode("utf-8")
        return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def jsonl_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote jsonl.")
        return _jsonl_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries


def _yaml_load_s3_cp(file_path):
    of = fsspec.open(file_path, "rb")
    with of as f:
        return yaml.safe_load(f)


def yaml_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote yaml.")
        return _yaml_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        out = yaml.safe_load(f)
    return out


def _list_directory_s3_ls(dir_path):
    fs, path = fsspec.core.url_to_fs(dir_path)
    try:
        entries = fs.ls(path, detail=False)
        # Extract just the basename (filename or dirname) from full paths
        items = []
        for entry in entries:
            # Remove the parent path to get just the name
            basename = entry.split("/")[-1]
            if basename:  # Skip empty strings
                items.append(basename)
        return items
    except Exception as e:
        raise RuntimeError(f"Failed to list S3 directory: {str(e)}") from e


def list_directory(dir_path):
    if dir_path.startswith("s3"):
        return _list_directory_s3_ls(dir_path)
    return os.listdir(dir_path)


def check_directory_has_files_with_prefix(dir_path, prefix):
    """Check if directory exists and contains files with the given prefix.

    Args:
        dir_path: Path to directory (can be S3 or local)
        prefix: Prefix to check for in filenames

    Returns:
        List of files matching the prefix, or empty list if directory doesn't exist
    """
    try:
        files = list_directory(dir_path)
        return [f for f in files if f.startswith(prefix)]
    except (RuntimeError, FileNotFoundError, OSError):
        # Directory doesn't exist or can't be accessed
        return []


def check_directory_has_files_with_substring(dir_path, substring):
    """Check if directory exists and contains files with the given substring."""
    try:
        files = list_directory(dir_path)
        return [f for f in files if substring in f]
    except (RuntimeError, FileNotFoundError, OSError):
        return []


def _is_dir_s3_ls(dir_path):
    """Check if an S3 path is a directory using fsspec."""
    fs, path = fsspec.core.url_to_fs(dir_path)
    try:
        return fs.isdir(path)
    except Exception:
        return False


def is_dir(path):
    if path.startswith("s3"):
        return _is_dir_s3_ls(path)
    return os.path.isdir(path)


def list_directory_recursive(dir_path):
    """Get all S3 objects under a prefix efficiently using pagination."""
    assert dir_path.startswith("s3"), "Only S3 paths are supported for now"
    s3_client = boto3.client("s3")
    bucket, prefix = parse_s3_path(dir_path)
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = set()

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Store relative path from prefix
                    key = obj["Key"]
                    if key.startswith(prefix):
                        relative_key = key[len(prefix) :]
                        objects.add(relative_key.lstrip("/"))
    except Exception as e:
        print(f"Error listing directory {dir_path}: {e}")
        pass

    return objects


def _file_exists_s3_ls(file_path):
    """Check if an S3 file exists using fsspec."""
    fs, path = fsspec.core.url_to_fs(file_path)
    try:
        return fs.exists(path)
    except Exception:
        return False


def file_exists(path):
    if path.startswith("s3"):
        return _file_exists_s3_ls(path)
    return os.path.exists(path)


def get_lowdim_past_future_timesteps(statistics_path: str) -> Tuple[int, int]:
    try:
        metadata_path = os.path.join(os.path.dirname(statistics_path), "processing_metadata.json")
        metadata = json_load(metadata_path)
        past_lowdim = metadata["command_line"]["arguments"]["past_lowdim_steps"]
        future_lowdim = metadata["command_line"]["arguments"]["future_lowdim_steps"]
        return int(past_lowdim), int(future_lowdim)
    except Exception as e:
        logging.warning(f"Failed to get lowdim past and future timesteps from {statistics_path}: {e}")
        return None, None


def parse_s3_path(s3_path: str):
    assert s3_path.startswith("s3://")
    parts = s3_path.removeprefix("s3://").split("/", 1)
    bucket = parts[0]
    directory = parts[1].removesuffix("/") if len(parts) > 1 else ""
    return bucket, directory


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
            of = fsspec.open(file_path, "rb")
            with of as f_in, open(temp_path, "wb") as f_out:
                f_out.write(f_in.read())
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
    max_checkpoint_limit,
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

    # Get model state dict and ensure consistent naming
    if cfg.distributed.fsdp:
        model_state_dict = cpu_state
    else:
        # Use unwrapped model state dict to avoid module prefix
        unwrapped_model = get_unwrapped_model(model)
        model_state_dict = unwrapped_model.state_dict()

    checkpoint_dict = {
        "checkpoint_num": checkpoint_num,
        "state_dict": model_state_dict,
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

    # Clean up old checkpoints if max_checkpoints is specified
    if max_checkpoint_limit is not None and checkpoint_num >= max_checkpoint_limit:
        oldest_checkpoint = checkpoint_num - max_checkpoint_limit
        for prefix in prefixes:
            old_path = os.path.join(checkpoint_path, f"{prefix}{oldest_checkpoint}.pt")
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Removed old checkpoint: {prefix}{oldest_checkpoint}.pt")


def get_unwrapped_model(model):
    """Get the unwrapped model from DDP wrapper if present, otherwise return the model itself."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


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


def load_model_checkpoint(model, resume_from_checkpoint):
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")

    # resuming a train checkpoint w/ epoch and optimizer state
    start_checkpoint_num = checkpoint["checkpoint_num"]
    sd = checkpoint["state_dict"]
    global_step = checkpoint["global_step"]
    shard_shuffle_seed_per_dataset = checkpoint.get("shard_shuffle_seed_per_dataset", None)
    if "_orig_mod" in next(iter(sd.items()))[0]:
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
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
        model = get_unwrapped_model(model)
        model.load_state_dict(sd)
    logging.info(f"=> resuming checkpoint '{resume_from_checkpoint}' (checkpoint {start_checkpoint_num})")
    return start_checkpoint_num, global_step, shard_shuffle_seed_per_dataset


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str):
    is_s3 = path.startswith("s3")
    fs, root_path = fsspec.core.url_to_fs(path)
    if not root_path.rstrip("/").endswith("checkpoints"):
        root_path = os.path.join(root_path, "checkpoints")
    checkpoints = fs.glob(os.path.join(root_path, "checkpoint_*.pt"))
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return f"s3://{checkpoints[-1]}" if is_s3 else checkpoints[-1]

    return None


def collect_processing_metadata(dataset_manifest_paths, experiment_path):
    """
    Collect processing_metadata.json files from all data sources and group them.

    Args:
        dataset_manifest_paths: List of paths to dataset manifest files
        experiment_path: Path to the experiment directory

    Returns:
        dict: Grouped processing metadata from all sources
    """

    all_metadata = []
    dataset_sources = []

    for manifest_path in dataset_manifest_paths:
        # Get the directory containing the manifest
        manifest_dir = os.path.dirname(manifest_path)

        # Look for processing_metadata.json in the same directory as the manifest
        processing_metadata_path = os.path.join(manifest_dir, "processing_metadata.json")

        try:
            if processing_metadata_path.startswith("s3://"):
                # Handle S3 paths
                metadata = json_load(processing_metadata_path)
            else:
                # Handle local paths
                if os.path.exists(processing_metadata_path):
                    metadata = json_load(processing_metadata_path)
                else:
                    logging.warning(f"Processing metadata not found at {processing_metadata_path}")
                    continue

            all_metadata.append(metadata)
            dataset_sources.append(manifest_path)
            logging.info(f"Loaded processing metadata from {processing_metadata_path}")

        except Exception as e:
            logging.warning(f"Failed to load processing metadata from {processing_metadata_path}: {e}")
            continue

    if not all_metadata:
        logging.warning("No processing metadata found for any data source")
        return None

    # Group metadata from multiple sources
    grouped_metadata = group_processing_metadata(all_metadata, dataset_sources)

    return grouped_metadata


def group_processing_metadata(metadata_list, dataset_sources):
    """
    Group processing metadata from multiple sources into a single structure.

    Args:
        metadata_list: List of metadata dictionaries from each source
        dataset_sources: List of dataset manifest paths corresponding to each metadata

    Returns:
        dict: Grouped metadata with lists for each field and global sources info
    """
    if len(metadata_list) == 1:
        # Single source - just add source info for compatibility
        metadata = metadata_list[0].copy()
        metadata["dataset_sources"] = dataset_sources
        metadata["num_sources"] = 1
        return metadata

    def _group_field_values(values):
        """Recursively group a list of values coming from different sources."""

        if all(value is None or isinstance(value, dict) for value in values):
            grouped_dict = {}
            all_keys = set()

            for value in values:
                if isinstance(value, dict):
                    all_keys.update(value.keys())

            for key in sorted(all_keys):
                grouped_sub_values = []
                for value in values:
                    if isinstance(value, dict) and key in value:
                        grouped_sub_values.append(value[key])
                    else:
                        grouped_sub_values.append(None)
                grouped_dict[key] = _group_field_values(grouped_sub_values)

            return grouped_dict

        return [
            {
                "source_index": idx,
                "source_manifest": dataset_sources[idx],
                "value": value if value is not None else None,
            }
            for idx, value in enumerate(values)
        ]

    grouped = {
        "grouping_metadata_version": "1.0",
        "dataset_sources": dataset_sources,
        "num_sources": len(metadata_list),
    }

    all_fields = set()
    for metadata in metadata_list:
        all_fields.update(metadata.keys())

    for field in sorted(all_fields):
        values = [metadata.get(field) for metadata in metadata_list]
        grouped[field] = _group_field_values(values)

    total_samples = sum(metadata.get("processing", {}).get("total_samples_created", 0) for metadata in metadata_list)

    grouped["summary"] = {
        "total_sources": len(metadata_list),
        "total_samples_across_sources": total_samples,
        "source_manifests": dataset_sources,
    }

    return grouped


def collect_preprocessing_configs(dataset_manifest_paths):
    """
    Return a list of preprocessing configs from all data sources (if exists)
    """

    def collect_single_source_preprocessing_config(dataset_manifest_path):
        """
        Collect the preprocessing config from a single data source
        """
        preprocessing_config_path = os.path.join(os.path.dirname(dataset_manifest_path), "preprocessing_config.yaml")
        if file_exists(preprocessing_config_path):
            return yaml_load(preprocessing_config_path)
        return None

    if len(dataset_manifest_paths) == 1:
        return collect_single_source_preprocessing_config(dataset_manifest_paths[0])
    else:
        all_preprocessing_configs = {}
        for idx, dataset_manifest_path in enumerate(dataset_manifest_paths):
            preprocessing_config = collect_single_source_preprocessing_config(dataset_manifest_path)
            all_preprocessing_configs[idx] = preprocessing_config
        return all_preprocessing_configs
