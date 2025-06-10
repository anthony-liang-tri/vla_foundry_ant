import copy
import io
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from itertools import cycle, islice

import fsspec
import numpy as np
import torch

from typing import List, Optional
from tqdm import tqdm

from distributed import is_master
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
)



# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)


def _pt_load_s3_cp(file_path, map_location=None):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Failed to fetch model from s3. stderr: {stderr.decode()}")
    return torch.load(io.BytesIO(stdout), map_location=map_location)

def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
        return _pt_load_s3_cp(file_path, map_location)
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
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


def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True


def get_metadata_file(path, shard_shuffle_seed=None):
    of = fsspec.open(path, "rb")
    with of as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]
    if shard_shuffle_seed is not None:
        rng_gen = np.random.default_rng(shard_shuffle_seed)
        rng_gen.shuffle(out)
    return out


def get_shards_for_chunk(num_samples, chunk, path, shard_shuffle_seed):
    """Function to get a chunk of shards to train on.

    Chunks are groups of shards with samples roughly equal to the number of samples
    that will be seen during training. This function uses the dataset manifest
    to split the shards into chunks, and assign shards to each chunk.
    """
    metadata = get_metadata_file(path, shard_shuffle_seed=shard_shuffle_seed)
    shard_list = []
    curr_shard_list = []
    chunk_count_list = []
    curr_chunk_count = 0
    for m in metadata:
        try:
            curr_chunk_count += m["num_sequences"]
        except KeyError:
            curr_chunk_count += m["num_chunks"]

        curr_shard_list.append(m["shard"])
        if curr_chunk_count >= num_samples:
            shard_list.append(curr_shard_list)
            chunk_count_list.append(curr_chunk_count)
            curr_shard_list = []
            curr_chunk_count = 0

    # Append remaining shards
    if len(curr_shard_list) > 0:
        shard_list.append(curr_shard_list)
        chunk_count_list.append(curr_chunk_count)

    return (
        shard_list[chunk % len(shard_list)],
        chunk_count_list[chunk % len(chunk_count_list)],
    )


def enough_shards(shard_lists: List[List[str]], min_shards_needed: int):
    for sl in shard_lists:
        if len(sl) < min_shards_needed:
            return False
    return True


def enough_samples(num_samples_per_source: List[List[int]], needed_samples_per_source: List[int]):
    for i, number_per_shard in enumerate(num_samples_per_source):
        if sum(number_per_shard) < needed_samples_per_source[i]:
            return False
    return True


def source_exhausted(paths, shard_list_per_source):
    for i, source in enumerate(paths):
        data = get_metadata_file(source)
        if len(data) < len(shard_list_per_source[i]):
            return True
    return False


def count_small_shards(path, ratio=0.9):
    """Count the number of shards with significantly fewer sequences than the largest shard.

    Small shards are defined as those that have size less than a ratio (default 90%) of the size of the largest shard.
    """
    shard_sizes = []
    data = get_metadata_file(path)
    for item in data:
        try:
            shard_sizes.append(item["num_sequences"])
        except KeyError:
            shard_sizes.append(item["num_chunks"])

    shard_sizes = np.array(shard_sizes)

    return np.sum(shard_sizes < ratio * max(shard_sizes))



def get_datastring_input(
    num_samples: int,
    curr_shard_idx_per_dataset: int,
    manifest_paths: str,
    dataset_weighting: str,
    num_workers_per_gpu: int,
    world_size: int,
    shard_shuffle_seed: Optional[int],
):
    manifests = [get_metadata_file(path, shard_shuffle_seed=shard_shuffle_seed) for path in manifest_paths]
    if dataset_weighting is None:
        dataset_weighting = [1 for i in range(len(manifests))]
    
    needed_samples_per_dataset = [int(np.ceil(dataset_weighting[i] * num_samples / sum(dataset_weighting))) for i in range(len(manifests))]
    next_shard_idx_per_dataset = copy.deepcopy(curr_shard_idx_per_dataset)
    shard_list_per_dataset = [[] for i in range(len(manifests))]
    num_samples_list_per_dataset = [[] for i in range(len(manifests))]
    total_num_workers = num_workers_per_gpu * world_size

    for i in range(len(manifests)):
        while len(shard_list_per_dataset[i]) < total_num_workers or sum(num_samples_list_per_dataset[i]) < needed_samples_per_dataset[i]:
            if sum(num_samples_list_per_dataset[i]) >= needed_samples_per_dataset[i]:
                logging.warning("num_samples requirement satisfied but not all workers have shards. Adding data to ensure each worker has a shard.")
            try:
                # Add shards incrementally
                shard_idx = curr_shard_idx_per_dataset[i]
                shard_list_per_dataset[i].append(manifests[i][shard_idx]["shard"])
                num_samples_list_per_dataset[i].append(manifests[i][shard_idx]["num_sequences"])
                curr_shard_idx_per_dataset[i] += 1
            except IndexError as e:
                logging.error("Number of shards requested for a single epoch is more than the number of shards available.")
                raise e

    for i in range(len(manifests)):
        # Ensure number of shards is a multiple of number of workers, so each worker has same number of shards.
        idx_div = (len(shard_list_per_dataset[i]) // total_num_workers) * total_num_workers
        shard_list_per_dataset[i] = shard_list_per_dataset[i][:idx_div]
        num_samples_list_per_dataset[i] = num_samples_list_per_dataset[i][:idx_div]

        # Put back unused shards.
        next_shard_idx_per_dataset[i] += len(shard_list_per_dataset[i])

    datastrings = []
    for i, manifest_path in enumerate(manifest_paths):
        shard_root_source = "/".join(manifest_path.split("/")[:-1]) + "/"
        curr_datastring = shard_root_source + "{" + ",".join(shard_list_per_dataset[i]) + "}.tar"
        if manifest_path.startswith("s3"):
            curr_datastring = f"pipe:aws s3 cp {curr_datastring} -"
        datastrings.append(curr_datastring)

    num_samples_list_per_dataset = [sum(i) for i in num_samples_list_per_dataset]
    return datastrings, num_samples_list_per_dataset, next_shard_idx_per_dataset



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


def remote_sync_with_expon_backoff(sync_every, local_dir, remote_dir, max_retries=6):
    for i in range(max_retries):
        success = remote_sync(local_dir, remote_dir)
        if success:
            return True
        time.sleep(sync_every * 2**i)
    return False

def keep_running_remote_sync(sync_every, local_dir, remote_dir):
    while True:
        remote_sync_with_expon_backoff(sync_every, local_dir, remote_dir)

def keep_running_remote_sync_process(sync_every, local_dir, remote_dir):
    p = multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, local_dir, remote_dir),
    )
    return p

def start_remote_sync_process(experiemnt_configs, experiment_path, experiment_name):
    # first make sure it works
    result = remote_sync_with_expon_backoff(
        experiemnt_configs.remote_sync_frequency,
        experiment_path,
        os.path.join(experiemnt_configs.remote_sync, experiment_name),
    )
    if result:
        logging.info("remote sync successful.")
    else:
        raise ValueError("Remote sync failed.")
    # if all looks good, start a process to do this every remote_sync_frequency seconds
    remote_sync_process = keep_running_remote_sync_process(
        experiemnt_configs.remote_sync_frequency,
        experiment_path,
        os.path.join(experiemnt_configs.remote_sync, experiment_name),
    )
    remote_sync_process.start()
    return remote_sync_process


def terminate_sync_process(p: multiprocessing.Process):
    if p is not None and p.is_alive():
        logging.info(f"Terminating remote sync process.")
        p.terminate()


def cleanup(sync_process, distributed=False):
    if sync_process:
        terminate_sync_process(sync_process)
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def end_remote_sync_process(remote_sync_process, experiemnt_configs, experiment_path, experiment_name):
    if remote_sync_process is not None:
        logging.info("Final remote sync.")
        terminate_sync_process(remote_sync_process)
        result = remote_sync_with_expon_backoff(
            experiemnt_configs.remote_sync_frequency,
            experiment_path,
            os.path.join(experiemnt_configs.remote_sync, experiment_name),
        )
        if result:
            logging.info("Final remote sync successful.")
        else:
            logging.info("Final remote sync failed.")


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

