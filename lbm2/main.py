import os
import torch
import numpy as np
import random
import logging
import json
from logger import setup_logging

from params.params import get_params, get_args
from utils import get_experiment_name
from distributed import init_distributed_device, wrap_fsdp_ddp, is_master
from models import create_model
from optimizer import create_optimizer, load_optimizer
from scheduler import create_scheduler
from losses import CrossEntropyLossWithZLoss
from data.dataloader import get_wds_dataloader, get_datastring_input
from data.utils import load_data_chunks
from file_utils import save_checkpoint, load_model_checkpoint, remote_sync
from train import train_one_checkpoint


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main():
    cfg = get_params(get_args())

    # This populates cfg.distributed
    device = init_distributed_device(cfg.distributed)

    assert cfg.experiment.global_batch_size % cfg.distributed.world_size == 0
    if cfg.experiment.accum_freq is not None:
        assert cfg.experiment.accum_freq * cfg.distributed.world_size * cfg.experiment.per_gpu_batch_size == cfg.experiment.global_batch_size
    assert len(cfg.data.dataset_manifest) == len(cfg.data.dataset_modality)
    if cfg.data.dataset_weighting is not None:
        assert len(cfg.data.dataset_manifest) == len(cfg.data.dataset_weighting)
    if cfg.distributed.fsdp and not cfg.distributed.use_distributed:
        raise ValueError(f"--fsdp can only be specified in distributed mode.")
    
    # Do this before logging to out.log
    # This populates cfg.model 
    random_seed(cfg.experiment.seed, 0)
    model = create_model(cfg.model)

    experiment_name = get_experiment_name(cfg)
    if cfg.experiment.save_path is None:
        experiment_path = os.path.join("experiments", experiment_name)
    else:
        experiment_path = os.path.join(cfg.experiment.save_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, "out.log")
    setup_logging(log_path, logging.INFO)
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    if is_master(cfg):
        with open(os.path.join(experiment_path, "config.json"), "w") as f:
            json.dump(cfg.asdict(), f, indent=2)
        if cfg.experiment.remote_sync:      # Initial sync to check remote_sync works:
            remote_sync(experiment_path, os.path.join(cfg.experiment.remote_sync, experiment_name))

    if cfg.distributed.use_distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {cfg.distributed.device}."
            f"Process (global: {cfg.distributed.rank}, local {cfg.distributed.local_rank}), total {cfg.distributed.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {cfg.distributed.device}.")

    random_seed(cfg.experiment.seed, cfg.distributed.rank)
    if cfg.experiment.grad_checkpointing:
        model.set_grad_checkpointing()

    if cfg.distributed.use_distributed:
        model = wrap_fsdp_ddp(model, device, cfg)

    # optionally resume model from a checkpoint
    start_checkpoint_num, global_step = 0, 0
    total_steps = cfg.experiment.total_train_samples // cfg.experiment.global_batch_size
    shard_shuffle_seed = cfg.experiment.seed
    if cfg.experiment.resume_from_checkpoint is not None:
        if cfg.experiment.resume_weights_only:
            load_model_checkpoint(model, cfg.experiment, cfg.distributed)
        else:
            start_checkpoint_num, global_step, shard_shuffle_seed = load_model_checkpoint(model, cfg.experiment, cfg.distributed)
    
    # create optimizer before torchcompile
    optimizer = create_optimizer(cfg.experiment, model)

    if cfg.experiment.torchcompile:
        logging.info("Compiling model...")
        model = torch.compile(model)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    count_parameters(model)

    # optionally resume optimizer from a checkpoint
    # this needs to be after torchcompile
    if cfg.experiment.resume_from_checkpoint is not None and not cfg.experiment.resume_weights_only:
        load_optimizer(cfg.experiment.resume_from_checkpoint, cfg.distributed.fsdp, model, optimizer)

    scheduler = create_scheduler(cfg.experiment, optimizer)

    if cfg.experiment.wandb and is_master(cfg):
        import wandb
        logging.debug("Starting wandb.")
        wandb.init(
            project=cfg.experiment.wandb_project_name,
            name=experiment_name,
            tags=[],
            resume=None,
            config=vars(cfg),
        )
        logging.debug("Finished loading wandb.")

    loss = torch.nn.CrossEntropyLoss()
    if cfg.experiment.z_loss_coefficient != 0.0:
        logging.info("Using CrossEntropyLossWithZLoss.")
        loss = CrossEntropyLossWithZLoss(cfg.experiment.z_loss_coefficient)

    done_training = global_step >= total_steps
    checkpoint_num = start_checkpoint_num
    curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
    samples_seen = 0
    if cfg.experiment.resume_from_checkpoint is not None and not cfg.experiment.resume_weights_only:
        curr_shard_idx_per_dataset, samples_seen = load_data_chunks(cfg.experiment.resume_from_checkpoint)

    # Main training loop
    while not done_training:
        if is_master(cfg):
            logging.info(f"Start checkpoint {checkpoint_num}")

        # if cfg.experiment.checkpoint_strategy == "epoch":
        #     pass
        # elif cfg.experiment.checkpoint_strategy == "samples":
        #     curr_num_samples = cfg.experiment.checkpoint_num_samples
        
        samples_per_checkpoint = cfg.experiment.total_train_samples // cfg.experiment.num_checkpoints
        datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset = get_datastring_input(
            num_samples = samples_per_checkpoint,
            curr_shard_idx_per_dataset = curr_shard_idx_per_dataset, 
            manifest_paths = cfg.data.dataset_manifest,
            dataset_weighting = cfg.data.dataset_weighting,
            num_workers_per_gpu = 2,
            world_size = cfg.distributed.world_size,
            shard_shuffle_seed = shard_shuffle_seed,
        )
        if is_master(cfg):
            logging.info(f"Now training on: {datastrings}")
        
        if cfg.distributed.use_distributed:
            all_datastrings = ["" for _ in range(cfg.distributed.world_size)]
            torch.distributed.all_gather_object(all_datastrings, datastrings)
            assert all(
                [x == datastrings for x in all_datastrings]
            ), "Dataset to train on is not the same across all nodes. This should not happen normally, unless there is an issue with shard shuffling during the dataset generation."

        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, cfg)
        prev_step = global_step

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()
        success, global_step = train_one_checkpoint(
            model, dataloader, loss, checkpoint_num, global_step, optimizer, scheduler, cfg,
        )
        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        samples_seen = samples_seen + (global_step - prev_step) * cfg.experiment.global_batch_size
        checkpoint_num += 1
        done_training = global_step >= total_steps

        save_checkpoint(cfg, checkpoint_num, checkpoint_path, model, optimizer, datastrings, curr_shard_idx_per_dataset, samples_seen, global_step, shard_shuffle_seed)
        if is_master(cfg) and cfg.experiment.remote_sync:
            remote_sync(experiment_path, os.path.join(cfg.experiment.remote_sync, experiment_name))

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        if done_training:
            if is_master(cfg):
                logging.info("Model has seen the desired number of samples. Ending training.")
            break
        
    if cfg.experiment.wandb and is_master(cfg):
        wandb.finish()



if __name__ == "__main__":
    main()