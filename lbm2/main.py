import os
import torch
import numpy as np
import random
import logging
import json
from lbm2.logger import setup_logging
import draccus

from lbm2.params.train_experiment_params import TrainExperimentParams
from lbm2.utils import get_experiment_name
from lbm2.distributed import init_distributed_device, wrap_fsdp_ddp, is_master, get_model_precision
from lbm2.models import create_model
from lbm2.optimizer import create_optimizer, load_optimizer
from lbm2.scheduler import create_scheduler
from lbm2.losses import get_loss_function
from lbm2.data.dataloader import get_wds_dataloader, get_datastring_input
from lbm2.data.utils import load_data_chunks
from lbm2.file_utils import save_checkpoint, load_model_checkpoint, remote_sync
from lbm2.train import train_one_checkpoint


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main():
    cfg = draccus.parse(config_class=TrainExperimentParams)
    device = cfg.distributed.device
    random_seed(cfg.hparams.seed, 0)

    experiment_name = get_experiment_name(cfg)
    if cfg.save_path is None:
        experiment_path = os.path.join("experiments", experiment_name)
    else:
        experiment_path = os.path.join(cfg.save_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, "out.log")
    setup_logging(log_path, logging.INFO)
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    if is_master(cfg):
        draccus.dump(cfg, open(os.path.join(experiment_path, "config.yaml"),'w')) 
        if cfg.remote_sync:      # Initial sync to check remote_sync works:
            remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))

    if cfg.distributed.use_distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {cfg.distributed.device}."
            f"Process (global: {cfg.distributed.rank}, local {cfg.distributed.local_rank}), total {cfg.distributed.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {cfg.distributed.device}.")

    model = create_model(cfg.model)
    random_seed(cfg.hparams.seed, cfg.distributed.rank)
    if cfg.hparams.grad_checkpointing:
        model.set_grad_checkpointing()

    if cfg.distributed.use_distributed:
        model = wrap_fsdp_ddp(model, device, cfg)
    else:
        model = model.to(device, dtype=get_model_precision(cfg))

    # optionally resume model from a checkpoint
    start_checkpoint_num, global_step = 0, 0
    total_steps = cfg.total_train_samples // cfg.hparams.global_batch_size
    shard_shuffle_seed = cfg.hparams.seed
    if cfg.model.resume_from_checkpoint is not None:
        if cfg.model.resume_weights_only:
            load_model_checkpoint(model, cfg.model.resume_from_checkpoint, cfg.distributed)
        else:
            start_checkpoint_num, global_step, shard_shuffle_seed = load_model_checkpoint(model, cfg.model.resume_from_checkpoint, cfg.distributed)
    
    # create optimizer before torchcompile
    optimizer = create_optimizer(cfg.hparams, model)

    if cfg.hparams.torchcompile:
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
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        load_optimizer(cfg.model.resume_from_checkpoint, cfg.distributed.fsdp, model, optimizer)

    scheduler = create_scheduler(cfg.hparams, optimizer, cfg.total_train_samples)
    loss = get_loss_function(cfg.hparams.loss_function, cfg.hparams)

    if cfg.wandb and is_master(cfg):
        import wandb
        logging.debug("Starting wandb.")
        wandb.init(
            project=cfg.wandb_project_name,
            name=experiment_name,
            tags=[],
            resume=None,
            config=vars(cfg),
        )
        logging.debug("Finished loading wandb.")

    done_training = global_step >= total_steps
    checkpoint_num = start_checkpoint_num
    curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
    shard_shuffle_seed_per_dataset = [shard_shuffle_seed for dataset in range(len(cfg.data.dataset_manifest))]
    samples_seen = 0
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        curr_shard_idx_per_dataset, samples_seen = load_data_chunks(cfg.model.resume_from_checkpoint)

    # Main training loop
    while not done_training:
        if is_master(cfg):
            logging.info(f"Start checkpoint {checkpoint_num}")
        
        samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
        datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = get_datastring_input(
            num_samples = samples_per_checkpoint,
            curr_shard_idx_per_dataset = curr_shard_idx_per_dataset, 
            shard_shuffle_seed_per_dataset = shard_shuffle_seed_per_dataset,
            manifest_paths = cfg.data.dataset_manifest,
            dataset_weighting = cfg.data.dataset_weighting,
            allow_multiple_epochs = cfg.data.allow_multiple_epochs,
            num_workers_per_gpu = cfg.data.num_workers,
            world_size = cfg.distributed.world_size,
        )
        if is_master(cfg):
            logging.info(f"Now training on: {datastrings}")
            logging.info(f"Samples: {samples_seen} / {cfg.total_train_samples}")
            logging.info(f"Samples in this checkpoint: {num_samples_per_dataset}")
        
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

        samples_seen = samples_seen + (global_step - prev_step) * cfg.hparams.global_batch_size
        checkpoint_num += 1
        done_training = global_step >= total_steps

        save_checkpoint(cfg, checkpoint_num, checkpoint_path, model, optimizer, datastrings, curr_shard_idx_per_dataset, samples_seen, global_step, shard_shuffle_seed_per_dataset)
        if is_master(cfg) and cfg.remote_sync:
            remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        if done_training:
            if is_master(cfg):
                logging.info("Model has seen the desired number of samples. Ending training.")
            break
        
    if cfg.wandb and is_master(cfg):
        wandb.finish()



if __name__ == "__main__":
    main()