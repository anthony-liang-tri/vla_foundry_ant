"""
Main training entrypoint for the LBM2 project.

This module wires together configuration parsing, model/optimizer/scheduler
construction, dataset selection per checkpoint, and the core training loop.

Notes
- The training budget is specified in *samples*, not *steps*.
- This file aims to remain a *thin orchestrator*; most heavy lifting is
delegated to subpackages (data, models, opt, train, etc.).
"""

import json
import logging
import os

import draccus
import torch

from lbm2.data.dataloader import get_datastring_input, get_wds_dataloader
from lbm2.data.utils import load_data_chunks
from lbm2.distributed import get_model_precision, is_master, wrap_fsdp_ddp
from lbm2.file_utils import collect_processing_metadata, load_model_checkpoint, remote_sync, save_checkpoint
from lbm2.logger import setup_logging
from lbm2.losses import get_loss_function
from lbm2.models import create_model
from lbm2.optimizer import create_optimizer, load_optimizer
from lbm2.params.train_experiment_params import TrainExperimentParams
from lbm2.scheduler import create_scheduler
from lbm2.train import train_one_checkpoint
from lbm2.utils import get_experiment_name, set_random_seed


def main():
    """
    Entry point for launching training.

    Arguments are parsed with draccus.parse to instantiate a TrainExperimentParams object.
    They are provided as a preset yaml file, or as command line arguments or both.
    When using both a preset yaml file and command line arguments, the command line arguments take precedence.
    The preset yaml file is loaded with draccus.load, which supports !include statements to link a sub-preset yaml file.
    Other sub-preset yaml files can be passed as command line arguments with '--arg.subarg "include <path>"'.

    This function orchestrates experiment directory setup, (distributed) model
    construction, optimizer/scheduler creation, dataset selection per
    checkpoint, the training loop, and periodic checkpointing + remote sync.

    See README.md for more details.
    """
    # Parse config.
    cfg = draccus.parse(config_class=TrainExperimentParams)
    device = cfg.distributed.device
    # Seed rank-0 before any object creation for reproducibility.
    set_random_seed(cfg.hparams.seed, 0)

    # Set path for experiment, log, checkpoints.
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
        # Persist the resolved config.
        with open(os.path.join(experiment_path, "config.yaml"), "w") as f:
            draccus.dump(cfg, f)
        with open(os.path.join(experiment_path, "config_model.yaml"), "w") as f:
            draccus.dump(cfg.model, f)

        # Collect and save processing metadata from all data sources
        processing_metadata = collect_processing_metadata(cfg.data.dataset_manifest, experiment_path)
        if processing_metadata:
            with open(os.path.join(experiment_path, "processing_metadata.json"), "w") as f:
                json.dump(processing_metadata, f, indent=2)

        # Initial sync to check that remote_sync works.
        if cfg.remote_sync:
            remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))

    if cfg.distributed.use_distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {cfg.distributed.device}."
            f"Process (global: {cfg.distributed.rank}, local {cfg.distributed.local_rank}), "
            f"total {cfg.distributed.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {cfg.distributed.device}.")

    # Model construction
    model = create_model(cfg.model)
    # Re-seed with rank to randomize across workers.
    set_random_seed(cfg.hparams.seed, cfg.distributed.rank)
    if cfg.hparams.grad_checkpointing:
        model.set_grad_checkpointing()

    # Wrap for distributed or move to device with the configured precision.
    if cfg.distributed.use_distributed:
        model = wrap_fsdp_ddp(model, device, cfg)
    else:
        model = model.to(device, dtype=get_model_precision(cfg))

    # Optionally resume model from a checkpoint.
    start_checkpoint_num, global_step = 0, 0
    total_steps = cfg.total_train_samples // cfg.hparams.global_batch_size
    shard_shuffle_seed_per_dataset = None
    if cfg.model.resume_from_checkpoint is not None:
        if cfg.model.resume_weights_only:
            load_model_checkpoint(model, cfg.model.resume_from_checkpoint)
        else:
            start_checkpoint_num, global_step, shard_shuffle_seed_per_dataset = load_model_checkpoint(
                model, cfg.model.resume_from_checkpoint
            )

    # Create optimizer before torchcompile
    optimizer = create_optimizer(cfg.hparams, model)

    if cfg.hparams.torchcompile:
        logging.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    count_parameters(model)

    # Optionally resume optimizer state from a checkpoint.
    # This needs to be after torchcompile.
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        load_optimizer(optimizer, checkpoint_path=cfg.model.resume_from_checkpoint, use_fsdp=cfg.distributed.fsdp)

    # Create LR scheduler, loss function.
    scheduler = create_scheduler(cfg.hparams, optimizer, cfg.total_train_samples)
    loss = get_loss_function(cfg.hparams.loss_function, cfg.hparams)

    # Logging.
    if cfg.wandb and is_master(cfg):
        import wandb  # imported lazily to avoid hard dependency when disabled

        logging.debug("Starting wandb.")
        wandb.init(
            project=cfg.wandb_project_name,
            name=experiment_name,
            tags=[],
            resume=None,
            config=vars(cfg),
        )
        logging.debug("Wandb initialized.")

    done_training = global_step >= total_steps
    checkpoint_num = start_checkpoint_num
    # Per-dataset cursors and shuffle seeds allow resuming mixed datasets.
    curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
    if shard_shuffle_seed_per_dataset is None:
        shard_shuffle_seed_per_dataset = [cfg.hparams.seed for dataset in range(len(cfg.data.dataset_manifest))]

    samples_seen = 0
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        # Also restore which shards were consumed and how many samples were seen.
        curr_shard_idx_per_dataset, samples_seen = load_data_chunks(cfg.model.resume_from_checkpoint)

    # Main training loop
    while not done_training:
        if is_master(cfg):
            logging.info(f"Start checkpoint {checkpoint_num}")

        # Partition the global sample budget into evenly-sized checkpoint chunks.
        samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
        datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
            get_datastring_input(
                num_samples=samples_per_checkpoint,
                curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
                shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
                manifest_paths=cfg.data.dataset_manifest,
                dataset_weighting=cfg.data.dataset_weighting,
                allow_multiple_epochs=cfg.data.allow_multiple_epochs,
                num_workers_per_gpu=cfg.data.num_workers,
                world_size=cfg.distributed.world_size,
            )
        )

        if is_master(cfg):
            logging.info(f"Now training on: {datastrings}")
            logging.info(f"Samples: {samples_seen} / {cfg.total_train_samples}")
            logging.info(f"Samples in this checkpoint (per dataset): {num_samples_per_dataset}")

        # Safety check: ensure all ranks see the same data slice.
        if cfg.distributed.use_distributed:
            all_datastrings = ["" for _ in range(cfg.distributed.world_size)]
            torch.distributed.all_gather_object(all_datastrings, datastrings)
            assert all([x == datastrings for x in all_datastrings]), (
                "Dataset to train on is not the same across all nodes. This should not happen normally, "
                "unless there is an issue with shard shuffling during the dataset generation."
            )

        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, cfg)
        if is_master(cfg):
            dataloader.save_configs(experiment_path)  # Save any necessary dataloader/pipeline configs.
            if cfg.remote_sync:
                remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))

        prev_step = global_step

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        success, global_step = train_one_checkpoint(
            model,
            dataloader,
            loss,
            checkpoint_num,
            global_step,
            optimizer,
            scheduler,
            cfg,
        )
        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        # Translate newly completed steps into samples.
        samples_seen = samples_seen + (global_step - prev_step) * cfg.hparams.global_batch_size
        checkpoint_num += 1
        done_training = global_step >= total_steps

        # Persist training state (model/opt/scheduler + data cursors).
        save_checkpoint(
            cfg,
            checkpoint_num,
            checkpoint_path,
            cfg.max_checkpoint_limit,
            model,
            optimizer,
            datastrings,
            curr_shard_idx_per_dataset,
            samples_seen,
            global_step,
            shard_shuffle_seed_per_dataset,
        )

        # Optionally push artifacts to remote storage after each checkpoint.
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
