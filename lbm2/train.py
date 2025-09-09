import itertools
import logging
import math
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.distributed_c10d import ReduceOp

from lbm2.data.sampler import sample_chunk
from lbm2.distributed import is_master
from lbm2.meters import AverageMeter
from lbm2.params.train_experiment_params import TrainExperimentParams
from lbm2.precision import get_autocast


def train_one_checkpoint(
    model: nn.Module,
    dataloader,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    checkpoint_num: int,
    step: int,
    optimizer: optim.Optimizer,
    scheduler: Callable[[int], None],
    cfg: TrainExperimentParams,
) -> tuple[bool, int]:
    """
    Trains model for one checkpoint on the provided data.

    This function:
      - Drives LR scheduling.
      - Performs forward/backward/step with optional gradient accumulation.
      - Computes and (optionally) all-reduces loss across ranks for logging.
      - Tracks timing/throughput metrics and logs periodically.
      - Exits when either:
          * the global training budget in samples is exhausted, or
          * the dataloader is depleted on any rank.

    Args:
        model: torch.nn.Module or a distributed-wrapped module.
        dataloader: dataloader object.
        loss: Callable loss function mapping (logits, targets) -> scalar loss.
        checkpoint_num: Index of the current checkpoint window (for logs).
        step: Current global training step **before** this window starts.
        optimizer: torch.optim.Optimizer instance.
        scheduler: Callable taking `step` and adjusting LR, etc.
        cfg: Training config.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the checkpoint.
    """
    device = torch.device(cfg.distributed.device)
    autocast = get_autocast(cfg.hparams.precision)

    model.train()

    # Let the dataloader know which window/checkpoint it's on.
    dataloader.set_checkpoint_num(checkpoint_num)
    num_batches_per_checkpoint = dataloader.dataloader.num_batches

    # Meters for logging.
    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    forward_time_m = AverageMeter()
    backward_time_m = AverageMeter()
    optim_step_time_m = AverageMeter()
    sync_time_m = AverageMeter()

    end = time.time()
    data_iterator = iter(dataloader.dataloader)

    # Open-ended loop; we break on budget or data exhaustion.
    for i in itertools.count():
        scheduler(step)

        # Hard-stop when we reach the sample budget translated into steps.
        total_steps = cfg.total_train_samples // cfg.hparams.global_batch_size
        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        # Try to fetch the next batch on this rank.
        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        # Ensure all ranks still have data; if any rank is out, break.
        if cfg.distributed.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < cfg.distributed.world_size:  # Not all gpus have data
            break

        input_ids = batch["input_ids"].to(device)
        image = batch["pixel_values"].to(device) if "pixel_values" in batch else None
        attention_mask = (
            batch["attention_mask"].to(device)
            if "attention_mask" in batch and batch["attention_mask"] is not None
            else None
        )

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if cfg.hparams.accum_freq == 1:
            # No gradient accumulation
            with autocast():
                forward_start = time.time()
                # Sample a contiguous chunk to the configured sequence length.
                input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)

                if cfg.model.type == "transformer" or cfg.model.type == "transformer_hf":
                    logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
                    forward_time_m.update(time.time() - forward_start)
                    targets = targets.long()
                    vocab_size = logits.shape[-1]
                    total_loss = loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
                elif cfg.model.type == "vlm" or cfg.model.type == "vlm_hf":
                    logits, _, _ = model(
                        input_ids=input_ids, image=image, attention_mask=attention_mask, output_hidden_states=False
                    )
                    forward_time_m.update(time.time() - forward_start)
                    targets = targets.long()
                    # Mask out padding and image-token positions when computing loss.
                    ignore_mask = (targets == cfg.data.pad_token_id) | (targets == cfg.data.image_token_id)
                    targets = targets.masked_fill(ignore_mask, -100)
                    vocab_size = logits.shape[-1]
                    total_loss = loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
                elif cfg.model.type == "stable_diffusion":
                    noise = torch.randn_like(image)
                    predicted_noise = model(
                        input_ids=input_ids, image=image, attention_mask=attention_mask, noise=noise
                    )
                    if getattr(cfg.model, "diffusion_use_flow_matching_scheduler", False):
                        # In flow-matching variant: predict (image -> noise) direction.
                        noise = noise - image
                    total_loss = loss(predicted_noise, noise)

            # Backward for single-step case.
            backward_start = time.time()
            total_loss.backward()
            backward_time_m.update(time.time() - backward_start)

        else:
            # Gradient accumulation path.
            input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)

            forward_total_time = 0
            backward_total_time = 0
            total_lm_loss = 0
            for ii in range(cfg.hparams.accum_freq):
                # Don't sync gradients until the final microbatch for FSDP.
                if cfg.distributed.fsdp:
                    is_final_accum = ii == cfg.hparams.accum_freq - 1
                    model.set_requires_gradient_sync(is_final_accum)
                    model.set_requires_all_reduce(is_final_accum)
                    model.set_reshard_after_backward(is_final_accum)
                    model.set_is_last_backward(is_final_accum)

                with autocast():
                    forward_start = time.time()
                    # Slice the microbatch for this accumulation step.
                    inputs_ii = input_ids[
                        ii * cfg.hparams.per_gpu_batch_size : (ii + 1) * cfg.hparams.per_gpu_batch_size
                    ]
                    mask_ii = (
                        attention_mask[ii * cfg.hparams.per_gpu_batch_size : (ii + 1) * cfg.hparams.per_gpu_batch_size]
                        if attention_mask is not None
                        else None
                    )
                    if inputs_ii.shape[0] == 0:
                        break
                    targets_ii = targets[
                        ii * cfg.hparams.per_gpu_batch_size : (ii + 1) * cfg.hparams.per_gpu_batch_size
                    ]
                    if image is not None:
                        images_ii = image[
                            ii * cfg.hparams.per_gpu_batch_size : (ii + 1) * cfg.hparams.per_gpu_batch_size
                        ]

                    if cfg.model.type == "transformer" or cfg.model.type == "transformer_hf":
                        logits, _, _ = model(input_ids=inputs_ii, attention_mask=mask_ii, output_hidden_states=False)
                        forward_total_time += time.time() - forward_start
                        targets_ii = targets_ii.long()
                        vocab_size = logits.shape[-1]
                        local_loss = loss(logits.reshape(-1, vocab_size), targets_ii.reshape(-1)) * (
                            inputs_ii.shape[0] / input_ids.shape[0]
                        )
                    elif cfg.model.type == "vlm" or cfg.model.type == "vlm_hf":
                        logits, _, _ = model(
                            input_ids=inputs_ii, image=images_ii, attention_mask=mask_ii, output_hidden_states=False
                        )
                        forward_total_time += time.time() - forward_start
                        targets_ii = targets_ii.long()
                        ignore_mask = (targets_ii == cfg.data.pad_token_id) | (targets_ii == cfg.data.image_token_id)
                        targets_ii = targets_ii.masked_fill(ignore_mask, -100)
                        vocab_size = logits.shape[-1]
                        local_loss = loss(logits.reshape(-1, vocab_size), targets_ii.reshape(-1)) * (
                            inputs_ii.shape[0] / input_ids.shape[0]
                        )
                    elif cfg.model.type == "stable_diffusion":
                        noise = torch.randn_like(images_ii)
                        predicted_noise = model(
                            input_ids=inputs_ii, image=images_ii, attention_mask=mask_ii, noise=noise
                        )
                        if getattr(cfg.model, "diffusion_use_flow_matching_scheduler", False):
                            noise = noise - images_ii  # Predict the direction from image to noise
                        local_loss = loss(predicted_noise, noise) * (inputs_ii.shape[0] / input_ids.shape[0])

                # Backward per microbatch.
                backward_start = time.time()
                local_loss.backward()
                backward_total_time += time.time() - backward_start
                total_lm_loss += local_loss

            forward_time_m.update(forward_total_time)
            backward_time_m.update(backward_total_time)
            total_loss = total_lm_loss

        # Optimizer step
        optim_step_start = time.time()
        # (Optional) grad clipping
        if cfg.hparams.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hparams.grad_clip_norm, norm_type=2.0)
        optimizer.step()
        optim_step_time_m.update(time.time() - optim_step_start)

        # For logging: clone a tensor copy of the loss and average across ranks.
        global_loss_tensor = total_loss.detach().clone()

        sync_start = time.time()
        if cfg.distributed.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)
        sync_time_m.update(time.time() - sync_start)

        # Update timing meters for this iteration.
        batch_time_m.update(time.time() - end)
        end = time.time()

        batch_count = i + 1
        step += 1  # Advance the global step after completing this batch.

        # Master-only logging & W&B
        if is_master(cfg):
            batch_size = len(input_ids)
            # update the loss meter with the global loss tensor every iteration,
            # so that the logging is of the avg of loss of the last cfg.log_every_n_steps iterations
            losses_m.update(global_loss_tensor.item(), batch_size)

            # Periodic log or end-of-window/end-of-training log.
            if (
                (i % cfg.log_every_n_steps == 0 and i > 0)
                or batch_count == num_batches_per_checkpoint
                or step == total_steps - 1
            ):
                num_samples = batch_count * batch_size * cfg.distributed.world_size
                samples_per_checkpoint = dataloader.dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_checkpoint

                # Throughput stats (samples / tokens per second).
                samples_per_second = batch_size * cfg.distributed.world_size / batch_time_m.val
                samples_per_second_per_gpu = batch_size / batch_time_m.val
                tokens_per_second = input_ids.numel() * cfg.distributed.world_size / batch_time_m.val
                tokens_per_second_per_gpu = input_ids.numel() / batch_time_m.val

                loss_str = f"Loss: {losses_m.avg:.3f}"
                sample_digits = math.ceil(math.log(dataloader.dataloader.num_samples + 1, 10))
                logging.info(
                    f"Train Checkpoint: {checkpoint_num} "
                    f"[{num_samples:>{sample_digits}}/{samples_per_checkpoint} "
                    f"({percent_complete:.0f}%)] "
                    f"{loss_str} "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, "
                    f"{samples_per_second:#g}/s, "
                    f"{samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": losses_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "forward_time": forward_time_m.val,
                    "backward_time": backward_time_m.val,
                    "optim_step_time": optim_step_time_m.val,
                    "sync_time": sync_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens": (step + 1) * cfg.hparams.global_batch_size * cfg.data.seq_len,
                    "samples": (step + 1) * cfg.hparams.global_batch_size,
                    "expected_steps_epoch": dataloader.dataloader.num_batches,
                    "seen_steps_epoch": batch_count,
                }

                for name, val in log_data.items():
                    name = "train/" + name
                    if cfg.wandb:
                        import wandb

                        wandb.log(
                            {name: val, "step": step, "tokens": log_data["tokens"], "samples": log_data["samples"]}
                        )

                # Reset short-horizon meters so next window reflects recent perf.
                batch_time_m.reset()
                data_time_m.reset()
                forward_time_m.reset()
                backward_time_m.reset()
                optim_step_time_m.reset()
                sync_time_m.reset()
                losses_m.reset()

    return True, step
