import itertools
import logging
import wandb
import math
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.sampler import sample_chunk
from distributed import is_master
from precision import get_autocast
from meters import AverageMeter


def train_one_checkpoint(
    model, dataloader, loss, checkpoint_num, step, optimizer, scheduler, cfg,
):
    """Trains model for one checkpoint on the provided data.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the checkpoint. 
    """
    device = torch.device(cfg.distributed.device)
    autocast = get_autocast(cfg.experiment.precision)

    model.train()

    dataloader.set_checkpoint_num(checkpoint_num)
    num_batches_per_checkpoint = dataloader.dataloader.num_batches

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    forward_time_m = AverageMeter()
    backward_time_m = AverageMeter()
    optim_step_time_m = AverageMeter()
    sync_time_m = AverageMeter()

    end = time.time()
    data_iterator = iter(dataloader.dataloader)

    for i in itertools.count():
        scheduler(step)

        total_steps = cfg.experiment.total_train_samples // cfg.experiment.global_batch_size
        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        if cfg.distributed.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < cfg.distributed.world_size:      # Not all gpus have data
            break

        input_ids = batch['input_ids'].to(device)
        image = batch['pixel_values'].to(device) if 'pixel_values' in batch else None
        attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
 
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if cfg.experiment.accum_freq is not None:
            accum_freq = cfg.experiment.accum_freq
        else:
            combined_batch_size = cfg.distributed.world_size * cfg.experiment.per_gpu_batch_size
            assert cfg.experiment.global_batch_size % combined_batch_size == 0
            accum_freq = cfg.experiment.global_batch_size // combined_batch_size

        if accum_freq == 1:
            with autocast():
                forward_start = time.time()
                input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)
                if image is not None:
                    logits, _ = model(input_ids=input_ids, image=image, attention_mask=attention_mask)
                else:
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                forward_time_m.update(time.time() - forward_start)

                targets = targets.long()
                total_lm_loss = loss(logits.reshape(-1, cfg.model.vocab_size), targets.reshape(-1))
                total_loss = total_lm_loss

            backward_start = time.time()
            total_loss.backward()
            backward_time_m.update(time.time() - backward_start)

        else:
            input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)

            forward_total_time = 0
            backward_total_time = 0
            total_lm_loss = 0
            for ii in range(accum_freq):
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(model, FSDP) and ii != accum_freq - 1:
                    maybe_no_sync = model.no_sync
                with maybe_no_sync():
                    with autocast():
                        forward_start = time.time()
                        inputs_ii = input_ids[ii * cfg.experiment.per_gpu_batch_size : (ii + 1) * cfg.experiment.per_gpu_batch_size]
                        mask_ii = attention_mask[ii * cfg.experiment.per_gpu_batch_size : (ii + 1) * cfg.experiment.per_gpu_batch_size] if attention_mask is not None else None
                        if inputs_ii.shape[0] == 0:
                            break
                        targets_ii = targets[ii * cfg.experiment.per_gpu_batch_size : (ii + 1) * cfg.experiment.per_gpu_batch_size]
                        if image is not None:
                            images_ii = image[ii * cfg.experiment.per_gpu_batch_size : (ii + 1) * cfg.experiment.per_gpu_batch_size]
                            logits, _ = model(input_ids=inputs_ii, image=images_ii, attention_mask=mask_ii)
                        else:
                            logits, _ = model(input_ids=inputs_ii, attention_mask=mask_ii)
                        forward_total_time += time.time() - forward_start

                        targets_ii = targets_ii.long()
                        local_loss = (
                            loss(logits.reshape(-1, cfg.model.vocab_size), targets_ii.reshape(-1))
                            * (inputs_ii.shape[0] / input_ids.shape[0])
                        )
                    backward_start = time.time()
                    local_loss.backward()
                    backward_total_time += time.time() - backward_start    
                total_lm_loss += local_loss
                    
            forward_time_m.update(forward_total_time)
            backward_time_m.update(backward_total_time)
            total_loss = total_lm_loss

        optim_step_start = time.time()
        if cfg.experiment.grad_clip_norm is not None:
            if isinstance(model, FSDP):
                model.clip_grad_norm_(cfg.experiment.grad_clip_norm, norm_type=2.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.experiment.grad_clip_norm, norm_type=2.0)
        optimizer.step()
        optim_step_time_m.update(time.time() - optim_step_start)

        global_loss_tensor = total_loss.detach().clone()

        sync_start = time.time()
        if cfg.distributed.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)
        sync_time_m.update(time.time() - sync_start)
        batch_time_m.update(time.time() - end)
        end = time.time()

        batch_count = i + 1
        step += 1
        if is_master(cfg):
            batch_size = len(input_ids)
            # update the loss meter with the global loss tensor every iteration, so that the logging is of the avg of loss of the last
            # cfg.experiment.log_every_n_steps iterations
            losses_m.update(global_loss_tensor.item(), batch_size)
            if (i % cfg.experiment.log_every_n_steps == 0 and i > 0) or batch_count == num_batches_per_checkpoint or step == total_steps - 1:
                num_samples = batch_count * batch_size * cfg.distributed.world_size
                samples_per_checkpoint = dataloader.dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_checkpoint

                # gathered_loss = [torch.zeros_like(total_loss) for _ in range(cfg.distributed.world_size)]
                # torch.distributed.all_gather(gathered_loss, total_loss)

                # losses_m.update(sum(gathered_loss).item() / cfg.distributed.world_size, batch_size * cfg.distributed.world_size)
                losses_m.update(global_loss_tensor.item(), batch_size)
                samples_per_second = batch_size * cfg.distributed.world_size / batch_time_m.val
                samples_per_second_per_gpu = batch_size / batch_time_m.val
                tokens_per_second = input_ids.numel() * cfg.distributed.world_size / batch_time_m.val
                tokens_per_second_per_gpu = input_ids.numel() / batch_time_m.val
                loss_str = f"Loss: {losses_m.avg:.3f}"
                sample_digits = math.ceil(math.log(dataloader.dataloader.num_samples + 1, 10))
                logging.info(
                    f"Train Checkpoint: {checkpoint_num} [{num_samples:>{sample_digits}}/{samples_per_checkpoint} ({percent_complete:.0f}%)] "
                    f"{loss_str} "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
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
                    "tokens": (step + 1) * cfg.experiment.global_batch_size * cfg.data.seq_len,
                    "expected_steps_epoch": dataloader.dataloader.num_batches,
                    "seen_steps_epoch": batch_count,
                }

                for name, val in log_data.items():
                    name = "train/" + name
                    if cfg.experiment.wandb:
                        wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
                forward_time_m.reset()
                backward_time_m.reset()
                optim_step_time_m.reset()
                sync_time_m.reset()
                losses_m.reset()

    return True, step
