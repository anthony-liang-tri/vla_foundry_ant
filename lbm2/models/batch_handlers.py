"""
Batch handlers for different model types.

This module contains batch preparation and loss computation logic for each model type,
eliminating the need for if/elif statements in the training loop.
"""

from abc import ABC, abstractmethod

import torch

from lbm2.data.sampler import sample_chunk


class BatchHandler(ABC):
    """Abstract base class for model-specific batch handlers."""

    @abstractmethod
    def prepare_inputs(self, batch, device, model_dtype, cfg):
        """
        Prepare model inputs from batch data.

        Args:
            batch: Raw batch dictionary from dataloader
            device: Target device for tensors
            model_dtype: Target dtype for model inputs
            cfg: Training configuration

        Returns:
            Dictionary of inputs ready for model(**inputs)
        """
        pass

    @abstractmethod
    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        """
        Prepare model inputs and targets from batch data, including chunking if needed.

        Args:
            batch: Raw batch dictionary from dataloader
            device: Target device for tensors
            model_dtype: Target dtype for model inputs
            cfg: Training configuration

        Returns:
            Tuple of (model_inputs_dict, targets_tensor)
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets, loss_fn, cfg):
        """
        Compute loss from model outputs and targets.

        Args:
            outputs: Model outputs
            targets: Target tensor (if needed)
            loss_fn: Loss function
            cfg: Training configuration

        Returns:
            Loss tensor
        """
        pass

    def slice_inputs_for_accumulation(self, model_inputs, start_idx, end_idx):
        """Slice model inputs for gradient accumulation microbatches."""
        sliced_inputs = {}
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                sliced_inputs[key] = value[start_idx:end_idx]
            else:
                # Non-tensor values or scalars pass through unchanged
                sliced_inputs[key] = value
        return sliced_inputs


class TransformerBatchHandler(BatchHandler):
    """Handles batch preparation for transformer and transformer_hf models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
            "output_hidden_states": False,
        }

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        # Move to device first
        input_ids = batch["input_ids"].to(device, non_blocking=True, dtype=torch.long)
        attention_mask = (
            batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)
            if "attention_mask" in batch and batch["attention_mask"] is not None
            else None
        )

        # Sample a contiguous chunk to the configured sequence length
        input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)

        # Prepare model inputs
        model_inputs = {"input_ids": input_ids, "output_hidden_states": False}

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        return model_inputs, targets

    def compute_loss(self, outputs, targets, loss_fn, cfg):
        logits = outputs.logits
        targets = targets.long()
        vocab_size = logits.shape[-1]
        return loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))


class VLMBatchHandler(BatchHandler):
    """Handles batch preparation for vlm and vlm_hf models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
            "output_hidden_states": False,
        }

        if "pixel_values" in batch:
            if cfg.model.type == "vlm_hf":
                # HF VLM models expect pixel_values, not image
                inputs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
            else:
                # Custom VLM models expect image parameter
                inputs["image"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        # Move to device first
        input_ids = batch["input_ids"].to(device, non_blocking=True, dtype=torch.long)
        attention_mask = (
            batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)
            if "attention_mask" in batch and batch["attention_mask"] is not None
            else None
        )

        # Sample a contiguous chunk to the configured sequence length
        input_ids, attention_mask, targets = sample_chunk(input_ids, attention_mask, cfg.data.seq_len)

        # Prepare model inputs
        model_inputs = {"input_ids": input_ids, "output_hidden_states": False}

        if "pixel_values" in batch:
            if cfg.model.type == "vlm_hf":
                # HF VLM models expect pixel_values, not image
                model_inputs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
            else:
                # Custom VLM models expect image parameter
                model_inputs["image"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        return model_inputs, targets

    def compute_loss(self, outputs, targets, loss_fn, cfg):
        logits = outputs.logits
        targets = targets.long()
        # Mask out padding and image-token positions when computing loss
        ignore_mask = (targets == cfg.data.pad_token_id) | (targets == cfg.data.image_token_id)
        targets = targets.masked_fill(ignore_mask, -100)
        vocab_size = logits.shape[-1]
        return loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))


class StableDiffusionBatchHandler(BatchHandler):
    """Handles batch preparation for stable_diffusion models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        image = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
        noise = torch.randn_like(image)

        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
            "image": image,
            "noise": noise,
        }

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        # For diffusion models, we don't do sequence chunking like text models
        # Instead, we prepare the image and noise directly
        image = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
        noise = torch.randn_like(image)

        # Prepare model inputs
        model_inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
            "image": image,
            "noise": noise,
        }

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            model_inputs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        # For diffusion, targets are the noise (or noise direction for flow matching)
        targets = noise
        if getattr(cfg.model, "diffusion_use_flow_matching_scheduler", False):
            # In flow-matching variant: target is (noise - image) direction
            targets = noise - image

        return model_inputs, targets

    def compute_loss(self, outputs, targets, loss_fn, cfg):
        predicted_noise = outputs
        return loss_fn(predicted_noise, targets)


class FakePolicyBatchHandler(BatchHandler):
    """Handles batch preparation for fake_policy models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
        }

        inputs["image"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
        inputs["actions"] = batch["actions"].to(device, non_blocking=True, dtype=model_dtype)
        inputs["proprioception"] = batch["proprioception"].to(device, non_blocking=True, dtype=model_dtype)
        inputs["past_mask"] = batch["past_mask"].to(device, non_blocking=True, dtype=model_dtype)
        inputs["future_mask"] = batch["future_mask"].to(device, non_blocking=True, dtype=model_dtype)

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        inputs = self.prepare_inputs(batch, device, model_dtype, cfg)
        targets = batch["actions"].to(device, non_blocking=True, dtype=model_dtype)
        return inputs, targets

    def compute_loss(self, outputs, targets, loss_fn, cfg):
        # For fake_policy, the model already computes the loss internally
        return outputs.loss


def create_batch_handler(model_type: str) -> BatchHandler:
    """
    Factory function to create the appropriate batch handler for a model type.

    Args:
        model_type: The type of model (e.g., 'transformer', 'vlm', etc.)

    Returns:
        BatchHandler instance for the specified model type
    """
    if model_type in ["transformer", "transformer_hf"]:
        return TransformerBatchHandler()
    elif model_type in ["vlm", "vlm_hf"]:
        return VLMBatchHandler()
    elif model_type == "stable_diffusion":
        return StableDiffusionBatchHandler()
    elif model_type == "fake_policy":
        return FakePolicyBatchHandler()
    else:
        raise ValueError(f"Batch handler not supported for model type: {model_type}")
