"""
Batch handlers for different model types.

This module contains batch preparation and loss computation logic for each model type,
eliminating the need for if/elif statements in the training loop.

Batch handlers are registered using decorators from the registry module.
"""

from abc import ABC, abstractmethod

import torch

from vla_foundry.data.sampler import sample_chunk
from vla_foundry.models.registry import register_batch_handler


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
            Tuple of (model_inputs_dict, targets_tensor, mask_tensor)

        Note:
            The returned mask and model_inputs["future_mask"] are mutually exclusive:
            - LLM/VLM handlers return a mask (for padding/image tokens) and no future_mask
            - Diffusion policy handlers return mask=None and put future_mask in model_inputs
            The training loop validates this invariant.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        """
        Compute loss from model outputs and targets.

        Args:
            outputs: Model outputs
            targets: Target tensor (if needed)
            loss_fn: Loss function
            cfg: Training configuration
            mask: Mask of valid actions (should be broadcastable to the shape of outputs)

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

    def slice_targets_for_accumulation(self, targets, start_idx, end_idx, sliced_inputs=None):
        """Slice targets for gradient accumulation microbatches.

        Args:
            targets: Full targets tensor.
            start_idx: Start index for slicing.
            end_idx: End index for slicing.
            sliced_inputs: The already-sliced model inputs (from slice_inputs_for_accumulation).
                Subclasses may use this to recompute targets when slicing changes inputs
                (e.g., fresh noise generation with num_action_head_repeats).
        """
        return targets[start_idx:end_idx]


@register_batch_handler("transformer")
@register_batch_handler("transformer_hf")
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

        # Mask out pad tokens from loss computation
        if hasattr(cfg.data, "pad_token_id") and cfg.data.pad_token_id is not None:
            mask = targets == cfg.data.pad_token_id
        else:
            mask = None

        # For reference:
        # model_inputs["attention_mask"] tells us which tokens the model skips in forward (e.g. padding)
        # mask tells us which tokens the model skips in loss computation (e.g. padding, image tokens)
        # For LLMs, these should be the same but shifted by one token due to the autoregressive nature.
        return model_inputs, targets, mask

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(outputs.logits, targets, mask=mask)


@register_batch_handler("vlm")
@register_batch_handler("vlm_hf")
class VLMBatchHandler(BatchHandler):
    """Handles batch preparation for vlm and vlm_hf models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
            "output_hidden_states": False,
        }

        if "pixel_values" in batch:
            inputs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)

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
            model_inputs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        mask = (targets == cfg.data.pad_token_id) | (targets == cfg.data.image_token_id)

        # model_inputs["attention_mask"] tells us which tokens the model skips in forward (e.g. padding)
        # mask tells us which tokens the model skips in loss computation (e.g. padding, image tokens)
        return model_inputs, targets, mask

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(outputs.logits, targets, mask=mask)


@register_batch_handler("stable_diffusion")
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
        if cfg.model.use_flow_matching_scheduler:
            # In flow-matching variant: target is (noise - image) direction
            targets = noise - image

        return model_inputs, targets, None

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        predicted_direction = outputs
        return loss_fn(predicted_direction, targets, mask=mask)


@register_batch_handler("diffusion_policy")
class DiffusionPolicyBatchHandler(BatchHandler):
    """Handles batch preparation for diffusion policy models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        actions = batch["actions"].to(device, non_blocking=True, dtype=model_dtype)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True, dtype=model_dtype)
        input_ids = batch["input_ids"].to(device, non_blocking=True, dtype=torch.long)
        attention_mask = (
            batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)
            if "attention_mask" in batch and batch["attention_mask"] is not None
            else None
        )
        attention_mask_images = (
            batch["attention_mask_images"].to(device, non_blocking=True, dtype=torch.bool)
            if "attention_mask_images" in batch and batch["attention_mask_images"] is not None
            else None
        )
        past_mask = batch["past_mask"].to(device, non_blocking=True, dtype=torch.bool)
        future_mask = batch["future_mask"].to(device, non_blocking=True, dtype=torch.bool)
        proprioception = batch.get("proprioception")
        if proprioception is not None:
            proprioception = proprioception.to(device, non_blocking=True, dtype=model_dtype)

        noise = torch.randn_like(actions)
        self._num_action_head_repeats = getattr(cfg.model, "num_action_head_repeats", None)
        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "actions": actions,
            "noise": noise,
            "attention_mask": attention_mask,
            "attention_mask_images": attention_mask_images,
            "past_mask": past_mask,
            "future_mask": future_mask,
        }
        if proprioception is not None:
            inputs["proprioception"] = proprioception
        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        inputs = self.prepare_inputs(batch, device, model_dtype, cfg)
        targets = inputs["noise"] - inputs["actions"]
        return inputs, targets, None

    # Keys whose batch dimension corresponds to the action head (tiled to [B*N]).
    _ACTION_SIDE_KEYS = frozenset({"actions", "noise", "past_mask", "future_mask", "proprioception"})

    def slice_inputs_for_accumulation(self, model_inputs, start_idx, end_idx):
        """Slice inputs for gradient accumulation, then apply num_repeats tiling.

        All tensors in model_inputs are at uniform batch size [B_full].
        After slicing the microbatch [start_idx:end_idx], action-side tensors
        are repeat_interleaved to [micro_batch * N] and N distinct noises are
        generated, while VLM-side tensors stay at [micro_batch].
        """
        sliced = super().slice_inputs_for_accumulation(model_inputs, start_idx, end_idx)

        num_repeats = getattr(self, "_num_action_head_repeats", None)
        if num_repeats is not None and num_repeats > 1:
            for key in self._ACTION_SIDE_KEYS:
                if key in sliced and isinstance(sliced[key], torch.Tensor):
                    sliced[key] = sliced[key].repeat_interleave(num_repeats, dim=0)
            # Generate N distinct noise samples per microbatch element.
            actions = sliced["actions"]
            sliced["noise"] = torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
            )

        return sliced

    def slice_targets_for_accumulation(self, targets, start_idx, end_idx, sliced_inputs=None):
        """Recompute targets from sliced inputs when num_repeats > 1.

        Fresh noise is generated in slice_inputs_for_accumulation, so the
        pre-computed targets (from the original noise) are stale.  Recompute
        as ``noise - actions`` from the already-sliced (and possibly repeated)
        model inputs.
        """
        num_repeats = getattr(self, "_num_action_head_repeats", None)
        if num_repeats is not None and num_repeats > 1:
            assert sliced_inputs is not None, (
                "sliced_inputs is required to recompute targets with num_action_head_repeats"
            )
            return sliced_inputs["noise"] - sliced_inputs["actions"]
        return targets[start_idx:end_idx]

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        # Reshape inputs and masks to match shapes
        predicted_direction = outputs
        target_direction = targets

        # Depending on the input strategy (past given in the same sequence or separate),
        # the mask may be shorter or longer than the loss
        if mask is not None:
            seq_len = min(mask.shape[1], predicted_direction.shape[1])
            predicted_direction = predicted_direction[:, -seq_len:]
            target_direction = target_direction[:, -seq_len:]
            mask = mask[:, -seq_len:]

        return loss_fn(input=predicted_direction, target=target_direction, mask=mask)


@register_batch_handler("maniflow")
class ManiFlowBatchHandler(BatchHandler):
    """Handles batch preparation for ManiFlow consistency flow models."""

    def prepare_inputs(self, batch, device, model_dtype, cfg):
        """Prepare inputs for ManiFlow inference."""
        # Load point cloud if available and enabled
        if batch.get("point_cloud") is not None and cfg.data.use_point_cloud:
            point_cloud = batch["point_cloud"].to(device, non_blocking=True, dtype=model_dtype)
        else:
            point_cloud = None

        proprioception = batch["proprioception"].to(device, non_blocking=True, dtype=model_dtype)

        inputs = {
            "point_cloud": point_cloud,
            "proprioception": proprioception,
        }

        # Add language conditioning if present
        if "task_name" in batch and batch["task_name"] is not None:
            inputs["task_name"] = batch["task_name"]

        return inputs

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        """Prepare inputs and targets for ManiFlow training.

        ManiFlow computes its own loss internally with flow and consistency objectives.
        We structure the inputs so ManiFlow.forward() receives batch= kwargs.
        EMA model should be set via model.set_ema_model() before training.
        """
        # Load point cloud if available and enabled
        if batch.get("point_cloud") is not None and cfg.data.use_point_cloud:
            point_cloud = batch["point_cloud"].to(device, non_blocking=True, dtype=model_dtype)
        else:
            point_cloud = None

        proprioception = batch["proprioception"].to(device, non_blocking=True, dtype=model_dtype)
        actions = batch["actions"].to(device, non_blocking=True, dtype=model_dtype)
        input_ids = batch["input_ids"].to(device, non_blocking=True, dtype=torch.long)

        # Prepare inputs for ManiFlow.forward() - flat structure for gradient accumulation compatibility
        model_inputs = {
            "point_cloud": point_cloud,
            "proprioception": proprioception,
            "actions": actions,
            "input_ids": input_ids,
        }

        # Add language conditioning if present
        if "task_name" in batch and batch["task_name"] is not None:
            model_inputs["task_name"] = batch["task_name"]

        # Create dummy targets tensor for training loop compatibility (not actually used)
        # Training loop expects targets to be sliceable, but ManiFlow computes loss internally
        dummy_targets = torch.zeros((actions.shape[0],), device=actions.device)

        return model_inputs, dummy_targets, None

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        """Compute loss for ManiFlow.

        Since ManiFlow.compute_loss returns (loss, loss_dict), we need to extract just the loss.
        The outputs here should be the (loss, loss_dict) tuple from model.compute_loss.
        """
        if isinstance(outputs, tuple) and len(outputs) == 2:
            loss, loss_dict = outputs
            return loss
        else:
            # If just a scalar loss is returned
            return outputs
