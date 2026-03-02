#! /usr/bin/env python3
"""
LBM Policy Server for DiffusionPolicy model.

This policy uses a trained DiffusionPolicy model to generate robot actions based on
visual observations and language instructions.
"""

import argparse
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict

import torch
from grpc_workspace.git_util import (
    maybe_get_current_commit_sha,
    maybe_get_remote_url_from_active_branch,
)
from grpc_workspace.lbm_policy_server import (
    LbmPolicyServerConfig,
    run_policy_server,
)
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import Policy, PolicyMetadata

import vla_foundry.visualizers.visualizer as vz
from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import (
    get_latest_checkpoint,
    load_ema_checkpoint,
    load_model_checkpoint,
    yaml_load,
)
from vla_foundry.inference.robotics.data_adapter import PolicyDataAdapter
from vla_foundry.logger import setup_logging
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml
from vla_foundry.precision import get_autocast
from vla_foundry.visualizers import visualizer


def _get_policy_metadata():
    return PolicyMetadata(
        name="LBMDiffusionPolicy",
        skill_type="LanguageConditionedManipulation",
        checkpoint_path="None",  # Will be set by the policy
        git_repo=maybe_get_remote_url_from_active_branch("Unknown"),
        git_sha=maybe_get_current_commit_sha("Undefined"),
        is_language_conditioned=True,
    )


class InferenceDiffusionPolicy(Policy):
    """A policy that uses DiffusionPolicy model for language-conditioned robot manipulation."""

    def __init__(
        self,
        checkpoint_directory: str,
        checkpoint_name: str = None,
        open_loop_steps: int = 4,
        device: str = "cuda",
        num_flow_steps: int = 10,
    ):
        self.model_config_path = os.path.join(checkpoint_directory, "config.yaml")

        # Load model configuration first to get EMA enabled setting
        self.cfg = load_experiment_params_from_yaml(
            self.model_config_path, localize_params=not self.model_config_path.startswith("s3://")
        )

        self.ema_enabled = self.cfg.ema.enabled

        if checkpoint_name is None or checkpoint_name == "":
            checkpoint_name = get_latest_checkpoint(checkpoint_directory)
            # get_latest_checkpoint returns full path, extract just the filename
            if checkpoint_name:
                checkpoint_name = os.path.basename(checkpoint_name)
        if not checkpoint_name.endswith(".pt"):
            checkpoint_name = f"{checkpoint_name}.pt"

        # Use EMA checkpoint if enabled in config
        if self.ema_enabled:
            # Replace "checkpoint_" with "ema_" to get EMA checkpoint path
            checkpoint_name = checkpoint_name.replace("checkpoint_", "ema_")
            if not checkpoint_name.startswith("ema_"):
                # If checkpoint name doesn't start with "checkpoint_", prepend "ema_"
                checkpoint_name = f"ema_{checkpoint_name}"

        self.checkpoint_path = os.path.join(checkpoint_directory, "checkpoints", f"{checkpoint_name}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.open_loop_steps = open_loop_steps
        self.current_open_loop_step = defaultdict(int)
        self.num_flow_steps = num_flow_steps

        # Load model configuration and create model
        self.cfg = load_experiment_params_from_yaml(
            self.model_config_path, localize_params=not self.model_config_path.startswith("s3://")
        )
        # Use load_pretrained=False to skip downloading pretrained weights (they'll be loaded from checkpoint)
        self.model = create_model(self.cfg.model, load_pretrained=False)

        # Create RoboticsProcessor for all data processing (text, images, normalization)
        self.robotics_processor = RoboticsProcessor.from_pretrained(checkpoint_directory)

        # Load checkpoint (EMA or regular)
        if self.ema_enabled:
            load_ema_checkpoint(self.model, self.checkpoint_path)
        else:
            load_model_checkpoint(self.model, self.checkpoint_path)

        self.model.to(self.device)
        self.model.eval()
        # DiffusionPolicy uses CLIP instead of VLM, so no need for VLM-specific dtype setting

        # Get field mapping path
        self.field_mapping_path = os.path.join(os.path.dirname(__file__), "field_mapping.yaml")

        # Load camera names from dataset statistics
        self.image_names = self.cfg.data.image_names

        # Load configuration from checkpoint
        self.future_timesteps = self.cfg.data.lowdim_future_timesteps
        self.num_past_timesteps = self.cfg.data.lowdim_past_timesteps

        # Load image size from preprocessing config
        if checkpoint_directory.startswith("s3://"):
            preprocessing_config_path = f"{checkpoint_directory.rstrip('/')}/preprocessing_configs.yaml"
        else:
            preprocessing_config_path = os.path.join(checkpoint_directory, "preprocessing_configs.yaml")
        preprocessing_config = yaml_load(preprocessing_config_path)
        # Handle indexed format from collect_preprocessing_configs (e.g. {0: {...}, 1: {...}})
        if preprocessing_config and all(isinstance(k, int) for k in preprocessing_config):
            preprocessing_config = preprocessing_config[0]
        self.preprocessor_image_size = preprocessing_config.get("resize_images_size")
        # Validate that resize_images_size is configured
        if self.preprocessor_image_size is None:
            raise ValueError(
                f"resize_images_size not found in preprocessing config at {preprocessing_config_path}. "
                "Image resizing is required for inference. Please ensure the model was trained with "
                "resize_images_size configured in the preprocessing parameters."
            )
        self.preprocessor_image_resize_method = preprocessing_config.get(
            "preprocessor_image_resize_method", ImageResizingMethod.CENTER_CROP
        )
        total_timesteps = self.num_past_timesteps + 1 + self.future_timesteps
        logging.info(
            f"Timestep configuration: total={total_timesteps}, "
            f"past={self.num_past_timesteps}, current=1, future={self.future_timesteps}"
        )

        # Initialize data adapter with robotics processor, data config, and field mapping
        self.data_adapter: Dict[uuid.UUID, PolicyDataAdapter] = {}
        self.should_reset: Dict[uuid.UUID, bool] = {}

        # Initialize state
        self.reset()

        # Initialize visualizer (enabled via VISUALIZER=rerun env var)
        visualizer.init(run_name="inference_policy")
        # Log test message to verify rerun is working
        visualizer.log_text("status", "Inference policy initialized")

        logging.info(f"LBMDiffusionPolicy initialized with model on {self.device}")

    def reset(self):
        """Reset the policy state."""
        logging.debug("Resetting policy state")
        self._step_count = defaultdict(int)
        self.current_open_loop_step.clear()

    def get_policy_metadata(self):
        metadata = _get_policy_metadata()
        metadata.checkpoint_path = self.checkpoint_path
        return metadata

    def step(self, observation: MultiarmObservation, client_id: uuid.UUID) -> PosesAndGrippers:
        """Generate robot actions based on a single observation."""
        logging.debug(f"Stepping with {client_id}")
        # Step the data adapter to update its state with the new observation
        self.data_adapter[client_id].step_observations(observation)

        # Log observation to visualizer
        visualizer.log_robot_gym_multiarm_observation("observation", observation)

        # Recompute the trajectory if we are at the beginning of a new open loop step
        if self.current_open_loop_step[client_id] % self.open_loop_steps == self.open_loop_steps - 1:
            # Step the data adapter before getting the model input that needs to be updated for current step
            # Get the model input
            model_input = self.data_adapter[client_id].get_model_input(observation)
            # Move to device
            input_ids = model_input["input_ids"].to(self.device) if "input_ids" in model_input else None
            attention_mask = model_input["attention_mask"].to(self.device) if "attention_mask" in model_input else None
            pixel_values = model_input["pixel_values"].to(self.device) if "pixel_values" in model_input else None
            actions_tensor = model_input["actions"].to(self.device) if "actions" in model_input else None

            # Log processed images going into the model (every 10 steps)
            if pixel_values is not None:
                # pixel_values shape: [batch, num_cameras * timesteps, channels, height, width]
                pv = pixel_values.cpu().numpy()
                for i in range(pv.shape[1]):
                    # Convert from CHW to HWC and denormalize from [-1,1] or [0,1] to [0,255]
                    img = pv[0, i].transpose(1, 2, 0)  # CHW -> HWC
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype("uint8")
                    visualizer.log_images(f"model_input/image_{i}", img, every_n=10)
            proprioception = None
            if "proprioception" in model_input and model_input["proprioception"] is not None:
                proprioception = model_input["proprioception"].to(self.device)
            point_cloud = None
            if "point_cloud" in model_input and model_input["point_cloud"] is not None:
                point_cloud = model_input["point_cloud"].to(self.device)
            # Generate the next chunk of actions using the model
            autocast = get_autocast(self.cfg.hparams.precision)
            with torch.no_grad(), autocast():
                # Use the model's generate_actions method (DiffusionPolicy interface)
                model_output = self.model.generate_actions(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    actions=actions_tensor,
                    attention_mask=attention_mask,
                    num_inference_steps=self.num_flow_steps,
                    past_mask=model_input["past_mask"].to(self.device),
                    proprioception=proprioception,
                    point_cloud=point_cloud,
                )

                # The model outputs need to be interpreted in context to have denormalized absolute actions
                self.data_adapter[client_id].update_action(observation, model_output.clone().detach().cpu())

            # Reset the open loop step counter for this client
            self.current_open_loop_step[client_id] = 0

        actions = self.data_adapter[client_id].step_action()
        self.current_open_loop_step[client_id] += 1
        self._step_count[client_id] += 1

        # Log action to visualizer
        visualizer.log_robot_gym_poses_and_grippers("action", actions)

        return actions

    def step_batch(self, observations: Dict[uuid.UUID, MultiarmObservation]) -> Dict[uuid.UUID, PosesAndGrippers]:
        """Generate robot actions for a batch of observations.

        Args:
            observations: Dictionary mapping client UUIDs to MultiarmObservation objects

        Returns:
            Dictionary mapping client UUIDs to PosesAndGrippers actions
        """
        logging.debug(f"Stepping batch with {len(observations)} parallel observations")
        batch_actions = {}

        for client_id, observation in observations.items():
            # Initialize data adapter if client hasn't been seen before
            if client_id not in self.data_adapter:
                logging.debug(f"Initializing data adapter for new client {client_id}")
                self.data_adapter[client_id] = PolicyDataAdapter(
                    robotics_processor=self.robotics_processor,
                    data_config=self.cfg.data,
                    field_mapping_path=self.field_mapping_path,
                    image_names=self.image_names,
                    preprocessor_image_size=self.preprocessor_image_size,
                    preprocessor_image_resize_method=self.preprocessor_image_resize_method,
                    num_past_timesteps=self.num_past_timesteps,
                    num_future_timesteps=self.future_timesteps,
                    image_indices=self.cfg.data.image_indices,
                )
                self.should_reset[client_id] = True

            if self.should_reset[client_id]:
                logging.debug(f"Resetting data adapter for {client_id}")
                self.data_adapter[client_id].reset(initial_observation=observation)
                self.should_reset[client_id] = False

        # TODO: Jean instead of running each step, we should run the model on the batch of observations at once
        for client_id, observation in observations.items():
            logging.debug(f"Stepping with {client_id}")
            batch_actions[client_id] = self.step(observation, client_id)

        return batch_actions

    def reset_batch(self, clients: Dict[uuid.UUID, int]):
        """Reset the policy state for a batch of observations.

        Args:
            clients: Dictionary mapping client UUIDs to initial seeds
        """
        logging.debug(f"Resetting batch with {len(clients)} parallel batches")

        for uuid_value, _seed in clients.items():
            if uuid_value not in self.data_adapter:
                self.data_adapter[uuid_value] = PolicyDataAdapter(
                    robotics_processor=self.robotics_processor,
                    data_config=self.cfg.data,
                    field_mapping_path=self.field_mapping_path,
                    image_names=self.image_names,
                    preprocessor_image_size=self.preprocessor_image_size,
                    num_past_timesteps=self.num_past_timesteps,
                    num_future_timesteps=self.future_timesteps,
                    image_indices=self.cfg.data.image_indices,
                )
            self.should_reset[uuid_value] = True

        # Reset the policy counters
        self._step_count = defaultdict(int)
        self.current_open_loop_step.clear()  # Clear all client counters


def main():
    parser = argparse.ArgumentParser(description="LBM DiffusionPolicy Policy Server")
    LbmPolicyServerConfig.add_argparse_arguments(parser)

    # Add LBM-specific arguments
    parser.add_argument("--checkpoint_directory", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Name of checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda/cpu)")
    parser.add_argument("--num_flow_steps", type=int, default=10, help="Number of diffusion steps to use")
    parser.add_argument("--open_loop_steps", type=int, default=4, help="Number of open loop steps to use")

    args = parser.parse_args()

    # Setup logging - use DEBUG level if DEBUG environment variable is set, otherwise INFO
    log_level = logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.INFO
    setup_logging(log_file=None, level=log_level)

    # Create the policy (use_ema is loaded from config.yaml automatically)
    policy = InferenceDiffusionPolicy(
        checkpoint_directory=args.checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
        open_loop_steps=args.open_loop_steps,
    )

    # Create run name with date identifier
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PolicyDataAdapter_{date_str}"
    vz.init(run_name=run_name, add_rank_to_run=True)

    # Run the policy server
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
