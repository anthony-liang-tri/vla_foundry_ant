import os

import numpy as np

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import get_latest_checkpoint, load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


class BaseEvalRunner:
    def __init__(self, eval_params):
        self.eval_params = eval_params
        self.processor = RoboticsProcessor.from_pretrained(eval_params.model_path)
        self.normalizer = self.processor.normalizer
        self.env = None
        self.model = None
        self.past_images = None
        self.past_actions = None
        self.past_mask = None
        self.instruction = None

        # Our model requires action inputs to be of shape (B, T, D) where T is the number of timesteps.
        # We use these parameters from data_params to properly construct the timestep dimension.
        self.num_past_actions = self.normalizer.lowdim_past_timesteps
        self.num_future_actions = self.normalizer.lowdim_future_timesteps
        self.num_past_image_timesteps = len(self.processor.data_params.image_indices) - 1

    def load_env(self):
        raise NotImplementedError("load_env method not implemented")

    def env_reset(self):
        obs = self.env.reset()
        self.obs = obs
        self.past_images = None
        self.past_actions = None
        self.past_mask = None
        return obs

    def env_step(self, action):
        obs = self.env.step(action)
        self.obs = obs
        return obs

    def env_close(self):
        self.env.close()

    def extract_from_obs(self, obs):
        return {"input_ids": None, "attention_mask": None, "pixel_values": None, "actions": None, "past_mask": None}

    def get_obs_tensor(self, obs):
        return obs

    def denormalize_actions(self, actions):
        raise NotImplementedError("denormalize_actions method not implemented")

    def update_action_buffer(self, action):
        # Updates the past_actions with the current action.
        # We need this to properly construct the input tensors for the model's next inference step.
        if self.past_actions is None:
            self.past_actions = np.array([action])
        else:
            self.past_actions = np.vstack([self.past_actions, action])
        if self.past_actions.shape[0] > self.num_past_actions:
            self.past_actions = self.past_actions[-self.num_past_actions :]

    def update_image_buffer(self, images):
        # Updates the past_images with the current images.
        # We need this to properly construct the input tensors for the model's next inference step.
        self.past_images = self.past_images + images
        num_past_images = len(images) * self.num_past_image_timesteps
        if len(self.past_images) > num_past_images:
            self.past_images = self.past_images[-num_past_images:]

    def get_image_for_video(self):
        raise NotImplementedError("get_image_for_video method not implemented")

    def get_current_images(self):
        raise NotImplementedError("get_current_images method not implemented")

    def check_success(self):
        raise NotImplementedError("check_success method not implemented")

    def load_model(self, model_path):
        cfg = load_experiment_params_from_yaml(
            os.path.join(model_path, "config.yaml"), localize_params=not model_path.startswith("s3://")
        )
        model = create_model(cfg.model)
        load_model_checkpoint(model, get_latest_checkpoint(model_path))
        model = model.to("cuda")
        self.model = model
        return model
