"""
Need to first install the RoboSuite repo and set up the environment variables to run this.
See https://github.com/robosuite/robosuite for more details.
"""

import robosuite as suite
import torch
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner

SUPPORTED_TASKS = ["Lift", "NutAssemblySquare", "PickPlaceCan"]


class RoboSuiteEvalRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)
        self.render_onscreen = False
        self.success = False
        if eval_params.image_names is None or not eval_params.image_names:
            self.image_names = ["agentview_image", "robot0_eye_in_hand_image"]
        else:
            self.image_names = eval_params.image_names
        self.action_dim = 7

    def load_env(self, env_name, task_name, robot_name="UR5e", horizon=150, render_onscreen=False):
        self.render_onscreen = render_onscreen
        assert task_name in SUPPORTED_TASKS, f"Task {task_name} not supported."

        # Load the desired controller
        controller_config = load_controller_config(default_controller="OSC_POSE")

        # Create the environment instance.
        self.env = suite.make(
            env_name=task_name,
            robots=robot_name,
            initialization_noise=None,
            has_renderer=render_onscreen,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            controller_configs=controller_config,
            horizon=horizon,
        )

        # Set the randomization for the cube location.
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.env.cube,
            x_range=[-0.2, 0.2],
            y_range=[0.04, 0.2],
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.env.table_offset,
            z_offset=0.01,
        )
        self.env.placement_initializer = placement_initializer

    def extract_from_obs(self, obs):
        curr_image = [obs[image_name] for image_name in self.image_names]
        if self.past_images is None:
            # At the start, just repeat the current image for the past image timesteps
            self.past_images = []
            for _ in range(self.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        expected_past_len = len(self.image_names) * self.num_past_image_timesteps
        assert self.past_images is None or len(self.past_images) == expected_past_len, (
            f"Mismatch in past images length. {len(self.past_images)} vs. {expected_past_len}"
        )

        processed = self.processor.vlm_processor(
            images=self.past_images + curr_image, text="", padding=True, return_tensors="pt"
        )

        if processed["pixel_values"] is not None and processed["pixel_values"].ndim == 4:
            processed["pixel_values"] = processed["pixel_values"].unsqueeze(0)
        assert processed["pixel_values"].shape[0] == 1 and processed["pixel_values"].shape[1] == len(
            self.image_names
        ) * (self.num_past_image_timesteps + 1), f"Unexpected pixel_values shape: {processed['pixel_values'].shape}"

        num_timesteps = self.num_past_actions + self.num_future_actions + 1

        assert self.past_mask is None, "Past mask is not currently supported with Robosuite."
        actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")

        return {
            "input_ids": processed["input_ids"].to("cuda"),
            "attention_mask": processed["attention_mask"].to("cuda"),
            "pixel_values": processed["pixel_values"].to("cuda"),
            "actions": actions,
            "past_mask": self.past_mask,
        }

    def denormalize_actions(self, actions):
        if self.processor.normalizer is None:
            return torch.squeeze(actions).cpu().numpy()
        else:
            actions = self.processor.normalizer.denormalize_tensor(actions.squeeze(0).cpu(), "actions")
            return actions.numpy()

    def get_obs_tensor(self, obs):
        return obs

    def get_image_for_video(self):
        return self.obs["agentview_image"]

    def get_current_images(self):
        return [self.obs[image_name] for image_name in self.image_names]

    def env_reset(self):
        self.env.reset()
        obs, reward, done, _ = self.env.step([0.0] * len(self.env.action_spec[0]))
        self.obs = obs
        self.success = reward > 0.5
        return obs

    def env_step(self, action):
        if action[6] < 0.0:
            action[6] = 0.0  # Special-case gripper.
        obs, reward, done, _ = self.env.step(action)
        self.obs = obs
        self.success = reward > 0.5
        self.done = done
        if self.render_onscreen:
            self.env.render()
        return obs

    def env_close(self):
        self.env.close()

    def check_success(self):
        return self.success

    def check_finished(self):
        return self.done
