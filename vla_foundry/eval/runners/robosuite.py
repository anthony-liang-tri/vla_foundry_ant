"""
Need to first install the RoboSuite repo and set up the environment variables to run this.
See https://github.com/robosuite/robosuite for more details.
"""

import numpy as np
import robosuite as suite
import torch
from PIL import Image
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner

SUPPORTED_TASKS = ["Lift", "NutAssemblySquare", "PickPlaceCan"]

TASK_INSTRUCTIONS = {
    "Lift": "lift the red cube off the table",
    "PickPlaceCan": "pick the coke can and place it in the bin",
    "NutAssemblySquare": "pick up the square nut and place it on the square peg",
}


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

    def load_env(self, env_name, task_name, robot_name="Panda", horizon=150, render_onscreen=False):
        self.render_onscreen = render_onscreen
        assert task_name in SUPPORTED_TASKS, f"Task {task_name} not supported."
        self.instruction = TASK_INSTRUCTIONS[task_name]

        # Load the desired controller
        controller_config = load_controller_config(default_controller="OSC_POSE")

        # Render at 84x84 to match training data (originally robomimic 84x84, upscaled to 256x256).
        self.render_height = 84
        self.render_width = 84
        self.upscale_size = (256, 256)

        # Create the environment instance.
        self.env = suite.make(
            env_name=task_name,
            robots=robot_name,
            initialization_noise=None,
            has_renderer=render_onscreen,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=self.render_height,
            camera_widths=self.render_width,
            controller_configs=controller_config,
            horizon=horizon,
        )

        if task_name == "Lift":
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

    def _flip_and_upscale(self, images):
        result = []
        for img in images:
            flipped = np.flip(img, axis=0)
            pil = Image.fromarray(flipped, mode="RGB")
            pil = pil.resize(self.upscale_size, Image.Resampling.BICUBIC)
            arr = np.array(pil)
            # Center crop 256x256 -> 224x224 to match training augmentation pipeline
            h, w = arr.shape[:2]
            crop_h, crop_w = 224, 224
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            arr = arr[top : top + crop_h, left : left + crop_w]
            result.append(arr)
        return result

    def extract_from_obs(self, obs):
        curr_image = self._flip_and_upscale([obs[image_name] for image_name in self.image_names])

        if not hasattr(self, "_logged_shapes"):
            self._logged_shapes = True
            raw_obs_img = obs[self.image_names[0]]
            print(f"[EVAL DEBUG] Raw obs image: shape={raw_obs_img.shape}, dtype={raw_obs_img.dtype}, "
                  f"min={raw_obs_img.min()}, max={raw_obs_img.max()}")
            print(f"[EVAL DEBUG] After flip+upscale+crop: shape={curr_image[0].shape}, "
                  f"dtype={curr_image[0].dtype}, min={curr_image[0].min()}, max={curr_image[0].max()}")

        if self.past_images is None:
            self.past_images = []
            for _ in range(self.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        expected_past_len = len(self.image_names) * self.num_past_image_timesteps
        assert self.past_images is None or len(self.past_images) == expected_past_len, (
            f"Mismatch in past images length. {len(self.past_images)} vs. {expected_past_len}"
        )

        processed = self.processor.vlm_processor(
            images=self.past_images + curr_image,
            text=self.instruction,
            padding=True,
            return_tensors="pt",
            **self.processor.processor_kwargs,
        )

        if processed["pixel_values"] is not None and processed["pixel_values"].ndim == 4:
            processed["pixel_values"] = processed["pixel_values"].unsqueeze(0)
        assert processed["pixel_values"].shape[0] == 1 and processed["pixel_values"].shape[1] == len(
            self.image_names
        ) * (self.num_past_image_timesteps + 1), f"Unexpected pixel_values shape: {processed['pixel_values'].shape}"

        if not hasattr(self, "_logged_processed"):
            self._logged_processed = True
            pv = processed["pixel_values"]
            print(f"[EVAL DEBUG] Processed pixel_values: shape={pv.shape}, dtype={pv.dtype}, "
                  f"min={pv.min():.4f}, max={pv.max():.4f}")
            print(f"[EVAL DEBUG] input_ids: shape={processed['input_ids'].shape}, dtype={processed['input_ids'].dtype}")

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
        flipped = np.flip(self.obs["agentview_image"], axis=0)
        pil = Image.fromarray(flipped, mode="RGB")
        pil = pil.resize(self.upscale_size, Image.Resampling.BICUBIC)
        arr = np.array(pil)
        h, w = arr.shape[:2]
        crop_h, crop_w = 224, 224
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return arr[top : top + crop_h, left : left + crop_w]

    def get_current_images(self):
        return self._flip_and_upscale([self.obs[image_name] for image_name in self.image_names])

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
