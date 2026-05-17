"""
Need to first install the RoboSuite repo and set up the environment variables to run this.
See https://github.com/robosuite/robosuite for more details.
"""

import numpy as np
import robosuite as suite
import torch
from PIL import Image, ImageDraw
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from vla_foundry.data.processor import apply_chat_template
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
        self.past_states = None
        self.state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
        ]

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

    def _device(self):
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _uses_state_proprioception(self):
        return self.processor.data_params.proprioception_fields == ["state"]

    def _state_from_obs(self, obs):
        missing = [key for key in self.state_keys if key not in obs]
        if missing:
            raise KeyError(f"Robosuite obs is missing state keys required by training data: {missing}")
        return np.concatenate([np.asarray(obs[key], dtype=np.float32).ravel() for key in self.state_keys]).astype(
            np.float32
        )

    def _update_state_buffer(self, state):
        if self.past_states is None:
            self.past_states = np.array([state], dtype=np.float32)
        else:
            self.past_states = np.vstack([self.past_states, state])
        if self.past_states.shape[0] > self.num_past_actions:
            self.past_states = self.past_states[-self.num_past_actions :]

    def _build_proprioception(self, obs, device):
        if not self._uses_state_proprioception():
            return None

        current_state = self._state_from_obs(obs)
        states = np.zeros((self.num_past_actions + 1, current_state.shape[0]), dtype=np.float32)
        states[-1] = current_state
        if self.past_states is not None and len(self.past_states) > 0 and self.num_past_actions > 0:
            n_past = min(len(self.past_states), self.num_past_actions)
            states[self.num_past_actions - n_past : self.num_past_actions] = self.past_states[-n_past:]

        proprio = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        if self.processor.normalizer is not None:
            proprio = self.processor.normalizer.normalize_tensor(
                proprio, "state", anchor_timestep=self.num_past_actions
            )
        return proprio.to(device)

    def extract_from_obs(self, obs):
        device = self._device()
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

        all_images = self.past_images + curr_image
        text = apply_chat_template(self.processor.vlm_processor, len(all_images), self.instruction)
        processed = self.processor.vlm_processor(
            images=all_images,
            text=text,
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

        actions = torch.zeros(1, num_timesteps, self.action_dim, device=device)

        if self.past_actions is not None and len(self.past_actions) > 0:
            past_act_tensor = torch.tensor(self.past_actions, dtype=torch.float32).unsqueeze(0)
            if self.processor.normalizer is not None:
                past_act_tensor = self.processor.normalizer.normalize_tensor(past_act_tensor, "actions")
            n_past = min(past_act_tensor.shape[1], self.num_past_actions)
            actions[:, self.num_past_actions - n_past : self.num_past_actions, :] = past_act_tensor[
                :, -n_past:, :
            ].to(device)

        past_mask = torch.zeros(1, num_timesteps, dtype=torch.bool, device=device)
        valid_past = 0 if self.past_actions is None else min(len(self.past_actions), self.num_past_actions)
        if valid_past > 0:
            past_mask[:, self.num_past_actions - valid_past : self.num_past_actions] = True
        future_mask = torch.zeros_like(past_mask)
        future_mask[:, self.num_past_actions :] = True
        result = {
            "input_ids": processed["input_ids"].to(device),
            "attention_mask": processed["attention_mask"].to(device),
            "pixel_values": processed["pixel_values"].to(device),
            "actions": actions,
            "past_mask": past_mask,
            "future_mask": future_mask,
        }
        proprioception = self._build_proprioception(obs, device)
        if proprioception is not None:
            result["proprioception"] = proprioception
        return result

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
        image = Image.fromarray(arr[top : top + crop_h, left : left + crop_w])
        draw = ImageDraw.Draw(image)
        text = f"success: {int(self.success)}"
        x, y = 8, 8
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
        return np.array(image)

    def get_current_images(self):
        return self._flip_and_upscale([self.obs[image_name] for image_name in self.image_names])

    def env_reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
        self.env.reset()
        obs, reward, done, _ = self.env.step([0.0] * len(self.env.action_spec[0]))
        self.obs = obs
        self.success = reward > 0.5
        self.done = done
        self.past_images = None
        self.past_actions = None
        self.past_states = None
        self.past_mask = None
        return obs

    def env_step(self, action):
        if self._uses_state_proprioception():
            self._update_state_buffer(self._state_from_obs(self.obs))
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
