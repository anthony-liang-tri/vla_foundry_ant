import gym_pusht  # noqa: F401 -- registers the gymnasium environment
import gymnasium as gym
import numpy as np
import torch

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner

SUPPORTED_TASKS = ["PushT"]

TASK_INSTRUCTIONS = {
    "PushT": "push the T-shaped block to the target",
}


class PushTEvalRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)
        self.success = False
        self.action_dim = 2
        self.img_size = self.processor.data_params.image_size or 96

    def load_env(self, env_name, task_name, robot_name=None, horizon=300, render_onscreen=False):
        assert task_name in SUPPORTED_TASKS, f"Task {task_name} not supported."
        self.instruction = TASK_INSTRUCTIONS[task_name]
        self.horizon = horizon

        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Match PassthroughProcessor training pipeline: HWC uint8 → CHW float32 [0,255], resized."""
        t = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear"
        ).squeeze(0)
        return t

    def extract_from_obs(self, obs):
        curr_image = [np.array(obs["pixels"])]

        if self.past_images is None:
            self.past_images = []
            for _ in range(self.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        all_images = self.past_images + curr_image

        pixel_values = torch.stack([self._preprocess_image(img) for img in all_images]).unsqueeze(0)

        num_timesteps = self.num_past_actions + self.num_future_actions + 1
        actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")

        # Build proprioception from agent_pos history
        proprio = None
        if hasattr(self, '_prev_agent_pos') and self._prev_agent_pos is not None:
            proprio_raw = torch.tensor(
                np.stack([self._prev_agent_pos, obs["agent_pos"]]),
                dtype=torch.float32,
            ).unsqueeze(0)
            if self.processor.normalizer is not None:
                proprio = self.processor.normalizer.normalize_tensor(proprio_raw, "observation.state")
            else:
                proprio = proprio_raw
            proprio = proprio.to("cuda")

        # Build past_mask
        past_mask = torch.zeros(1, num_timesteps, dtype=torch.bool, device="cuda")
        past_mask[:, : self.num_past_actions + 1] = True

        return {
            "input_ids": torch.zeros(1, 1, dtype=torch.long, device="cuda"),
            "attention_mask": torch.ones(1, 1, dtype=torch.long, device="cuda"),
            "pixel_values": pixel_values.to("cuda", dtype=torch.float32),
            "actions": actions,
            "past_mask": past_mask,
            "proprioception": proprio,
        }

    def denormalize_actions(self, actions):
        if self.processor.normalizer is None:
            return torch.squeeze(actions).cpu().numpy()
        else:
            actions = self.processor.normalizer.denormalize_tensor(actions.squeeze(0).cpu(), "action")
            return actions.numpy()

    def get_obs_tensor(self, obs):
        return obs

    def get_image_for_video(self):
        return self.env.render()

    def get_current_images(self):
        return [np.array(self.obs["pixels"])]

    def env_reset(self):
        obs, info = self.env.reset()
        self.obs = obs
        self.info = info
        self.success = False
        self.done = False
        self.past_images = None
        self.past_actions = None
        self.past_mask = None
        self._prev_agent_pos = obs["agent_pos"].copy()
        self._step_count = 0
        return obs

    def env_step(self, action):
        action = np.clip(action, 0.0, 512.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info
        self._prev_agent_pos = obs["agent_pos"].copy()
        self._step_count += 1
        self.success = info.get("is_success", False)
        self.done = terminated or truncated or self._step_count >= self.horizon
        return obs

    def env_close(self):
        self.env.close()

    def check_success(self):
        return self.success

    def check_finished(self):
        return self.done
