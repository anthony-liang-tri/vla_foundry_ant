import gym_pusht  # noqa: F401 -- registers the gymnasium environment
import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

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
        """Match PassthroughProcessor training pipeline: HWC uint8 → CHW float32 [0,1], resized."""
        t = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)
        if t.max() > 1.5:
            t = t / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear"
        ).squeeze(0)
        return t

    def _device(self):
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_from_obs(self, obs):
        device = self._device()
        curr_image = [np.array(obs["pixels"])]

        if self.past_images is None:
            self.past_images = []
            for _ in range(self.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        all_images = self.past_images + curr_image

        pixel_values = torch.stack([self._preprocess_image(img) for img in all_images]).unsqueeze(0)

        num_timesteps = self.num_past_actions + self.num_future_actions + 1
        actions = torch.zeros(1, num_timesteps, self.action_dim, device=device)

        # Fill in past actions from buffer (normalized)
        if self.past_actions is not None and len(self.past_actions) > 0:
            past_act_tensor = torch.tensor(self.past_actions, dtype=torch.float32).unsqueeze(0)
            if self.processor.normalizer is not None:
                past_act_tensor = self.processor.normalizer.normalize_tensor(past_act_tensor, "action")
            n_past = min(past_act_tensor.shape[1], self.num_past_actions)
            actions[:, self.num_past_actions - n_past:self.num_past_actions, :] = past_act_tensor[:, -n_past:, :].to(device)

        # Build proprioception from agent_pos history
        proprio = None
        if hasattr(self, '_prev_agent_pos') and self._prev_agent_pos is not None:
            proprio_positions = [obs["agent_pos"]]
            if self.num_past_actions > 0:
                proprio_positions = [self._prev_agent_pos, *proprio_positions]
            proprio_raw = torch.tensor(np.stack(proprio_positions), dtype=torch.float32).unsqueeze(0)
            if self.processor.normalizer is not None:
                proprio = self.processor.normalizer.normalize_tensor(proprio_raw, "observation.state")
            else:
                proprio = proprio_raw
            proprio = proprio.to(device)

        # Training masks mark only strict past actions as preserved; the current
        # action slot belongs to future_mask and must be denoised.
        past_mask = torch.zeros(1, num_timesteps, dtype=torch.bool, device=device)
        past_mask[:, : self.num_past_actions] = True
        future_mask = torch.zeros_like(past_mask)
        future_mask[:, self.num_past_actions:] = True

        return {
            "input_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, 1, dtype=torch.long, device=device),
            "pixel_values": pixel_values.to(device, dtype=torch.float32),
            "actions": actions,
            "past_mask": past_mask,
            "future_mask": future_mask,
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
        frame = self.env.render()
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        coverage = getattr(self, "_coverage", 0.0)
        max_coverage = getattr(self, "_max_coverage", 0.0)
        text = f"coverage: {coverage:.3f}  max: {max_coverage:.3f}"
        x, y = 8, 8
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
        return np.array(image)

    def get_current_images(self):
        return [np.array(self.obs["pixels"])]

    def env_reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        self.info = info
        self.success = False
        self.done = False
        self.past_images = None
        self.past_actions = None
        self.past_mask = None
        self._prev_agent_pos = obs["agent_pos"].copy()
        self._step_count = 0
        self._coverage = float(info.get("coverage", 0.0))
        self._max_coverage = 0.0
        return obs

    def env_step(self, action):
        prev_agent_pos = self.obs["agent_pos"].copy()
        action = np.clip(action, 0.0, 512.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info
        self._prev_agent_pos = prev_agent_pos
        self._step_count += 1
        self.success = info.get("is_success", False)
        coverage = float(info.get("coverage", reward))
        self._coverage = coverage
        self._max_coverage = max(self._max_coverage, coverage)
        self.done = terminated or truncated or self._step_count >= self.horizon
        return obs

    def get_max_coverage(self):
        return self._max_coverage

    def env_close(self):
        self.env.close()

    def check_success(self):
        return self.success

    def check_finished(self):
        return self.done
