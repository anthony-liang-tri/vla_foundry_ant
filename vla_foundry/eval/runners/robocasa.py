"""
Need to first install the RoboCasa repo and set up the environment variables to run this.
See https://github.com/robocasa/robocasa for more details.
"""

import torch
from robocasa.utils.env_utils import create_env

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner


class RoboCasaEvalRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)
        if eval_params.image_names is None or len(eval_params.image_names) == 0:
            self.image_names = [
                "robot0_agentview_left_image",
                "robot0_agentview_right_image",
                "robot0_eye_in_hand_image",
            ]
        self.action_dim = 12

    def load_env(self, env_name, task_name):
        self.env = create_env(
            env_name=task_name,
            render_onscreen=False,
            seed=0,  # set seed=None to run unseeded
        )
        self.instruction = self.env.get_ep_meta()["lang"]

    def extract_from_obs(self, obs):
        curr_image = [obs[image_name] for image_name in self.image_names]
        if self.past_images is None:
            # At the start, just repeat the current image for the past image timesteps
            self.past_images = []
            for _ in range(self.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        num_images = len(curr_image) * (self.num_past_image_timesteps + 1)
        text = self.processor.apply_chat_template(num_images, self.instruction)

        processed = self.processor.vlm_processor(
            images=self.past_images + curr_image, text=text, padding=True, return_tensors="pt"
        )

        num_timesteps = self.num_past_actions + self.num_future_actions + 1
        if self.past_mask is None:
            self.past_mask = torch.zeros(1, num_timesteps).to("cuda")
            actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")
        else:
            self.past_mask[:, : self.num_past_actions] = 1
            actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")
            actions[:, : self.num_past_actions, :] = torch.from_numpy(self.past_actions).to("cuda")

        return {
            "input_ids": processed["input_ids"].to("cuda"),
            "attention_mask": processed["attention_mask"].to("cuda"),
            "pixel_values": processed["pixel_values"].unsqueeze(0).to("cuda"),
            "actions": actions,
            "past_mask": self.past_mask,
        }

    def denormalize_actions(self, actions):
        actions = self.normalizer.denormalize_tensor(actions.squeeze(0).cpu(), "actions")
        return actions.numpy()

    def get_obs_tensor(self, obs):
        obs_tensor, reward, done, info = obs
        return obs_tensor

    def get_current_images(self):
        left = self.env.sim.render(height=224, width=224, camera_name="robot0_agentview_left")[::-1]
        right = self.env.sim.render(height=224, width=224, camera_name="robot0_agentview_right")[::-1]
        in_hand = self.env.sim.render(height=224, width=224, camera_name="robot0_eye_in_hand")[::-1]
        return [left, right, in_hand]

    def get_image_for_video(self):
        return self.env.sim.render(height=512, width=768, camera_name="robot0_agentview_center")[::-1]

    def check_success(self):
        return self.env._check_success()
