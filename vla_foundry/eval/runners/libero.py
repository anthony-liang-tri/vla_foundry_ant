"""
Need to first install the LIBERO repo and set up the environment variables to run this.
See https://github.com/Lifelong-Robot-Learning/LIBERO for more details.
"""

import os

# We need this import to be able to import libero.
import libero_wrapper  # noqa: F401
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner


class LiberoRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)
        if eval_params.image_names is None or len(eval_params.image_names) == 0:
            self.image_names = ["agentview_image", "robot0_eye_in_hand_image"]
        self.action_dim = 7

    def load_env(self, env_name, task_id):
        # Here, we pass --task=(task_id) index instead of string
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[env_name]()
        task_id = int(task_id)

        # retrieve a specific task
        task = task_suite.get_task(task_id)
        self.instruction = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(
            f"[info] retrieving task {task_id} from suite {env_name}, the "
            + f"language instruction is {self.instruction}, and the bddl file is {task_bddl_file}"
        )

        # step over the environment
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 224, "camera_widths": 224}
        self.env = OffScreenRenderEnv(**env_args)
        init_states = task_suite.get_task_init_states(
            task_id
        )  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        self.env.set_init_state(init_states[init_state_id])

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

    def get_image_for_video(self):
        # Flip the image vertically since LIBERO images are upside down
        return np.flipud(self.obs[0]["agentview_image"])

    def get_current_images(self):
        return [self.obs[0]["agentview_image"], self.obs[0]["robot0_eye_in_hand_image"]]

    def check_success(self):
        return self.obs[2]  # (obs, reward, done, info)
