"""
Need to first install the RoboCasa repo and set up the environment variables to run this.
See https://github.com/robocasa/robocasa for more details.
"""

from robocasa.utils.env_utils import create_env

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner


class RoboCasaEvalRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)

    def load_env(self, env_name):
        self.env = create_env(
            env_name=env_name,
            render_onscreen=False,
            seed=0,  # set seed=None to run unseeded
        )

    def extract_from_obs(self, obs):
        image_left = obs["robot0_agentview_left_image"]
        image_right = obs["robot0_agentview_right_image"]
        image_in_hand = obs["robot0_eye_in_hand_image"]
        return {"images": [image_left, image_right, image_in_hand], "text": None}

    def get_obs_tensor(self, obs):
        obs_tensor, reward, done, info = obs
        return obs_tensor

    def get_current_image(self):
        return self.env.sim.render(height=512, width=768, camera_name="robot0_agentview_center")[::-1]

    def check_success(self):
        return self.env._check_success()
