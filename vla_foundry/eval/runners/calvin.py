"""
Need to first install the CALVIN repo and set up the environment variables to run this.
See https://github.com/mees/calvin for more details.
"""

from pathlib import Path

import numpy as np
import torch

from vla_foundry.eval.runners.base_eval_runner import BaseEvalRunner


class CalvinRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)
        if eval_params.image_names is None or len(eval_params.image_names) == 0:
            self.image_names = ["rgb_static", "rgb_gripper"]
        self.action_dim = 7

    def load_env(self, env_name, task_name):
        import calvin_env
        from calvin_env.envs.play_table_env import get_env
        from omegaconf import OmegaConf

        # Get calvin_env base path - use same approach as calvin_env's own code
        # Config and data directories are in the parent directory of the package
        calvin_env_base_path = Path(calvin_env.__file__).parents[1]

        # Create dummy dataset path structure for get_env
        dataset_path = calvin_env_base_path / "data" / "dummy_dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        hydra_dir = dataset_path / ".hydra"
        hydra_dir.mkdir(exist_ok=True)

        # Load configs
        conf_path = calvin_env_base_path / "conf"
        static_camera_cfg = OmegaConf.load(conf_path / "cameras" / "cameras" / "static.yaml")
        gripper_camera_cfg = OmegaConf.load(conf_path / "cameras" / "cameras" / "gripper.yaml")
        scene_cfg = OmegaConf.load(conf_path / "scene" / "calvin_scene_D_eval.yaml")
        robot_cfg = OmegaConf.load(conf_path / "robot" / "panda.yaml")
        env_cfg = OmegaConf.load(conf_path / "env" / "play_table_env.yaml")

        # Set data_path
        data_path = str(calvin_env_base_path / "data")
        scene_cfg.data_path = data_path

        # Resolve robot config interpolations from scene config
        if hasattr(scene_cfg, "robot_base_position"):
            robot_cfg.base_position = scene_cfg.robot_base_position
        if hasattr(scene_cfg, "robot_base_orientation"):
            robot_cfg.base_orientation = scene_cfg.robot_base_orientation
        if hasattr(scene_cfg, "robot_initial_joint_positions"):
            robot_cfg.initial_joint_positions = scene_cfg.robot_initial_joint_positions

        # Resolve euler_obs interpolation
        if hasattr(robot_cfg, "euler_obs"):
            scene_cfg.euler_obs = robot_cfg.euler_obs

        # Remove movable objects to avoid missing URDF files
        if hasattr(scene_cfg, "objects") and hasattr(scene_cfg.objects, "movable_objects"):
            scene_cfg.objects.movable_objects = {}

        # Merge configs
        merged_config = OmegaConf.merge(
            {
                "data_path": data_path,
                "robot": robot_cfg,
                "scene": scene_cfg,
                "env": env_cfg,
                "cameras": {
                    "rgb_static": static_camera_cfg,
                    "rgb_gripper": gripper_camera_cfg,
                },
            }
        )

        # Update env config
        merged_config.env.robot_cfg = robot_cfg
        merged_config.env.scene_cfg = scene_cfg
        merged_config.env.show_gui = False
        merged_config.env.use_vr = False
        merged_config.env.use_scene_info = True
        merged_config.env.use_egl = True

        # Resolve interpolations
        try:
            OmegaConf.resolve(merged_config)
        except Exception:
            # If resolution fails, ensure euler_obs is set manually
            if hasattr(robot_cfg, "euler_obs") and not hasattr(scene_cfg, "euler_obs"):
                scene_cfg.euler_obs = robot_cfg.euler_obs

        OmegaConf.save(merged_config, hydra_dir / "merged_config.yaml")

        # Workaround for editable packages: set __file__ if missing
        if not hasattr(calvin_env, "__file__") or calvin_env.__file__ is None:
            calvin_env.__file__ = str(calvin_env_base_path / "__init__.py")

        # Initialize environment
        self.env = get_env(str(dataset_path), obs_space=None, show_gui=False)
        self.instruction = task_name

    def extract_from_obs(self, obs):
        # Handle tuple format from env_step: (obs, reward, done, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        # Calvin env returns obs with structure: {'rgb_obs': {'rgb_static': ..., 'rgb_gripper': ...}, ...}
        curr_image = [obs["rgb_obs"][image_name] for image_name in self.image_names]

        if self.past_images is None:
            self.past_images = []
            for _ in range(self.eval_params.num_past_image_timesteps):
                self.past_images.extend(curr_image)

        num_timesteps = self.eval_params.num_past_actions + self.eval_params.num_future_actions + 1
        if self.past_mask is None:
            self.past_mask = torch.zeros(1, num_timesteps).to("cuda")
            actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")
        else:
            self.past_mask[:, : self.eval_params.num_past_actions] = 1
            actions = torch.zeros(1, num_timesteps, self.action_dim).to("cuda")
            actions[:, : self.eval_params.num_past_actions, :] = torch.from_numpy(self.past_actions).to("cuda")

        if self.processor is not None:
            num_images = len(curr_image) * (self.eval_params.num_past_image_timesteps + 1)
            text = self.processor.apply_chat_template(num_images, self.instruction)

            processed = self.processor.vlm_processor(
                images=self.past_images + curr_image, text=text, padding=True, return_tensors="pt"
            )

            return {
                "input_ids": processed["input_ids"].to("cuda"),
                "attention_mask": processed["attention_mask"].to("cuda"),
                "pixel_values": processed["pixel_values"].unsqueeze(0).to("cuda"),
                "actions": actions,
                "past_mask": self.past_mask,
            }
        else:
            # Dummy values for RandomModel
            return {
                "input_ids": torch.zeros(1, 1).long().to("cuda"),
                "attention_mask": torch.ones(1, 1).to("cuda"),
                "pixel_values": torch.zeros(1, 3, 224, 224).to("cuda"),
                "actions": actions,
                "past_mask": self.past_mask,
            }

    def denormalize_actions(self, actions):
        if self.processor is not None:
            actions = self.processor.normalizer.denormalize_tensor(actions.squeeze(0).cpu(), "actions")
            actions_np = actions.numpy()
            # CALVIN expects gripper action (last dimension) to be exactly -1 or 1
            # Clip values close to -1 or 1 to the exact values
            actions_np[:, -1] = np.where(actions_np[:, -1] < 0, -1.0, 1.0)
            return actions_np
        else:
            # For RandomModel, actions are already in the right format
            if isinstance(actions, torch.Tensor):
                return actions.squeeze(0).cpu().numpy()
            return actions

    def get_obs_tensor(self, obs):
        # Handle tuple format from env_step: (obs, reward, done, info)
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def get_image_for_video(self):
        return self.obs["rgb_obs"]["rgb_static"]

    def get_current_images(self):
        return [self.obs["rgb_obs"]["rgb_static"], self.obs["rgb_obs"]["rgb_gripper"]]

    def check_success(self):
        # CALVIN env.step() returns (obs, reward, done, info) tuple
        # We store info in self.info from env_step
        if self.info is not None:
            return self.info.get("success", False)
        return False
