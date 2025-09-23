"""Eval wrapper around lbm_eval oss to test policies

This file is modified from the evaluate.py file
in the oss release of lbm_eval; in particular the
evaluate_one() method.

uv run lbm2/eval/run_eval.py --env lbm_eval --task pick_and_place_box

"""

import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# evaluate must be imported before the other anzu libraries
from lbm_eval.evaluate import _LastStepRecorder # noqa

from anzu.common.anzu_model_directives import MakeDefaultAnzuPackageMap
from anzu.intuitive.typing_ import from_dict
from anzu.intuitive.visuomotor.bases import (
    GymEnvWrappingAnzuEnv,
)
from anzu.intuitive.visuomotor.demonstration_seed import get_demonstration_seed
from anzu.intuitive.visuomotor.multiarm_simulations import (
    HardwareStationScenarioSimulationEnvConfig,
)

from pydrake.common.yaml import yaml_load
from robot_gym.multiarm_spaces import PosesAndGrippers

# lbm2 imports
from lbm2.eval.runners.base_eval_runner import BaseEvalRunner


@dataclass
class RolloutResults:
    total_time: float
    is_success: bool


class LBMEval(BaseEvalRunner):
    def __init__(self, eval_params):
        self.eval_params = eval_params
        self.env = None
        self.model = None
        self.scenario_index = 0

    def load_env(self, skill_type):
        # TODO(katliu): Remove these hardcodes; put in the right config
        use_eval_seed = True

        # Find the skill_type within our known packages. Packages that contain
        # skills must have a file named "skill_filenames.txt" at their root,
        # containing a newline-separated list of skill filename relative paths.
        package_map = MakeDefaultAnzuPackageMap()
        config_file = None
        for package_name in package_map.GetPackageNames():
            base = Path(package_map.GetPath(package_name))
            inventory = base / "skill_filenames.txt"
            if not inventory.exists():
                continue
            for line in inventory.read_text(encoding="utf-8").split():
                if line.endswith(f"/{skill_type}.yaml"):
                    config_file = base / line
                    break
        if not config_file:
            raise RuntimeError(f"Unknown {skill_type=}")

        # Materialize the visuomotor scenario environment.
        raw_config_all = yaml_load(filename=config_file, private=True)
        scenario_config_raw = raw_config_all["DiffusionInProcessSim"]
        env_config_raw = scenario_config_raw["env"]
        simulation_config_raw = env_config_raw["simulation_scenario_config"]
        simulation_config_raw["num_sample_processes"] = 1
        random_seed = get_demonstration_seed(self.scenario_index, use_eval_seed)
        simulation_config_raw["random_seed"] = random_seed
        env_config = from_dict(HardwareStationScenarioSimulationEnvConfig, env_config_raw)
        env_config.simulation_scenario_package = "anzu.sim.station.open_source"

        anzu_env = env_config.create()
        self.recorder = _LastStepRecorder()
        self.env = GymEnvWrappingAnzuEnv(anzu_env, self.recorder)

    def env_reset(self):
        # TODO(katliu): some of this logic probably shouldn't live here
        self.env.stop_episode()
        self.scenario_index += 1
        random_seed = get_demonstration_seed(self.scenario_index, True)
        options = {
            "demonstration_index": self.scenario_index,
            "save_dir": str(self.scenario_index),
        }
        print(random_seed)
        obs = self.env.reset(seed=random_seed, options=options)
        self.obs = obs
        return obs

    def env_step(self, action):
        obs = self.env.step(action)
        # TODO(katliu): Not 100% we need this render step
        self.env.render()
        self.obs = obs
        return obs

    def env_close(self):
        self.env.close()

    def extract_from_obs(self, obs):
        # in this case, we'll pass the obs straight through
        return obs.obs

    def get_obs_tensor(self, obs):
        return obs

    def get_current_image(self):
        return self.obs.obs.visuo["scene_left_0"].rgb.array

    def check_success(self):
        results = self.gather_stats()
        return results.is_success

    def gather_stats(self):
        # return more informative information on the roll out
        total_time = self.recorder.last_time_step.info["time"]
        is_success = self.recorder.last_time_step.info["is_success"]
        return RolloutResults(total_time, is_success)

    def load_model(self, model_path):
        class DanceBot:
            def __init__(self, env):
                self.env = env
                self._initial_poses = None
                self._initial_grippers = None
                self._counter = 0

            def get_action(self, obs) -> PosesAndGrippers:
                if self._initial_poses is None:
                    self._initial_poses = copy.deepcopy(obs.robot.actual.poses)
                    self._initial_grippers = copy.deepcopy(obs.robot.actual.grippers)

                grippers = copy.deepcopy(self._initial_grippers)
                poses = copy.deepcopy(self._initial_poses)
                offset = (
                    0.2
                    * np.sin(self._counter * 2 * np.pi / 50 + np.array([0.0, np.pi / 2, 0.0]))
                    * np.array([1.0, 1.0, 0.0])
                )
                offset_grippers = 0.05 * np.sin(self._counter)
                for robot_name, _pose in self._initial_poses.items():
                    observed_xyz = self._initial_poses[robot_name].translation()
                    poses[robot_name].set_translation(observed_xyz + offset)
                for gripper_name in self._initial_grippers:
                    grippers[gripper_name] = offset_grippers
                self._counter += 1
                return PosesAndGrippers(poses=poses, grippers=grippers)

        self.model = DanceBot(self.env)
