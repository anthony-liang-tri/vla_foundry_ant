import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lbm2.eval.runners.base_eval_runner import BaseEvalRunner


class LiberoRunner(BaseEvalRunner):
    def __init__(self, eval_params):
        super().__init__(eval_params)

    def load_env(self, env_name):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[env_name]()

        # retrieve a specific task
        task_id = 0
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(
            f"[info] retrieving task {task_id} from suite {env_name}, the "
            + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
        )

        # step over the environment
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
        self.env = OffScreenRenderEnv(**env_args)
        init_states = task_suite.get_task_init_states(task_id)  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        self.env.set_init_state(init_states[init_state_id])


    def extract_from_obs(self, obs):
        image_agentview = obs['agentview_image']
        image_eyeinhand = obs['robot0_eye_in_hand_image']
        return {'images': [image_agentview, image_eyeinhand], 'text': None}

    def get_obs_tensor(self, obs):
        obs_tensor, reward, done, info = obs
        return obs_tensor

    def get_current_image(self):
        return self.obs[0]['agentview_image']

    def check_success(self):
        return self.obs[2]      # (obs, reward, done, info)