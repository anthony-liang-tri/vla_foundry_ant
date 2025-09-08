class BaseEvalRunner:
    def __init__(self, eval_params):
        self.eval_params = eval_params
        self.env = None
        self.model = None

    def load_env(self):
        raise NotImplementedError("load_env method not implemented")

    def env_reset(self):
        obs = self.env.reset()
        self.obs = obs
        return obs

    def env_step(self, action):
        obs = self.env.step(action)
        self.obs = obs
        return obs
    
    def env_close(self):
        self.env.close()

    def extract_from_obs(self, obs):
        return {'images': None, 'text': None}

    def get_obs_tensor(self, obs):
        return obs

    def get_current_image(self):
        raise NotImplementedError("get_current_image method not implemented")

    def check_success(self):
        raise NotImplementedError("check_success method not implemented")

    def load_model(self, model_path):
        class RandomModel:
            def __init__(self, env):
                self.env = env

            def get_action(self, images, text):
                # import numpy as np
                # action = np.random.uniform(low=self.env.action_spec[0], high=self.env.action_spec[1])
                # return action
                return [0.0] * 7

        self.model = RandomModel(self.env)