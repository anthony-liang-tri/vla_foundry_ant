import contextlib
import unittest

from robot_gym.policy import Policy


class ToyPolicy(Policy):
    """
    A very simple toy policy exercising the Policy interface.
    """

    def __init__(self):
        self._counter = None
        self._was_closed = False

    def reset(self):
        self._counter = 0

    def step(self, obs):
        assert self._counter is not None
        act = {
            "counter": self._counter,
            "obs": obs,
        }
        self._counter += 1
        return act

    def close(self):
        self._was_closed = True

    def was_closed(self):
        return self._was_closed


class Test(unittest.TestCase):
    def test_toy_policy(self):
        policy = ToyPolicy()
        self.assertFalse(policy.was_closed())
        with contextlib.closing(policy):
            policy.reset()
            obs = {"x": 1}
            act = policy.step(obs)
            self.assertEqual(act, {"counter": 0, "obs": {"x": 1}})
            obs = {"x": 10}
            act = policy.step(obs)
            self.assertEqual(act, {"counter": 1, "obs": {"x": 10}})
            act = policy.step(obs)
            self.assertEqual(act, {"counter": 2, "obs": {"x": 10}})

            policy.reset()
            obs = {"y": 0}
            act = policy.step(obs)
            self.assertEqual(act, {"counter": 0, "obs": {"y": 0}})
        self.assertTrue(policy.was_closed())


if __name__ == "__main__":
    unittest.main()
