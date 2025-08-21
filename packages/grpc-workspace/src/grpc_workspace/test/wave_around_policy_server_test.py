import copy
import unittest

import numpy as np
from grpc_workspace.test.grpc_policy_test_case import GrpcPolicyTestCase
from robot_gym.assert_equal_recursive import assert_equal_recursive
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import PolicyMetadata


class WaveAroundPolicyServerTest(GrpcPolicyTestCase):
    def server_respath(self) -> str:
        return "wave_around_policy_server.py"

    def filename(self) -> str:
        return __file__

    def policy_metadata_test(self, policy_metadata: PolicyMetadata):
        self.assertEqual("WaveAround", policy_metadata.name)
        self.assertEqual("Undefined", policy_metadata.skill_type)

    def action_test(
        self,
        observation: MultiarmObservation,
        action: PosesAndGrippers,
        i: int,
    ):
        grippers = copy.deepcopy(observation.robot.actual.grippers)
        poses = copy.deepcopy(observation.robot.actual.poses)
        offset = np.sin(i * 0.1 * np.array([0.02, 0.03, 0.05]))
        for robot_name in poses:
            observed_xyz = poses[robot_name].translation()
            # Note that ths only works because the observation is unchanging.
            poses[robot_name].set_translation(observed_xyz + offset)
        expected = PosesAndGrippers(poses=poses, grippers=grippers)
        assert_equal_recursive(self, expected, action)

    def test_policy_functions(self):
        self.run_test(num_steps=2)


if __name__ == "__main__":
    unittest.main()
