import unittest

from grpc_workspace.test.grpc_policy_test_case import GrpcPolicyTestCase
from robot_gym.assert_equal_recursive import assert_equal_recursive
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import PolicyMetadata


# TODO: Isolate the gRPC traffic either by controlling
# the server URI and setting it to an unused port, or through
# the container isolation mechanisms we leverage with Anzu's ROS
# testing.
class LbmPolicyRpcIntegration(GrpcPolicyTestCase):
    def server_respath(self) -> str:
        return "test/loopback_policy_server.py"

    def filename(self) -> str:
        return __file__

    def policy_metadata_test(self, policy_metadata: PolicyMetadata):
        self.assertEqual("LoopbackPolicy", policy_metadata.name)
        self.assertEqual("test_policy_skill", policy_metadata.skill_type)

    def action_test(
        self,
        observation: MultiarmObservation,
        action: PosesAndGrippers,
        i: int,
    ):
        assert_equal_recursive(self, observation.robot.actual, action)

    def test_policy_functions(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
