import subprocess
import unittest
from pathlib import Path

from grpc_workspace.lbm_policy_client import LbmPolicyClientConfig
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.multiarm_spaces_conversions import make_example_obs_and_act
from robot_gym.policy import PolicyMetadata

# The purpose of this list is to provide unique port numbers for all gRPC
# tests. The existence of this file shows that we need some form of network
# isolation for our tests in lbm.
# TODO: Port anzu's network isolation mechanism into lbm or
# find some other solution.
_FILES = [
    "diffusion_policy_server_test",
    "test_lbm_policy_rpc_integration",
    "vla_policy_server_test",
    "wave_around_policy_server_test",
]
# This is deliberately a different range from the default used in the configs
# so that we can run a policy server, and tests at the same time in most
# circumstances.
_BASE_PORT = 60051
_FILE_TO_PORT = {f: _BASE_PORT + i for i, f in enumerate(_FILES)}

# Ensure that the above list stays sorted, and that we have not done something
# absolutely absurd with the number of times we've used this mechanism before
# we fix it.
assert sorted(_FILES) == _FILES
assert _FILE_TO_PORT[_FILES[-1]] <= _BASE_PORT + 10


def grpc_server_uri(filename: str) -> str:
    return f"localhost:{_FILE_TO_PORT[filename]}"


class GrpcPolicyTestCase(unittest.TestCase):
    def setUp(self):
        server_bin = (Path(__file__).parent.parent / self.server_respath()).resolve()
        self._server_uri = grpc_server_uri(Path(self.filename()).stem)
        self._server = subprocess.Popen(
            [
                str(server_bin),
                f"--server-uri={self._server_uri}",
                *self.server_args(),
            ]
        )

    def server_respath(self) -> str:
        raise NotImplementedError()

    def filename(self) -> str:
        raise NotImplementedError()

    def server_args(self) -> list[str]:
        return []

    def policy_metadata_test(self, policy_metadata: PolicyMetadata):
        pass

    def reset_test(self):
        pass

    def action_test(
        self,
        observation: MultiarmObservation,
        action: PosesAndGrippers,
        i: int,
    ):
        pass

    def run_test(self, num_steps: int = 1):
        client_config = LbmPolicyClientConfig(wait_for_server=True, server_uri=self._server_uri)
        client = client_config.create()
        print("Testing get_policy_metadata() ...", flush=True)
        policy_metadata = client.get_policy_metadata()
        self.policy_metadata_test(policy_metadata)
        print("Testing reset() ...", flush=True)
        client.reset()
        self.reset_test()
        observation, _ = make_example_obs_and_act()
        for i in range(num_steps):
            print(f"Testing step() call {i} of {num_steps} ...", flush=True)
            action = client.step(observation)
            self.action_test(observation, action, i)

    def tearDown(self):
        # TODO: Find a reasonable way for this test to fail if the
        # subprocess cleanup doesn't properly complete,
        # e.g. using self.server.wait(forever) and get a bazel test timeout.
        self._server.terminate()
        self._server.wait(timeout=3.0)
        self._server.kill()
