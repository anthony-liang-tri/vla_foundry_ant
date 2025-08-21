import unittest

from grpc_workspace.lbm_policy_client import (
    LbmPolicyClient,
    LbmPolicyClientConfig,
)


class LbmPolicyClientTester(unittest.TestCase):
    def test_smoke_lbm_policy_client(self):
        """
        Create and veryify an instance of a LbmPolicyClient through its config.
        """
        dut = LbmPolicyClientConfig(wait_for_server=False)
        policy_client = dut.create()
        self.assertIsInstance(
            policy_client,
            LbmPolicyClient,
        )


if __name__ == "__main__":
    unittest.main()
