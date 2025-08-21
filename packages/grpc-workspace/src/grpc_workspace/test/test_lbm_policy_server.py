import unittest
from collections import OrderedDict

from grpc_workspace.lbm_policy_server import (
    LbmPolicyServer,
    LbmPolicyServerConfig,
)
from robot_gym.policy import Policy


class LbmPolicyServerTester(unittest.TestCase):
    def test_smoke_lbm_policy_server(self):
        """
        Create and veryify an instance of a LbmPolicyServer.
        """
        policy = Policy()
        dut = LbmPolicyServerConfig().create(policy=policy)
        self.assertIsInstance(dut, LbmPolicyServer)

    def test_get_and_update_step_inputs(self):
        """
        Tests the private _get_and_update_step_inputs() function
        which stores policy observations in an OrderedDict in order
        to ensure they are serviced in the order they are received if
        depending on the number of observations vs the batch_max_size.
        """
        policy = Policy()
        # The policy doesn't matter here, just batch_max_size of 2.
        dut = LbmPolicyServerConfig(batch_max_size=2).create(policy=policy)
        # Check the case when len(_step_mailbox_inputs) > _batch_max_size.
        # Populate the dictionary with three entries. The insertion order
        # matters due to _step_mailbox_inputs being an OrderedDict.
        dut._step_mailbox_inputs["a"] = 1
        dut._step_mailbox_inputs["b"] = 2
        dut._step_mailbox_inputs["c"] = 3
        result = dut._get_and_update_step_inputs()
        self.assertEqual(result, {"a": 1, "b": 2})
        self.assertEqual(dut._step_mailbox_inputs, OrderedDict(c=3))

        # Check the case when len(_step_mailbox_inputs) < _batch_max_size.
        # Retrieve the remaining element, and check that the
        # _step_mailbox_inputs dictionary is now empty.
        result = dut._get_and_update_step_inputs()
        self.assertEqual(result, {"c": 3})
        self.assertEqual(dut._step_mailbox_inputs, OrderedDict())

        # Check the case when len(_step_mailbox_inputs) == _batch_max_size.
        dut._step_mailbox_inputs["d"] = 4
        dut._step_mailbox_inputs["e"] = 5
        result = dut._get_and_update_step_inputs()
        self.assertEqual(result, {"d": 4, "e": 5})
        self.assertEqual(dut._step_mailbox_inputs, OrderedDict())


if __name__ == "__main__":
    unittest.main()
