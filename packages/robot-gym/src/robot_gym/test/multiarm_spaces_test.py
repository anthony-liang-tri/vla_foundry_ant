import unittest

import numpy as np
from robot_gym.assert_equal_recursive import assert_equal_recursive
from robot_gym.multiarm_spaces import RestorePosesAndGrippersConfig
from robot_gym.multiarm_spaces_conversions import (
    dp_vector_to_multiarm_action,
    make_example_obs_and_act,
    make_example_obs_and_obs_dict,
    matrix_to_rotation_6d,
    multiarm_action_to_dp_vector,
    multiarm_observation_to_dp_dict,
    rotation_6d_to_matrix,
    rpy_deg,
)


class Test(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None  # Ensure we get good assertion errors here.

    def test_rotation_6d(self):
        R = rpy_deg([15, 30, 45]).matrix()
        d6 = matrix_to_rotation_6d(R)
        self.assertEqual(d6.shape, (6,))
        R_again = rotation_6d_to_matrix(d6)
        np.testing.assert_allclose(R, R_again, atol=1e-15, rtol=0)

    def test_rotation_6d_orthonormality_regression_test(self):
        # This caused orthonormality errors (#13969).
        d6 = np.array(
            [
                0.6298105716705322,
                0.035225264728069305,
                -0.7759491205215454,
                0.22841385006904602,
                -0.9632018208503723,
                0.14166943728923798,
            ],
            dtype=np.float32,
        )
        R = rotation_6d_to_matrix(d6)
        # This is the tolerance used by Drake's
        # RotationMatrix::kInternalToleranceForOrthonormality.
        tolerance = 128 * np.finfo(np.float64).eps
        orthonormality_measure = np.max(np.abs(R @ R.T - np.eye(3)))
        self.assertLess(orthonormality_measure, tolerance)

    def test_observation_to_dp_dict(self):
        obs, obs_dict_expected = make_example_obs_and_obs_dict()
        obs_dict = multiarm_observation_to_dp_dict(obs)
        assert_equal_recursive(self, obs_dict, obs_dict_expected, np_tol=1e-8)

    def test_action_to_and_from_dp_vector(self):
        _, act = make_example_obs_and_act()
        vector = multiarm_action_to_dp_vector(act)
        config = RestorePosesAndGrippersConfig.make_default()
        act_again = dp_vector_to_multiarm_action(config, vector)
        # N.B. `DeepDiff` does not seem to respect more precise tolerances;
        # DeepDiff(*, significant_digits=16, number_format_notation="e") does
        # not trigger an expected failure.
        act.debugging_output = None
        assert_equal_recursive(self, act, act_again, np_tol=2e-16)


if __name__ == "__main__":
    unittest.main()
