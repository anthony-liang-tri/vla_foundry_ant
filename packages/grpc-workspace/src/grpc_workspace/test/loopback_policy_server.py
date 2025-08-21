#! /usr/bin/env python3

import argparse
import copy
import uuid

from grpc_workspace.lbm_policy_server import (
    LbmPolicyServerConfig,
    run_policy_server,
)
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import Policy, PolicyMetadata


class LoopbackPolicy(Policy):
    """A fixture loopback policy.

    This policy copies the robot's observed state (positions, velocities) and
    returns them as the next action's commanded state.
    """

    def __init__(self):
        # Used when this policy is called via the non-batch interface.
        self._internal_uuid = uuid.uuid4()

    def reset(self, seed: int | None, options=None) -> None:
        return self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(self, seeds: dict[uuid.UUID, int | None], options=None) -> None:
        pass

    def get_policy_metadata(self):
        return PolicyMetadata(
            name="LoopbackPolicy",
            skill_type="test_policy_skill",
            checkpoint_path="path_to_checkpoint",
            git_repo="Unknown",
            git_sha="Undefined",
            raw_policy_config={"key": "value"},
            is_language_conditioned=False,
            runtime_information={"one": "two"},
        )

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        # When called in non-batch mode we don't have an external UUID
        # associated with the observation, so use one that we've defined
        # ourselves.
        batch_actions = self.step_batch({self._internal_uuid: observation})
        return batch_actions[self._internal_uuid]

    def step_batch(self, observations: dict[uuid.UUID, MultiarmObservation]) -> dict[uuid.UUID, PosesAndGrippers]:
        batch_actions: dict[uuid.UUID, PosesAndGrippers] = {}
        for identifier, obs in observations.items():
            batch_actions[identifier] = copy.deepcopy(obs.robot.actual)
        return batch_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake-argument",
        dest="fake_argument",
        action="store_true",
        help="Unused argument for testing purposes only.",
    )
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()
    policy = LoopbackPolicy()
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
