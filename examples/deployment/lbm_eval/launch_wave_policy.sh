#!/usr/bin/env bash

# This script runs a scripted policy that waves around the robot.

# Here's an example command for launching an lbm eval scenario locally (from Anzu):
# bazel run //intuitive/visuomotor:demonstrate -- \
# --config_file `pwd`/intuitive/visuomotor/config/put_cup_in_center_of_table_cabot.yaml \
# --scenario GrpcServerToSim \
# --demonstration_indices 1000:1005 \
# --t_max 10.0 \
# --save_dir=/tmp/lbm/rollouts/ \
# --summary_dir=/tmp/lbm/rollouts/

uv run --group inference python packages/grpc-workspace/src/grpc_workspace/wave_around_policy_server.py
