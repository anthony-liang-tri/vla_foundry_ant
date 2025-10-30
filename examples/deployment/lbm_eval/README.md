## LBM Evaluation Utilities

This directory provides lightweight scripts for evaluating LBM robotics policies
against a gRPC client. Each script is ready to run with `uv` so that you can
reuse the repository's managed environment.

### Contents
- `launch_wave_policy.sh` – launches a dummy gRPC policy server that waves the
  robot end-effectors in a simple sinusoidal pattern. Helpful for verifying the
  evaluation pipeline.

### Prerequisites
- Complete the project setup in the repository root (see main README for
  `uv sync --frozen` instructions).
- Provide any required credentials (e.g., AWS, WANDB, Hugging Face tokens) for
  accessing checkpoints or datasets referenced in your custom configuration.
- Ensure your shell is in the repository root before running the scripts.

### Running the Wave Policy Demo
The wave policy is a deterministic scripted policy that validates the gRPC
stack.

```bash
bash examples/deployment/lbm_eval/launch_wave_policy.sh
```

From the Anzu repo, in simulation, you can run the following command to launch the wave policy:
```bash
bazel run //intuitive/visuomotor:demonstrate -- \
--config_file `pwd`/lbm_eval/scenarios/3_cabot_breakfast/put_cup_in_center_of_table.yaml \ --scenario GrpcServerToSim \
--demonstration_indices 1000:1005 \
--t_max 10.0 \
--save_dir=/tmp/lbm/rollouts/ \
--summary_dir=/tmp/lbm/rollouts/
```

Key behavior:
- Starts a gRPC server defined in
  `packages/grpc-workspace/src/grpc_workspace/wave_around_policy_server.py`.
- Streams sinusoidal joint poses to connected clients until interrupted.

