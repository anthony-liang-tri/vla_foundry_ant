Launch a SageMaker training job from a config directory.

The argument is the config path, e.g. `/launch-sagemaker vla_foundry/tri/ablations/ablation_configs_v03_16`

Steps:
1. Verify the config path exists and list the YAML files in it.
2. Ask the user which ablation name to use (if not specified in the argument).
3. Confirm the SageMaker queue and instance type. Defaults: `SAGEMAKER_QUEUE=vla`, `SAGEMAKER_INSTANCE_TYPE=p5en`.
4. Run: `./vla_foundry/tri/sagemaker/launch_from_configs.sh <config_path> [--ablation <name>]`
5. Report the job submission output.

If no argument is provided, list available config directories under `vla_foundry/tri/ablations/` and ask the user to pick one.