Launch or monitor a data preprocessing pipeline.

The argument specifies the action, e.g. `/preprocess launch`, `/preprocess status`, `/preprocess <dataset_name>`

Steps:
1. Read `vla_foundry/tri/preprocessing_config.yaml` to understand the current preprocessing configuration.
2. Based on the argument:
   - **launch** or **<dataset_name>**: Submit a Ray job to run `vla_foundry/tri/data_generation_ray.py` with the appropriate config. Confirm parameters with the user before launching.
   - **status**: Check running preprocessing jobs via `ray job list` and show progress.
   - **no argument**: Show the current preprocessing config and ask what the user wants to do.
3. Report the job ID and how to monitor it.