Visualize a dataset using the project's visualization tools.

The argument is the dataset name or path, e.g. `/visualize red_bell_pepper`

Steps:
1. If no argument, list available datasets from `vla_foundry/tri/preprocessing_config.yaml` or recent S3 paths and ask the user to pick one.
2. Run: `VISUALIZER=rerun timeout 90 ./visualize_data.sh <dataset>` to launch the Rerun visualizer.
3. If the visualizer script is not available, fall back to `python visualize_data.py <dataset>`.
4. Report any errors and suggest fixes.