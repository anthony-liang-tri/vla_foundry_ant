Gather and compare evaluation results from S3.

The argument specifies what to gather, e.g.:
- `/gather-results` — gather results for the latest eval of the default checkpoint
- `/gather-results s3://path/to/checkpoint/` — gather all evals under a checkpoint
- `/gather-results compare normal ds` — compare two eval runs side by side

## How results are organized

Results live on S3 under checkpoint paths:
```
s3://.../evaluation/<eval_id>/<TaskName>/rollouts/demonstration_*/summary.yaml
s3://.../evaluation/dist_shift/<eval_id>/<TaskName>/rollouts/demonstration_*/summary.yaml
```

The gather script is at `vla_foundry/tri/utils/gather_results.py` and accepts S3 paths directly.

## Steps

1. **Identify what to gather.** Check the argument:
   - If an S3 path is given, use it directly
   - If "compare" is specified, gather multiple runs and produce a comparison table
   - If no argument, look at `results/` subdirectories for recent campaign_state.json files to find eval IDs and S3 paths

2. **Run the gather script** for each eval run:
   ```
   AWS_PROFILE=sagemaker uv run python vla_foundry/tri/utils/gather_results.py "<s3_path>" --no-interactive --no-cache
   ```

3. **For comparisons**, produce an unweighted mean table:
   - List all tasks with per-task success rates for each run
   - Compute deltas between runs
   - Use **unweighted mean** across tasks (each task counts equally regardless of trial count)
   - Flag tasks with large degradation (>10pp drop)

4. **Check for missing data:**
   - Some tasks may upload results to different S3 prefixes (e.g., without eval_id). Check parent directories.
   - If a task has results in one run but not another, flag it and try to find the data.
   - Verify distribution_shift_info in summary.yaml files to confirm DS was actually applied.

5. **Report** the results clearly with a markdown table.

## Important notes
- Use `AWS_PROFILE=sagemaker` for S3 access to model checkpoints
- The default checkpoint path is: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/qwen3_5_2b/2026_03_24-01_40_07-model_diffusion_policy-lr_5e-05-bsz_1024/`
- Distribution shift results are under `evaluation/dist_shift/`
- Normal results are under `evaluation/`
- Always verify DS was applied by checking `distribution_shift_info` in a summary.yaml
