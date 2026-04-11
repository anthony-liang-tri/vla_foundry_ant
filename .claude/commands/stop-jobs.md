Stop/cancel SageMaker Batch jobs from a queue.

The argument specifies a job name pattern and optional queue, e.g.:
- `/stop-jobs bench-flex` — cancel all jobs matching "bench-flex" in the default vla queue
- `/stop-jobs killable --queue tri-cam-humanoid` — cancel "killable" jobs in tri-cam-humanoid queue
- `/stop-jobs bench-flex-vlm-1n-jeanmercat --queue vla` — cancel a specific job

## Tools

Uses `batchy` from `/media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/sagey/batchy`:
```bash
cd /media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/sagey/batchy
uv run batchy <command> --profile sagemaker --region us-west-2
```

## Queue name mapping

User-friendly names map to full AWS Batch queue names:
- `vla` → `fss-vla-p5-48xlarge-us-west-2`
- `tri-cam-humanoid` → `fss-tri-cam-humanoid-p5-48xlarge-us-west-2`
- `ml` → `fss-ml-p5-48xlarge-us-west-2`
- `testing` → `fss-testing-p5-48xlarge-us-west-2`

If the user provides a short name, expand it. If they provide the full queue name, use it as-is.

Default queue: `fss-vla-p5-48xlarge-us-west-2`

## Steps

### 1. List matching jobs

First, list all jobs in the queue to find matches:
```bash
cd /media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/sagey/batchy
uv run batchy ls <full-queue-name> --profile sagemaker --region us-west-2
```

Filter the output for jobs matching the user's pattern. Show the matching jobs and ask for confirmation before canceling.

If `batchy ls` is slow or returns empty, you can also check SageMaker directly for running jobs:
```bash
aws --region us-west-2 --profile sagemaker sagemaker list-training-jobs \
  --name-contains <pattern> --max-results 20 --no-cli-pager \
  --sort-by CreationTime --sort-order Descending \
  --query 'TrainingJobSummaries[].{Name:TrainingJobName, Status:TrainingJobStatus}'
```

### 2. Cancel jobs

Use `batchy cancel-job` with the **exact full job name** (not a short/partial name):
```bash
cd /media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/sagey/batchy
uv run batchy cancel-job <full-queue-name> --job-name "<exact-job-name>" --profile sagemaker --region us-west-2
```

**Important**: The `--job-name` must be the **exact full job name** as it appears in the queue (e.g., `bench-flex-vlm-1n-jeanmercat-vla-foundry-2026-03-29-18-22-16`), not a short prefix. The job name from `sagemaker/launch_training.py` output (the "Queued ..." line) is the correct name to use.

For jobs that are already running in SageMaker (not just queued in Batch), stop them via SageMaker:
```bash
aws --region us-west-2 --profile sagemaker sagemaker stop-training-job \
  --training-job-name "<AWSBatch-prefixed-name>"
```

### 3. Report results

Show which jobs were cancelled/stopped and which weren't found. If a job wasn't found by name, it may have:
- Already been picked up by SageMaker (check with `sagemaker list-training-jobs`)
- Already completed or failed
- Had a slightly different name (the launch script truncates to 63 chars)

## Common patterns

Cancel all jobs by a user with a prefix:
```bash
# List first to get exact names
cd /media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/sagey/batchy
uv run batchy ls fss-vla-p5-48xlarge-us-west-2 --profile sagemaker --region us-west-2 2>&1 | grep "bench-flex"

# Then cancel each one
uv run batchy cancel-job fss-vla-p5-48xlarge-us-west-2 --job-name "<exact-name>" --profile sagemaker --region us-west-2
```
