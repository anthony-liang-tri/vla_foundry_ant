Monitor SageMaker training job status by name pattern.

The argument specifies a search pattern and optional action, e.g.:
- `/training-status qwen3 ft` — check all qwen3 fine-tuning jobs
- `/training-status sim-only-ft` — check sim-only fine-tuning jobs
- `/training-status` — check all recent training jobs by jmercat

## Steps

### 1. Find matching jobs

Search across all statuses (InProgress, Completed, Failed, Stopped) using the SageMaker list API. The `--name-contains` filter only supports `[a-zA-Z0-9\-]+` (no underscores), so use a broader query with client-side filtering:

```bash
for status in InProgress Completed Failed Stopped; do
  aws sagemaker list-training-jobs \
    --profile sagemaker --region us-west-2 \
    --max-results 100 \
    --sort-by CreationTime --sort-order Descending \
    --status-equals $status \
    --creation-time-after "$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)" \
    --output json 2>/dev/null | python3 -c "
import json, sys
jobs = [j for j in json.load(sys.stdin)['TrainingJobSummaries']
        if all(kw.lower() in j['TrainingJobName'].lower() for kw in '$KEYWORDS'.split())]
for j in jobs:
    print(f\"$status {j['TrainingJobName']}\")" 2>/dev/null
done
```

Where `$KEYWORDS` are the space-separated search terms from the argument.

### 2. If jobs are found from a previous launch, also check by direct describe

The list API has pagination limits and may miss older jobs. If you have known job names (from launch logs or prior checks), verify them directly:
```bash
aws sagemaker describe-training-job --profile sagemaker --region us-west-2 \
  --training-job-name "<job_name>" \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,FailureReason:FailureReason}' \
  --output json
```

### 3. Report summary

Present a concise table:
```
Status      | Count
------------|------
Completed   |  7
InProgress  |  3
Failed      |  0
Queued      |  6   (submitted but not yet visible in SageMaker)
------------|------
Total       | 16
```

For failed jobs, show the failure reason:
```bash
aws sagemaker describe-training-job --profile sagemaker --region us-west-2 \
  --training-job-name "<failed_job>" \
  --query 'FailureReason' --output text
```

### 4. Optional: Set up recurring monitoring

If the user wants continuous monitoring, set up a cron (every 15 minutes) that runs this same check and reports status changes. Use CronCreate with the monitoring prompt.

## Tips

- SageMaker job names from `launch_from_configs.sh` follow the pattern: `AWSBatch<ShortTaskName>-<ablation>-<hash>`
- Short task names are mapped in the launch script (e.g., `BimanualPutRedBellPepperInBin` -> `RedPepper`)
- The `--creation-time-after` filter helps narrow results to recent jobs
- For jobs that haven't started yet (queued in SageMaker), they won't appear in the list API until an instance is allocated
- The `SecondaryStatus` field shows detailed progress: `Starting`, `Downloading`, `Training`, `Uploading`, `Completed`
