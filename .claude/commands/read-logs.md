Read SageMaker training job logs and diagnose failures.

The argument specifies a job name (full or partial), e.g.:
- `/read-logs AWSBatchreal-only-wd001-jmercatfd4825...` — full job name
- `/read-logs wd001` — partial match (finds most recent matching job)

## Steps

### 1. Find the job

If a full job name is given, use it directly. Otherwise, search for matching jobs:

```bash
for status in Failed InProgress Completed Stopped; do
  aws sagemaker list-training-jobs \
    --profile sagemaker --region us-west-2 \
    --max-results 20 \
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

Present the matches and pick the most relevant one (prefer Failed > InProgress > others).

### 2. Download logs with sagey

Use `sagey` to download CloudWatch logs for all nodes:

```bash
sagey logs <job_name>
```

This writes one `.log` file per node (algo-1 through algo-N) in the current directory.

### 3. Diagnose the issue

Search all log files for errors:

```bash
grep -i "error\|traceback\|OOM\|CUDA out of memory\|exception\|killed\|SIGSEGV\|signal\|RuntimeError\|dropout" <job_name>-algo-*.log | tail -60
```

Common failure patterns:
- **SIGSEGV / Signal 11**: Bad GPU or node — transient hardware failure, just retry
- **CUDA out of memory / OOM**: Reduce `per_gpu_batch_size` or `global_batch_size`, or use instances with more GPU memory (p5en > p5)
- **NCCL error / network error / remote process exited**: Usually cascading from another node's failure — find the root cause node
- **ncclRemoteError**: Another node crashed first — look for the node with SIGSEGV or OOM
- **Timeout on NCCL init**: Network issue between nodes, retry
- **RuntimeError in model code**: Actual bug — read the full traceback

### 4. Find root cause node

For NCCL cascade failures, the root cause is typically the node that failed first. Check timestamps:

```bash
grep -h "SIGSEGV\|CUDA out of memory\|exitcode" <job_name>-algo-*.log | sort | head -5
```

### 5. Read detailed logs from the root cause node

Once you identify the failing node, read its full log for context:

```bash
tail -100 <job_name>-algo-<N>-*.log
```

### 6. Report

Present a concise summary:
- **Job name**: full name
- **Status**: Failed/InProgress/etc.
- **Root cause**: which node, what error
- **Recommendation**: retry / fix config / fix code
- If the job is InProgress, show recent training metrics (loss, step, throughput) from the logs

### 7. Cleanup

Remove downloaded log files after analysis:
```bash
rm -f <job_name>-algo-*.log
```
