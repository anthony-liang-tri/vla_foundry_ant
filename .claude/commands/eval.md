Launch, monitor, and manage evaluation campaigns on a Ray cluster.

## Mandatory checklist

**Create a task list at the start of every /eval invocation and check off each step as you complete it.** Do not skip any step. Do not consider the eval done until all boxes are checked.

- [ ] **Resolve** — S3 path, checkpoint, task detection (Steps 1-3)
- [ ] **Create** — task list + campaign config (Steps 4-5)
- [ ] **Confirm** — present launch plan to user, wait for go (Step 5.5)
- [ ] **Push** — commit, push to both remotes, update cached git clone (Step 6)
- [ ] **Collision check** — verify no existing results at target S3 subfolder (Step 6.5)
- [ ] **Launch** — start campaign in background (Step 7)
- [ ] **Monitor** — set up cron, check Ray head node + local log + S3 every cycle (Step 8)
- [ ] **Gather** — wait for process to fully exit (process count = 0), THEN download results from S3 and parse per-task (Step 9)
- [ ] **Save report** — write markdown report to `eval_campaigns/results/`, commit+push (Step 10)
- [ ] **Cancel cron** — delete the monitoring cron job

## Input formats

The argument can be:
- **wandb run name**: e.g. `2026_04_07-06_39_29-model_diffusion_policy-lr_5e-05-bsz_256` — auto-discovers S3 path, creates task list, launches
- **Multiple wandb run names / S3 paths** (space-separated): all models are resolved and their tasks are **combined into a single task list and evaluated together in one campaign**. This is more efficient than sequential campaigns and should always be used when evaluating multiple models at once.
- **S3 checkpoint path**: e.g. `s3://tri-ml-datasets-uw2/.../checkpoints/checkpoint_11.pt` or the parent dir — launches directly
- **Action command**: `launch <config.yaml>`, `status`, `kill`, `relaunch`
- **Sim type flag**: append `--oss` (default), `--anzu`, or `--both` to run with specific sim(s)
- **Task set flag**: append `--unseen` to eval on unseen tasks only, or `--all` for seen + unseen (multitask models only)

Examples:
- `/eval 2026_04_07-06_39_29-model_diffusion_policy-lr_5e-05-bsz_256` — find checkpoint, auto-detect tasks, launch OSS eval
- `/eval run_A run_B run_C` — resolve all 3, combine into one task list, launch a single campaign evaluating all at once
- `/eval s3://tri-ml-datasets-uw2/.../my_model/ --both` — run both anzu and OSS evals
- `/eval launch campaigns/my_campaign.yaml` — launch existing campaign
- `/eval status` — check running campaign progress
- `/eval kill` — kill running campaign and tear down cluster

## Multi-model eval (multiple inputs)

When given multiple wandb run names or S3 paths, **do not queue them sequentially**. Instead:

1. Resolve each model's S3 path and detect its task(s) (Steps 1–2 of the auto-eval pipeline, run in parallel for all models)
2. Create a **single combined task list** with all entries from all models, using per-model prefixes to distinguish them (e.g. `model_A_TaskName`, `model_B_TaskName`)
3. Create a **single campaign config** pointing to that combined task list
4. Run one campaign that evaluates everything in parallel on the same cluster — this is faster and cheaper than N sequential campaigns
5. When gathering results, parse and report per-model success rates separately

## Auto-eval pipeline (wandb name or S3 path)

### Step 1: Resolve checkpoint S3 path

If input is a **wandb run name** (matches pattern `\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}-model_.*`):

**Method 1: wandb API (fast, try first)**
```python
import wandb
api = wandb.Api()
# IMPORTANT: Use api.runs() to find the run ID, then api.run() to get full config.
# The runs iterator doesn't always load config; direct access by ID does.
runs = api.runs("tri/vla_foundry", filters={"display_name": RUN_NAME}, per_page=5)
for r in runs:
    # Re-fetch the run directly by ID to get full config
    full_run = api.run(f"tri/vla_foundry/{r.id}")
    if full_run.config:
        remote_sync = full_run.config.get("remote_sync")
        if remote_sync:
            S3_PATH = remote_sync  # e.g. s3://tri-ml-datasets-uw2/.../model_name/TaskName
            # The actual run dir is remote_sync + "/" + RUN_NAME + "/"
            # Verify by checking if config.yaml exists there
            break
        # Can also detect single/multi from config:
        # full_run.config["data"]["dataset_manifest"] contains task-specific paths
```
Note: The wandb entity is `tri`, project is `vla_foundry`. Fall back to Method 2 if wandb API fails or run not found.

**Method 2: S3 scan (slower, always works)**
```bash
for prefix in \
  "s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/" \
  "s3://tri-ml-datasets-uw2/lbm2_vla/model_checkpoints/" \
  "s3://tri-ml-datasets-uw2/vla_foundry_checkpoints/" \
  "s3://tri-ml-datasets/vla_foundry_scratch/"; do
  aws s3 ls "$prefix" --recursive --profile sagemaker 2>/dev/null | grep "$RUN_NAME/config.yaml" | head -1
done
```

In both cases:
- Extract the checkpoint directory (parent of config.yaml)
- Verify `config.yaml` exists at that path

If input is an **S3 path**: use it directly. Strip `/checkpoints/checkpoint_*.pt` suffix if present to get the run directory.

### Step 2: Detect single-task vs multi-task

Download `preprocessing_configs.yaml` from the checkpoint and extract unique task names from `source_episodes`:
```bash
aws s3 cp "$CKPT/preprocessing_configs.yaml" - --profile sagemaker | grep "source_episodes" -A1000 | grep "tasks/" | sed 's|.*/tasks/||;s|/.*||' | sort -u
```
- **1 unique task name** → single-task model. Eval only that task.
- **Multiple task names** → multi-task model. Task set depends on flags:

  **Standard 16 seen tasks** (default):
  ```
  BimanualPlaceAppleFromBowlIntoBin BimanualPlaceFruitFromBowlIntoBin BimanualPutRedBellPepperInBin
  BimanualPutSpatulaOnPlateFromDryingRack BimanualPutSpatulaOnPlateFromTable
  BimanualStackPlatesOnTableFromDryingRack BimanualStoreCerealBoxUnderShelf PlaceCupByCoaster
  PushCoasterToCenterOfTable PushCoasterToMug PutBananaOnSaucer PutKiwiInCenterOfTable
  PutMugOnSaucer PutSpatulaInUtensilCrock TurnCupUpsideDown TurnMugRightsideUp
  ```

  **3 unseen tasks** (`--unseen` flag, multitask only):
  ```
  BimanualPlaceAvocadoFromBowlIntoBin BimanualPutSpatulaOnPlateFromUtensilCrock PutMugInCenterOfTable
  ```

  **All 19 tasks** (`--all` flag): seen + unseen combined.

  Unseen tasks are NOT included by default — they must be explicitly requested via `--unseen` or `--all`. They are only meaningful for multitask models (singletask models are only evaluated on their training task).

### Step 3: Find the best checkpoint

List checkpoints and pick the latest EMA (if EMA enabled) or latest regular:
```bash
aws s3 ls "$CKPT/checkpoints/" --profile sagemaker | grep -E "ema_|checkpoint_" | sort -t_ -k2 -n | tail -5
```
Check `config.yaml` for `ema.enabled`. The inference code auto-selects EMA vs regular, so pointing to the run directory is sufficient.

### Step 4: Create task list file

Generate at `vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/<run_short_name>.txt`:
```
    ("<prefix>_<TaskName>", "<TaskName>", "<s3_checkpoint_dir>/"),
```
Use a short name derived from the run (e.g., `fromvlm_ckpt11` or the wandb run name truncated).

### Step 5: Create campaign config

Generate at `vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/<run_short_name>.yaml`:
```yaml
campaign_name: "<run_short_name>"
description: "Auto-generated eval for <run_name>"

cluster:
  config_file: "vla_foundry/tri/lbm_eval/eval_campaigns/ray_policy_runner_cluster.yaml"
  num_workers: 256
  aws_profile: "manip-cluster"
  owner_email: "jean.mercat@tri.global"
  teardown_on_completion: true
  teardown_on_failure: false

evaluation:
  jobs_per_gpu: 1
  evaluation_subfolder: "<date>_<run_short_name>"
  tasks_file: "vla_foundry/tri/lbm_eval/eval_campaigns/task_lists/<run_short_name>.txt"
  num_samples: 200
  start_index: 0
  launch_scenario: "GrpcServerToSim"
  launch_script: "launch_sim.sh"
  num_flow_steps: 8
  open_loop_steps: 8
  device: "cuda"
  entrypoint_num_gpus: 1
  entrypoint_memory: 62277025792
  max_retries: 3
  inference_aws_profile: "sagemaker"

docker:
  image: "682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest"

vla_foundry:
  repo_url: "git@github.com:jmercat/vla_foundry.git"
  ref: "<current_branch>"
  local_dir: "~/.cache/vla_foundry/ray_mount"
  use_remote_repo: true
  remote_dir: "/home/ubuntu/vla_foundry_mount"
  mount_target: "/opt/vla_foundry"

output:
  results_dir: "results/<run_short_name>"
  save_job_logs: true
```

### Step 5.5: Present launch plan and wait for confirmation

Before launching, present a concise summary to the user:

> **Eval plan:** This is a [single-task / multi-task (16 tasks)] model. Latest checkpoint: [checkpoint_X.pt / ema_X.pt]. I will run [200] samples per task on a Ray EC2 cluster (256 workers) using the [OSS Docker Hub / anzu / both] sim.
> - Checkpoint: `s3://.../<short_path>/`
> - Tasks: [TaskName] (single) or [16 standard tasks] (multi)
> - Sim: [OSS (toyotaresearch/lbm-eval-oss) / anzu-vla-foundry / both]
>
> Launching in 5 minutes unless you say otherwise.

**Implementation:** Start a `sleep 300 && echo "5 minutes elapsed"` background Bash command as a timer. Then proceed with Step 6 (commit, push, update cache) and Step 6.5 (collision check) while the timer runs — these pre-launch steps are needed regardless. If the user responds before the timer fires (confirming, modifying, or cancelling), act on their response immediately. If the timer fires with no user response, proceed with the launch. If user modifies the plan (e.g., "use anzu instead", "only 100 samples"), adjust accordingly.

### Step 6: Commit, push, update cache

```bash
git add <task_list> <campaign_config>
git commit -m "Add eval campaign for <run_name>"
git push origin <branch> && git push jean <branch>
cd ~/.cache/vla_foundry/ray_mount/default && git fetch origin <branch> && git reset --hard FETCH_HEAD
```

### Step 6.5: Check for existing results (collision guard)

Before launching, check if results already exist at the target S3 path:
```bash
aws s3 ls "$CKPT/evaluation/$SUBFOLDER/" --profile sagemaker 2>/dev/null | head -5
```
If results exist:
- **Warn the user**: "Found existing results at `s3://.../$SUBFOLDER/`. Launching will add to (not overwrite) these results, which may mix data from different runs."
- **Default action (no user response)**: automatically use a new unique subfolder (append `_v2`, `_v3`, etc.) to avoid mixing.
- Only reuse the same subfolder if the user explicitly says so (e.g., "proceed", "same subfolder is fine").

Also check the local `results/<name>/` directory:
```bash
ls results/<name>/campaign_state.json 2>/dev/null
```
If a stale `campaign_state.json` exists, `rm -rf results/<name>/` before launching (stale state causes "Head node not found" errors).

### Step 7: Launch

Based on `--oss` (default), `--anzu`, or `--both`:

- **OSS** (default): `AWS_PROFILE=manip-cluster bash launch_campaign.sh <config>.yaml --use-oss-sim > /tmp/eval_<name>_oss.log 2>&1 &`
- **Anzu**: `AWS_PROFILE=manip-cluster bash launch_campaign.sh <config>.yaml > /tmp/eval_<name>_anzu.log 2>&1 &`
- **Both**: Launch both sequentially (anzu first, then OSS after anzu tears down — they compete for EC2). Or in parallel if using different cluster names.

### Step 8: Monitor

Set up a cron (every 5 minutes initially, then 20 minutes once stable).

**Every status check MUST include ALL of these** (the local log lags behind Ray — never rely on it alone):
1. Process alive: `ps aux | grep run_evaluation_campaign | grep -v grep | wc -l`
2. Local log counts: `grep -oP "SUCCEEDED|FAILED" /tmp/eval_<name>.log | sort | uniq -c`
3. **Ray head node status** (authoritative — always check this): SSH to head node and query `curl -s http://localhost:8265/api/jobs/` for full SUCCEEDED/FAILED/RUNNING/PENDING counts
4. Report ALL counts together: "Ray: X SUCCEEDED, Y FAILED, Z RUNNING. Log: A SUCCEEDED, B FAILED."

- If >20% failed on Ray: **DO NOT KILL immediately.** First check if **completed** results are actually landing on S3:
  ```bash
  # Step 1: check file count
  aws s3 ls "$CKPT/evaluation/$SUBFOLDER/" --recursive --profile sagemaker 2>/dev/null | grep "results.json" | wc -l
  # Step 2: VERIFY CONTENT — file count alone means nothing, files may be empty placeholders
  aws s3 cp "$CKPT/evaluation/$SUBFOLDER/" /tmp/spot_check/ --recursive --exclude "*" --include "*/results.json" --profile sagemaker 2>/dev/null
  python3 -c "import json,glob; files=[f for f in glob.glob('/tmp/spot_check/**/results.json',recursive=True)]; completed=[f for f in files if json.load(open(f)).get('num_evaluated',0)>0]; print(f'{len(completed)}/{len(files)} files have actual results')"
  ```
  If completed results exist on S3 despite "FAILED" status, the issue is the S3 verification logic, not the actual inference. In that case:
  - **Let the campaign keep running** — the results are being produced correctly
  - Fix the verification bug for future runs
  - Gather results manually from S3 after the campaign finishes or exhausts retries
  - **NEVER kill a campaign that is producing completed results on S3**
  If files exist but are all empty/placeholder (`num_evaluated: 0`), the inference IS failing — investigate job logs.
- If campaign completes: gather results

**Monitoring frequency**:
- First 30 min: every 5 min (catch early failures fast)
- After first successes confirmed: every 20 min
- When >80% done: every 5 min again (watch for completion)

### Step 9: Gather results

For **anzu** results (summary.yaml format):
```bash
aws s3 sync "$CKPT/evaluation/$SUBFOLDER/" /tmp/results/ --exclude "*" --include "*/summary.yaml" --profile sagemaker
```

For **OSS** results (results.json format) — use `aws s3 cp --recursive` NOT `aws s3 sync` (sync silently drops files):
```bash
aws s3 cp "$CKPT/evaluation/$SUBFOLDER/" /tmp/results/ --recursive --exclude "*" --include "*/results.json" --profile sagemaker
```

Parse and aggregate per-task success rates, compute unweighted mean.

### Step 10: Save results report

**This step is mandatory — always save the report, even for single-model evals.**

Save to `vla_foundry/tri/lbm_eval/eval_campaigns/results/<run_short_name>_<date>.md`:

```markdown
# Eval Report: <run_short_name> — <date>

## Model(s)

| Campaign | Model dir | Checkpoint | WandB | S3 Checkpoint |
|----------|-----------|-----------|-------|---------------|
| <campaign_name> | <model_dir_name> | <ckpt_N> | [run](<wandb_url>) | [s3](<s3_checkpoint_dir>/) |

> WandB URL format: https://wandb.ai/tri/vla_foundry/runs/<run_id>
> Run ID obtained via: `api.runs("tri/vla_foundry", filters={"display_name": RUN_NAME})[0].id`

## Results — <sim_type> Sim

### <campaign_name> (single-task OR multi-task mean)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| ... | ... | ... | ... |
| **Unweighted Mean** | | | **XX.X%** |

## S3 Result Paths
- OSS: `<s3_checkpoint_dir>/evaluation/<oss_subfolder>/`
- Anzu: `<s3_checkpoint_dir>/evaluation/<anzu_subfolder>/`
```

After saving, commit the report:
```bash
git add vla_foundry/tri/lbm_eval/eval_campaigns/results/
git commit -m "Add eval report for <run_short_name>"
git push origin <branch> && git push jean <branch>
```

## Manual campaign operations

### `/eval launch <config>`

1. **Pre-flight checks:**
   - Verify the campaign YAML exists
   - Check that all code changes are committed and pushed to the branch specified in `vla_foundry.ref`
   - Update the cached git clone: `cd ~/.cache/vla_foundry/ray_mount/default && git fetch origin <branch> && git reset --hard origin/<branch>`
   - Verify the task list file exists and has no leading spaces in S3 paths

2. **Launch:**
   ```
   AWS_PROFILE=manip-cluster bash vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/<config>.yaml > /tmp/eval_campaign_output.log 2>&1
   ```
   Run in background. Add `--use-oss-sim` for OSS eval.

3. **Set up monitoring cron** as described in Step 8 above.

### `/eval status`

1. Check if a campaign process is running: `ps aux | grep run_evaluation_campaign | grep -v grep`
2. Check campaign log counts: `grep -oP "SUCCEEDED|FAILED" /tmp/eval_*.log | sort | uniq -c`
3. Check Ray directly if possible (via `ray exec` on head node)
4. Report progress concisely

### `/eval kill`

**NEVER kill without checking first.** A campaign that shows "FAILED" may still be producing valid results on S3 (e.g., if the S3 verification logic has a bug but inference is fine).

1. **Check status first** — before killing, always report the current state:
   - How many jobs SUCCEEDED/FAILED/RUNNING (from Ray head node)
   - **Check S3 for actual results** regardless of FAILED count:
     ```bash
     aws s3 ls "$CKPT/evaluation/$SUBFOLDER/" --recursive --profile sagemaker 2>/dev/null | grep "results.json\|summary.yaml" | wc -l
     ```
   - If S3 has results despite FAILED status: "Campaign has X FAILED but Y results on S3 — inference is working, verification is broken. Do NOT kill. Gather results manually instead?"
   - Present full picture to the user and **wait for explicit confirmation** before killing
2. **Save logs before teardown** (download job logs from head node first!)
3. Kill campaign processes: `pkill -f "launch_campaign|run_evaluation_campaign|ray_policy_runner"`
4. Tear down cluster: `AWS_PROFILE=manip-cluster uv run ray down <cluster_config> -y`

### `/eval relaunch`

1. **Check status first** — report current state (SUCCEEDED/FAILED/RUNNING counts) AND check S3 for actual results before killing. If results exist on S3, ask the user before proceeding.
2. Kill the current campaign (only after user confirms)
3. **Check for partial results on S3** before cleaning:
   ```bash
   aws s3 ls "$CKPT/evaluation/$SUBFOLDER/" --profile sagemaker 2>/dev/null | head -5
   ```
   If partial results exist, warn: "Found partial results on S3. Relaunching with the same subfolder will mix new results with old."
   **Default action (no user response)**: automatically use a new subfolder with incremented suffix (e.g., `_v2`). Only reuse the same subfolder if the user explicitly confirms.
3. Check if there are uncommitted code changes that need pushing
4. Update cached git clone at `~/.cache/vla_foundry/ray_mount/default`
5. Clean local results directory: `rm -rf results/<name>/` (only local state, NOT S3)
6. Relaunch with the same config (and new subfolder if changed)

## Key infrastructure details

- **Script deployment:** `run_inference_bundle.sh` is mounted from the git clone at `/home/ubuntu/vla_foundry_mount/default/vla_foundry/tri/lbm_eval/eval_campaigns/cluster_scripts/` (NOT from Ray file_mounts which are stale). This is configured in `ray_policy_runner.py` via `--mount-local-run-bundle`.
- **launch_sim.sh** is mounted from Ray file_mounts at `/home/ubuntu/anzu_vla_foundry/` (synced during `ray up`). CRITICAL: the launch subshell must `cd "${LAUNCH_WORKDIR}"` before running the script.
- **Venv isolation:** Each Docker container creates its own venv in `/tmp/vla_foundry_runtime/` to avoid cross-container corruption on shared mounts. `INFERENCE_USE_WRITABLE_COPY=1` is always set.
- **Docker images:**
  - Anzu: `682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest` — internal sim with Bazel binary. Uses mounted launch_sim.sh.
  - OSS: `toyotaresearch/lbm-eval-oss:vla-foundry` — Docker Hub image with Python-based sim. Uses `--use-oss-sim` flag which sets `skip_mount_scripts=true` and mounts `run_inference_bundle.sh` via docker_run_extra_args.
- **Results format:** Anzu produces `summary.yaml` per demo. OSS produces `results.json` per batch.
- **S3 download:** Use `aws s3 cp --recursive` NOT `aws s3 sync` for gathering results — sync silently drops files.
- **IMDSv2:** The Ray cluster config must include `MetadataOptions: {HttpTokens: required, HttpEndpoint: enabled}`.
- **Cached git clone:** Always update `~/.cache/vla_foundry/ray_mount/default` before launching.
- **AWS profiles:** `manip-cluster` for EC2/Ray cluster operations, `sagemaker` for S3 model checkpoint access.
- **Always save logs before teardown.** Once the cluster is destroyed, job logs are gone forever.
