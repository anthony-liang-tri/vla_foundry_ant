# OSS Eval

Run vla_foundry model evaluation using the OSS simulator Docker image (`lbm-eval-oss`) on a Ray cluster.

## Quick Start

Use the same `launch_campaign.sh` your team already uses — just add
`--use-oss-sim` to switch to the OSS simulator.

```bash
# Standard sim (what you do today)
./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh \
  vla_foundry/tri/oss_eval/campaigns/smoke_test.yaml

# OSS sim — same YAML, just add the flag
./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh \
  vla_foundry/tri/oss_eval/campaigns/smoke_test.yaml --use-oss-sim

# Works with any campaign YAML
./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh \
  vla_foundry/tri/lbm_eval/eval_campaigns/campaigns/example_campaign.yaml --use-oss-sim

# Dry-run to validate
./vla_foundry/tri/lbm_eval/eval_campaigns/launch_campaign.sh \
  vla_foundry/tri/oss_eval/campaigns/smoke_test.yaml --use-oss-sim --dry-run

# Monitor — use commands from CLUSTER_INFO.txt in results dir
ray dashboard results/<campaign>/cluster_config_expanded.yaml   # port-forward dashboard
cat results/<campaign>/CLUSTER_INFO.txt                         # show dashboard URL + useful commands
```

## Switching Between OSS and Standard Simulators

Campaign YAMLs are sim-agnostic — the same YAML works with both simulators.
The `--use-oss-sim` flag controls which simulator runs. Everything else
(inference server, checkpoints, tasks) stays the same.

The flag atomically applies all OSS overrides: Docker image,
`skip_mount_scripts`, `run_extra_args` (mounts standard bundle + video env
vars), `evaluation_subfolder` (`oss_eval`), and `results_dir` (`_oss` suffix).

**Using someone else's campaign YAML:** Just add `--use-oss-sim` — the flag
works with any campaign YAML. If the YAML already has custom
`docker.run_extra_args` (e.g., extra volume mounts), the OSS args are
appended rather than replacing them. The `evaluation_subfolder` is only
set if the YAML doesn't already specify one.

You can also set `docker.sim_type: "oss"` in a campaign YAML to make it
permanently OSS without needing the flag.

**Running both sims concurrently:** Use separate `cluster_name` values to
avoid cluster collisions. See `campaigns/smoke_test_standard.yaml` for an
example that can run alongside an OSS eval.

## The Two Docker Images

### `anzu-vla-foundry` (standard / internal)

The standard eval image. Contains the full internal Anzu simulation stack built
with Bazel, including the `demonstrate` binary that drives sim episodes. The
standard `launch_sim.sh` (mounted from `cluster_scripts/`) calls
`bazel run //intuitive/visuomotor:demonstrate` to run episodes.

### `lbm-eval-oss` (OSS)

The open-source eval image. Instead of the Bazel-based simulation, it ships:

- `/opt/lbm_eval/` — the `lbm_eval` Python package with an `evaluate` CLI
  entry point that drives sim episodes via Drake/PyDrake (no Bazel needed)
- `/opt/lbm_eval/.venv/` — a pre-built venv (not used by our campaigns — we
  run `uv sync` to create a vla_foundry venv, same as the standard flow)
- `/opt/anzu/launch_sim.sh` — a baked-in script that wraps the `evaluate` CLI
  with the same env-var interface as the standard `launch_sim.sh`
  (`LAUNCH_TASK_NAME`, `LAUNCH_DEMONSTRATION_INDICES`, `LAUNCH_SAVE_DIR`, etc.)
- `/opt/anzu/run_inference_bundle.sh` — a baked-in bundle (NOT used by our
  campaigns — we mount the standard bundle instead)
- **No** `/opt/vla_foundry/` — mounted from the cluster
- **No** Bazel, no `demonstrate` binary

Key differences:

| | `anzu-vla-foundry` | `lbm-eval-oss` |
|---|---|---|
| Sim backend | Bazel `demonstrate` binary | `evaluate` CLI (Drake/PyDrake) |
| `launch_sim.sh` | Mounted from cluster (runs Bazel) | Baked into image (runs `evaluate`) |
| `/opt/vla_foundry` | Exists (or mounted) | Must be mounted |

Everything else — `run_inference_bundle.sh`, `uv sync`, `uv run`, inference
server, checkpoint download, retries, S3 upload — is **identical**.

## How It Works

The only difference from a standard eval campaign is which `launch_sim.sh`
runs the simulator. Everything else uses the same code path.

### What the campaign YAML does differently

```yaml
docker:
  image: "...lbm-eval-oss:latest"  # OSS sim image instead of anzu-vla-foundry
  skip_mount_scripts: true          # Keep the baked-in OSS launch_sim.sh
  # Mount the standard run_inference_bundle.sh (for uv sync, retries, etc.)
  run_extra_args: "-v .../run_inference_bundle.sh:/usr/local/bin/run_inference_bundle.sh:ro"
```

- `skip_mount_scripts: true` prevents mounting both `run_inference_bundle.sh`
  AND `launch_sim.sh` from the cluster. This keeps the baked-in OSS
  `launch_sim.sh` (which uses `evaluate` instead of Bazel).
- `docker.run_extra_args` mounts just the standard `run_inference_bundle.sh`
  back in, so we get the full orchestration pipeline.
- No `inference.cmd_override` needed — the standard bundle's default already
  runs `uv run --group inference python inference_policy.py`.

### Execution flow

```
launch.sh
  |--> Injects --email into campaign YAML
  |--> uv run python -m ...run_evaluation_campaign <config.yaml>
         |
         |--> Clones vla_foundry_internal (ref: main) to ~/.cache/vla_foundry/ray_mount
         |--> Spins up Ray cluster (ray up)
         |--> Submits Ray jobs via ray_policy_runner.py
                |
                |--> For each (task, checkpoint), runs docker with:
                       - Standard run_inference_bundle.sh (mounted via run_extra_args)
                       - Baked-in OSS launch_sim.sh (kept via skip_mount_scripts)
                       - vla_foundry repo mounted at /opt/vla_foundry
```

### Inside the Docker container

```
run_inference_bundle.sh (standard — mounted from cluster)
  |
  |--> 1. uv sync (installs vla_foundry + deps into /opt/vla_foundry/.venv)
  |       Same as standard eval — uses flock to prevent concurrent corruption.
  |
  |--> 2. Downloads checkpoint from S3 to local disk
  |
  |--> 3. Launches BOTH in parallel:
  |       a) launch_sim.sh (baked-in OSS version)
  |          - Runs: evaluate --skill_type=<skill> --num_evaluations=N ...
  |          - Talks to gRPC policy server on localhost:50051
  |       b) uv run --group inference python inference_policy.py
  |          - Loads checkpoint, listens on port 50051
  |
  |--> 4. Monitors progress, handles retries on SimFailure/OOM
  |
  |--> 5. launch_sim.sh uploads results to S3:
  |       {checkpoint}/evaluation/{evaluation_subfolder}/{task}/rollouts/
  |
  |--> 6. Bundle cleans up checkpoint and rollout files from disk
```

Steps 1-4 and 6 are identical to the standard (anzu-vla-foundry) eval.
Step 5 uses the baked-in `launch_sim.sh` which respects
`LAUNCH_EVALUATION_SUBFOLDER`.

### Environment isolation: inference vs sim

The inference server and the simulator run inside the same Docker container but
use **completely separate Python environments** with independent dependency
sets. This mirrors the local workflow where a developer runs
`inference_policy.py` on their host machine (in a uv venv) and `evaluate`
inside the Docker container (in the image's venv), communicating over gRPC.

**Inference environment** (`/opt/vla_foundry/.venv/`):
- Created by `uv sync` from vla_foundry's `pyproject.toml` + `uv.lock`
- Contains vla_foundry's deps: torch, grpcio (specific version), transformers,
  timm, einops, diffusers, safetensors, etc.
- Started via: `env -u PYTHONPATH -u VIRTUAL_ENV uv run --group inference python inference_policy.py`
- The `env -u PYTHONPATH -u VIRTUAL_ENV` prefix strips any inherited
  environment before entering the uv-managed venv

**Simulator environment** (`/opt/lbm_eval/.venv/`):
- Pre-built into the `lbm-eval-oss` Docker image
- Contains lbm_eval's deps: Drake/PyDrake, robot_gym, grpcio (may differ
  from vla_foundry's version), etc.
- Started via: `env -u PYTHONPATH -u VIRTUAL_ENV bash launch_sim.sh`
- The baked-in `launch_sim.sh` calls the `evaluate` CLI, which is a
  console_scripts entry point with a hardcoded shebang to
  `/opt/lbm_eval/.venv/bin/python` — so it uses the OSS venv's Python
  regardless of any environment variables

**No cross-contamination:** each process uses its own venv with its own
installed packages. The standard bundle's `env -u PYTHONPATH -u VIRTUAL_ENV`
ensures neither side inherits the other's environment. They communicate
solely via gRPC on `localhost:50051`.

### Output format differences: OSS vs standard

The OSS `evaluate` CLI and the standard Bazel `demonstrate` binary write
different output structures. This matters for downloading results and for
the results dashboard.

**Standard (`anzu-vla-foundry` / Bazel `demonstrate`):**
```
rollouts/
  demonstration_100/
    summary.yaml            ← per-episode success/failure + metadata
    episode_100.pkl          ← full episode data
  demonstration_101/
    summary.yaml
    episode_101.pkl
```

**OSS (`lbm-eval-oss` / `evaluate` CLI):**
```
rollouts/
  results-2026-04-02T01:18:57.json   ← aggregated results for all episodes
  put_banana_on_saucer/               ← skill_name subdirectory (snake_case)
    demonstration_100/
      recording.html                  ← interactive 3D recording
      keyframes.txt
      random_scenario.yaml
      video_mosaic.mp4                ← if RECORD_VIDEO=1
    demonstration_101/
      ...
```

Key differences:
- OSS nests demonstrations under a `skill_name/` subdirectory; standard
  puts them directly in rollouts
- OSS writes `results-*.json` (aggregated); standard writes per-demo
  `summary.yaml`
- OSS writes `recording.html`; standard writes `episode_*.pkl`

**Known limitation:** The campaign orchestrator's automatic download
(`_download_cluster_artifacts`) was written for the standard format — it
looks for `demonstration_*/summary.yaml` at the top level. The OSS format
nests demos under `skill_name/` and uses `results-*.json` instead of
`summary.yaml`, so the automatic download reports "No demonstrations found"
even though the data is in S3.

### Viewing results with the dashboard

Download results from S3 manually and use the results dashboard:

```bash
# Download results (replace the S3 path with your checkpoint + evaluation_subfolder)
AWS_PROFILE=sagemaker aws s3 sync \
  "s3://<checkpoint_path>/evaluation/<evaluation_subfolder>/" \
  rollouts/ \
  --exclude "*.pkl"

# Example with the smoke test checkpoint:
AWS_PROFILE=sagemaker aws s3 sync \
  "s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/diffusion_policy/ablations/multitask/100m/2026_01_07-23_38_39-model_diffusion_policy-lr_5e-05-bsz_1024_converted/evaluation/oss_smoke_test/" \
  rollouts/ \
  --exclude "*.pkl"

# Launch the dashboard
uv run --group eval-viewer python vla_foundry/eval/results_explorer.py rollouts/
```

This downloads the OSS output structure:
```
rollouts/
  BimanualPutRedBellPepperInBin/
    rollouts/
      results-*.json              ← dashboard reads this
      bimanual_put_.../
        demonstration_100/
          recording.html          ← viewable in dashboard
        demonstration_101/
          ...
  PutBananaOnSaucer/
    rollouts/
      results-*.json
      put_banana_.../
        demonstration_100/
          recording.html
```

The dashboard's `load_episodes` finds `results-*.json` recursively and
displays per-episode success/failure, durations, and links to recordings.

**Comparing multiple checkpoints:** Use `download_results.sh` to download
results from multiple checkpoints into a layout the dashboard can compare:

```bash
./download_results.sh \
  s3://.../checkpoint_A/evaluation/oss_eval \
  s3://.../checkpoint_B/evaluation/oss_eval

uv run --group eval-viewer python vla_foundry/eval/results_explorer.py rollouts/
```

This creates `rollouts/{TaskName}/{checkpoint_name}/results-*.json`, so the
dashboard shows each checkpoint as a separate model for side-by-side comparison.

### S3 output paths

The `evaluation_subfolder` in the campaign YAML controls the S3 destination:

```
{checkpoint_s3_path}/evaluation/{evaluation_subfolder}/{TaskName}/rollouts/
```

When using `--use-oss-sim`, the subfolder defaults to `oss_eval`.
Standard eval (no flag) has no subfolder.

This ensures OSS eval results don't overwrite results from other campaigns.

### What vla_foundry code runs

The campaign clones `vla_foundry_internal` at the `ref` specified in the YAML
(default: `main`) and mounts it into each container. To test code from a
different branch, change the `vla_foundry` section:

```yaml
vla_foundry:
  repo_url: "git@github.com:TRI-ML/vla_foundry_internal.git"
  ref: "my-feature-branch"   # <-- change this
```

Or to use your local checkout directly (no git clone):

```yaml
vla_foundry:
  use_remote_repo: false
  local_source: "."          # uses the current repo root
```

## Files

```
tri/oss_eval/
  download_results.sh        # Download results from S3 for dashboard viewing
  campaigns/
    smoke_test.yaml           # 2 tasks, 5 episodes, 2 workers (sim-agnostic)
    full_16_tasks.yaml        # 16 tasks, 200 episodes, 256 workers (sim-agnostic)
    smoke_test_standard.yaml  # Same as smoke_test but with separate cluster_name for concurrent runs
  task_lists/
    smoke_test.txt            # PutBananaOnSaucer + BimanualPutRedBellPepperInBin
    all_16.txt                # All 16 standard eval tasks
```

### How scripts get into the container

The campaign uses `skip_mount_scripts: true` + `docker.run_extra_args`:

1. `skip_mount_scripts: true` tells `ray_policy_runner.py` not to mount
   cluster scripts into the container. This preserves the baked-in OSS
   `launch_sim.sh` at `/opt/anzu/launch_sim.sh`.

2. `docker.run_extra_args` mounts the standard `run_inference_bundle.sh`
   from the cluster node into the container at `/usr/local/bin/`. This is
   the same script used by `anzu-vla-foundry` campaigns.

3. The standard bundle's `LAUNCH_SCRIPT` defaults to `launch_sim.sh`, which
   resolves to `/opt/anzu/launch_sim.sh` — the baked-in OSS version.

This is wired up via `docker.run_extra_args` in the campaign YAML (plumbed
through `run_evaluation_campaign.py` -> `ray_policy_runner.py
--docker-run-extra-args`). The standard `run_inference_bundle.sh` is uploaded
to Ray nodes via `file_mounts` in `ray_policy_runner_cluster.yaml`.
