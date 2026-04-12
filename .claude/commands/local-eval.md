Run a local simulation evaluation for a checkpoint on a single task.

Arguments: `<checkpoint_s3_or_local_path> <TaskName> [--samples N] [--dist-shift] [--compare <other_checkpoint>]`

Examples:
- `/local-eval s3://tri-ml-datasets-uw2/vla_foundry_checkpoints/sim_only/.../checkpoint/ BimanualPutRedBellPepperInBin`
- `/local-eval /tmp/my_checkpoint TurnCupUpsideDown --samples 20 --dist-shift`
- `/local-eval s3://...ft_checkpoint/ PushCoasterToCenterOfTable --compare s3://...mt_checkpoint/`

## Overview

This runs evaluation locally using the Anzu Docker simulation + the vla_foundry inference policy server. Useful for quick spot-checks before launching full cluster evaluations.

## Steps

### 1. Parse arguments

- `CHECKPOINT`: S3 or local path to the checkpoint directory
- `TASK`: Task name (e.g., `BimanualPutRedBellPepperInBin`, `PushCoasterToCenterOfTable`)
- `--samples N`: Number of rollouts (default: 10)
- `--dist-shift`: Enable distribution shift in sim (adds `-e WITH_DIST_SHIFT=1` to Docker)
- `--compare <path>`: Run a second checkpoint on the same task for A/B comparison

### 2. Download checkpoint if S3

If the checkpoint is an S3 path:
1. Create a local directory under `/home/jeanmercat/eval_checkpoints/<short_name>/`
2. Download all config files: `config.yaml`, `config_model.yaml`, `config_normalizer.yaml`, `config_processor.yaml`, `stats.json`, `preprocessing_configs.yaml`, `processing_metadata.json`
3. Find the highest EMA checkpoint: `aws s3 ls <path>/checkpoints/ --profile sagemaker | grep ema_ | sort -n | tail -1`
4. Download the EMA checkpoint
5. Create a symlink: `checkpoint_N.pt -> ema_N.pt` (needed for `get_latest_checkpoint`)
6. Verify disk space FIRST (`df -h /`). Each checkpoint is ~6GB. Warn if < 10GB free.

### 3. Launch simulation Docker container

```bash
DIST_SHIFT_FLAGS=""
if [[ "$DIST_SHIFT" == "true" ]]; then
    DIST_SHIFT_FLAGS="-e WITH_DIST_SHIFT=1"
fi

SIM_CID="$(docker run --rm -d --network host \
    --runtime=nvidia \
    --gpus all \
    --device /dev/dri \
    --group-add video \
    --group-add $(stat -c '%g' /dev/dri/renderD128) \
    $DIST_SHIFT_FLAGS \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e SKIP_BUILD=1 \
    -e LAUNCH_DEMONSTRATION_INDICES="0:$SAMPLES" \
    -e LAUNCH_SAVE_DIR="/tmp/lbm/rollouts/" \
    -e LAUNCH_SUMMARY_DIR="/tmp/lbm/rollouts/" \
    -v $SSH_AUTH_SOCK:/ssh-agent \
    -e SSH_AUTH_SOCK=/ssh-agent \
    -v ~/.aws:/home/anzu/.aws \
    anzu-vla-foundry:latest \
    bash /opt/anzu/launch_sim.sh $TASK)"
```

### 4. Launch inference policy server

**CRITICAL**: Use the `/tmp/run_inference.sh` wrapper to avoid Python path contamination between worktrees:

```bash
cat > /tmp/run_inference.sh << 'EOF'
#!/bin/bash
VLA_DIR=/media/jeanmercat/471dbe07-0fd5-429a-b5d9-63c4eb5824b4/Code/lbm2_bis/worktrees/jean/vla_integration
cd "$VLA_DIR"
export PYTHONPATH="$VLA_DIR:$VLA_DIR/packages/grpc-workspace/src:$VLA_DIR/packages/robot-gym/src"
exec "$VLA_DIR/.venv/bin/python" -P vla_foundry/inference/robotics/inference_policy.py "$@"
EOF
chmod +x /tmp/run_inference.sh
```

Then launch:
```bash
CUDA_VISIBLE_DEVICES=0 /tmp/run_inference.sh \
    --checkpoint_directory $LOCAL_CHECKPOINT \
    --num_flow_steps 8 \
    --device cuda \
    --open_loop_steps 8 \
    --gripper_debounce_open_threshold 0.6 \
    --gripper_debounce_close_threshold 0.4
```

**Important notes:**
- The `-P` flag prevents CWD from leaking into `sys.path` (worktree cross-contamination)
- `PYTHONPATH` must explicitly point to the vla_integration worktree
- `tifffile` must be installed in the venv: `cd $VLA_DIR && uv pip install tifffile`
- Gripper debounce thresholds (0.6/0.4) must be passed explicitly (CLI defaults are None)

### 5. Monitor rollouts

Poll the Docker container for completed `summary.yaml` files:
```bash
while true; do
    count=$(docker exec $SIM_CID bash -c \
        'ls /tmp/lbm/rollouts/demonstration_*/summary.yaml 2>/dev/null | wc -l')
    echo "$(date +%H:%M:%S) - $count/$SAMPLES"
    if [ "$count" = "$SAMPLES" ]; then break; fi
    sleep 30
done
```

### 6. Report results

Extract success/failure from summary files:
```bash
docker exec $SIM_CID bash -c \
    'for f in $(ls /tmp/lbm/rollouts/demonstration_*/summary.yaml | sort -V); do
        grep "^success:" "$f"
    done'
successes=$(docker exec $SIM_CID bash -c \
    'grep -rl "^success: true" /tmp/lbm/rollouts/demonstration_*/summary.yaml | wc -l')
echo "Result: $successes/$SAMPLES"
```

### 7. Cleanup and compare (if --compare)

Kill inference + Docker, then repeat steps 2-6 for the comparison checkpoint. Report side-by-side:

```
=== SUMMARY: $TASK ===
Checkpoint A: X/N (XX%)
Checkpoint B: Y/N (YY%)
```

### 8. Final cleanup

```bash
kill $(pgrep -f inference_policy.py) 2>/dev/null
docker kill $SIM_CID 2>/dev/null
```

## Known issues

- **Disk space**: Each checkpoint is ~6GB. Check `df -h /` before downloading. Clean `/home/jeanmercat/eval_checkpoints/` or `/tmp/` if needed.
- **First episode is slow**: The sim takes 3-5 minutes for the first episode (scene build + GPU warmup). Subsequent episodes are ~2-3 min each.
- **Docker image**: Uses `anzu-vla-foundry:latest`. Pull the latest if results seem off: `docker pull 682769330988.dkr.ecr.us-east-1.amazonaws.com/anzu-vla-foundry:latest`
