"""
Unitree G1 Diffusion Policy Training Launcher.

Builds draccus overrides and launches training locally (torchrun) or on
SageMaker, depending on flags. Replaces the bash wrapper.

Observation modes (--obs):
  vision_propio         Vision + EE pose + finger joints (32D)
  vision_propio_tactile Vision + EE pose + finger joints + torque + pressure
  vision_only           Vision only — no proprioception

Domain (--domain):
  real                  Real robot data (default)
  sim                   Simulation data
  real_and_sim          Both data domains

Camera config (--camera-config):
  zed_mini              ZED Mini stereo head + D435 wrist (default for real data)
  d435                  D435 head + D435 wrist (default for sim data)

Camera selection (--cameras):
  auto                  Load camera list from camera-config YAML (default)
  JSON list             Explicit subset, e.g. '["stereo_head_left","stereo_head_right"]'

Tactile proprioception (--tactile-propio):
  Off by default. Appends dex3 torque + pressure sensors to proprioception_fields.
  Requires dataset recorded with tactile hardware.
  Note: vision_propio_tactile obs mode always includes tactile regardless of this flag.

Action space: absolute EE pose + finger joints (32D). Relative actions do not train well.

Batch size guidance (p5, 8x H100 80GB, float32):
  Per-GPU: 16
  Global:  128 (8 GPUs x 16, no grad accumulation)
  global_batch = per_gpu * num_gpus

Usage:
  uv run python examples/training/extended/diffusion_policy/unitree_g1/train_g1_policy.py \\
      TASK --obs vision_propio [OPTIONS]

Examples:
  # Local 2-GPU run, real ZED Mini data (default)
  uv run python train_g1_policy.py stack_cubes_ordered --obs vision_propio

  # SageMaker p5
  uv run python train_g1_policy.py stack_cubes_ordered --obs vision_propio \\
      --batch 16 --global-batch 128 --sagemaker --user firstname.lastname

  # Sim data with D435 cameras
  uv run python train_g1_policy.py move_block_on_plate --obs vision_propio \\
      --domain sim --camera-config d435

  # With tactile proprioception
  uv run python train_g1_policy.py move_block_on_plate --obs vision_propio \\
      --domain sim --camera-config d435 --tactile-propio

  # Head-cam only, ZED Mini real data
  uv run python train_g1_policy.py move_block_on_plate --obs vision_propio \\
      --cameras '["stereo_head_left","stereo_head_right"]'
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

from vla_foundry.aws.s3_constants import DEFAULT_REGION, S3_PREFIX

# ---------------------------------------------------------------------------
# Field definitions — loaded from g1_data_params.yaml, not hardcoded here
# ---------------------------------------------------------------------------

_G1_DATA_DIR = "vla_foundry/config_presets/data/unitree_g1"
_DATA_PARAMS_PATH = f"{_G1_DATA_DIR}/g1_data_params.yaml"

with open(_DATA_PARAMS_PATH) as _f:
    _DATA_PARAMS = yaml.safe_load(_f)

_BASE_PROPIO_FIELDS = _DATA_PARAMS["proprioception_fields"]
_TACTILE_FIELDS = _DATA_PARAMS["tactile_fields"]
ACTION_FIELDS = _DATA_PARAMS["action_fields"]
# Camera name lists per camera hardware config
_CAMERA_YAML_BY_CONFIG = {
    "zed_mini": f"{_G1_DATA_DIR}/g1_data_camera_names_zed2mini.yaml",
    "d435": f"{_G1_DATA_DIR}/g1_data_camera_names_d435.yaml",
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_S3_TARFILE = "s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles"
DEFAULT_DATA_ROOT_V3 = f"{_S3_TARFILE}/v3.2"  # non-tactile
DEFAULT_DATA_ROOT_V2 = f"{_S3_TARFILE}/v2"  # tactile (dex3 torque/pressure)
CKPT_ROOT = "s3://robotics-cam-checkpoints/platform/unitree_g1_dex3/model_checkpoints"
REMOTE_SYNC_FIXED_PATH = "s3://robotics-cam-checkpoints/platform/unitree_g1_dex3/model_checkpoints_fixed/"

_CFG = "vla_foundry/config_presets/training_jobs/unitree_g1"

OBS_MODES = ["vision_propio", "vision_propio_tactile", "vision_only"]
MODEL_SIZES = ["100m", "410m"]

# (obs_mode, model_size) → config YAML path
_CONFIG_MATRIX = {
    ("vision_propio", "100m"): f"{_CFG}/diffusion_policy_unitree_g1_100m.yaml",
    ("vision_propio", "410m"): f"{_CFG}/diffusion_policy_unitree_g1_410m.yaml",
    ("vision_only", "100m"): f"{_CFG}/diffusion_policy_unitree_g1_100m.yaml",
    ("vision_only", "410m"): f"{_CFG}/diffusion_policy_unitree_g1_410m.yaml",
    ("vision_propio_tactile", "100m"): f"{_CFG}/diffusion_policy_unitree_g1_100m_tactile.yaml",
    ("vision_propio_tactile", "410m"): f"{_CFG}/diffusion_policy_unitree_g1_410m_tactile.yaml",
}

SM_REGION = DEFAULT_REGION
SM_ARN = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"
SM_PROFILE = "sagemaker"
SM_QUEUE = "tri-cam-humanoid"
SM_INSTANCE = "p5"
SM_PRIORITY = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _samples_tag(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _dataset_info(data_root: str) -> str:
    """Best-effort shard/sample count from manifest.jsonl."""
    manifest = f"{data_root}/manifest.jsonl"
    try:
        if data_root.startswith(S3_PREFIX):
            result = subprocess.run(
                ["aws", "s3", "cp", manifest, "-"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            content = result.stdout if result.returncode == 0 else ""
        else:
            with open(manifest) as f:
                content = f.read()

        if not content.strip():
            return data_root

        lines = [json.loads(line) for line in content.splitlines() if line.strip()]
        total = sum(line.get("num_samples", 0) for line in lines)
        suffix = f", {total:,} samples" if total else ""
        return f"{data_root}  ({len(lines)} shards{suffix})"
    except Exception:
        return data_root


def _verify_sagemaker_credentials(profile: str, arn: str) -> None:
    result = subprocess.run(
        ["aws", "--profile", profile, "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit(f"ERROR: Could not authenticate with profile '{profile}'\nTry: aws sso login --profile {profile}")
    caller = result.stdout.strip()
    arn_account = arn.split(":")[4]
    if caller != arn_account:
        sys.exit(f"ERROR: Profile authenticates to {caller} but ARN targets {arn_account}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", help="Task name (e.g. stack_cubes_ordered).")
    p.add_argument(
        "--obs",
        required=True,
        choices=OBS_MODES,
        metavar="MODE",
        help="Observation mode: vision_propio | vision_propio_tactile | vision_only",
    )
    p.add_argument(
        "--model",
        default="410m",
        choices=MODEL_SIZES,
        help="Transformer size: 100m or 410m (default: 410m).",
    )
    p.add_argument(
        "--domain",
        default="real",
        choices=["real", "sim", "real_and_sim"],
        help="Data domain: real (default), sim, or both. Affects default data path and W&B tags.",
    )
    p.add_argument(
        "--camera-config",
        default="zed_mini",
        dest="camera_config",
        choices=list(_CAMERA_YAML_BY_CONFIG),
        help=(
            "Camera hardware config: zed_mini (default) | d435. Selects camera name list. "
            "WARNING: verify this matches your dataset recording hardware — wrong config "
            "will silently train on a mismatched camera subset."
        ),
    )
    p.add_argument(
        "--cameras",
        default="auto",
        metavar="CAMERAS",
        help=(
            "Camera subset: 'auto' = all cameras from camera-config YAML (default), "
            "'head' = head cameras only, 'wrist' = wrist cameras only, "
            'or a JSON list e.g. \'["stereo_head_left","stereo_head_right"]\'.'
        ),
    )
    p.add_argument(
        "--tactile-propio",
        action="store_true",
        dest="tactile_propio",
        help="Append dex3 torque + pressure fields to proprioception (default: off).",
    )
    # Training budget
    budget = p.add_mutually_exclusive_group()
    budget.add_argument(
        "--samples",
        type=int,
        default=10_000_000,
        help="Total training samples (default: 10M; use 3M for quick debug runs).",
    )
    budget.add_argument("--steps", type=int, help="Training steps instead of samples (requires --global-batch).")

    # Batch size
    p.add_argument("--batch", type=int, help="Per-GPU batch size (default: from YAML).")
    p.add_argument(
        "--global-batch",
        type=int,
        dest="global_batch",
        help="Global batch size (default: from YAML). Required with --steps.",
    )

    # Data / checkpoints
    p.add_argument("--data-root", dest="data_root", help="Override shard directory (S3 or local path).")
    p.add_argument(
        "--norm-scope",
        dest="norm_scope",
        choices=["global", "per_timestep"],
        default=None,
        help="Override normalization scope (default: from config YAML).",
    )
    p.add_argument(
        "--run-tag",
        dest="run_tag",
        default=None,
        help="Suffix appended to checkpoint path and W&B run name (e.g. '100m_pertimestep').",
    )
    p.add_argument(
        "--wandb-tags",
        dest="wandb_tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Extra W&B tags appended to the auto-generated tag list (e.g. --wandb-tags ablation apr2026).",
    )
    p.add_argument(
        "--wandb-project",
        dest="wandb_project",
        default="vla_foundry_g1",
        help="W&B project name (default: vla_foundry_g1).",
    )
    p.add_argument(
        "--config-override",
        dest="config_override",
        default=None,
        help="Override the auto-selected config YAML path (e.g. for 100M or 410M ablations).",
    )
    p.add_argument(
        "--val-samples",
        type=int,
        dest="val_samples",
        help="Enable validation visualizations (default: off).",
    )

    # Local execution
    p.add_argument("--gpu", default="0,1", help="GPU indices for local training, comma-separated (default: 0,1).")

    # SageMaker
    p.add_argument("--sagemaker", action="store_true", help="Submit to SageMaker instead of running locally.")
    p.add_argument("--user", dest="sm_user", help="SageMaker user (firstname.lastname). Required with --sagemaker.")
    p.add_argument("--queue", default=SM_QUEUE, dest="sm_queue", help=f"SageMaker queue (default: {SM_QUEUE}).")
    p.add_argument(
        "--instance",
        default=SM_INSTANCE,
        dest="sm_instance",
        help=f"SageMaker instance type (default: {SM_INSTANCE}).",
    )

    p.add_argument("--dry-run", action="store_true", help="Print command without running it.")

    args = p.parse_args()

    if args.steps is not None and args.global_batch is None:
        p.error("--steps requires --global-batch")
    if args.sagemaker and not args.sm_user:
        p.error("--user is required for SageMaker runs")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Resolve samples
    samples = args.steps * args.global_batch if args.steps is not None else args.samples

    # Obs-mode config and proprioception fields
    config = args.config_override or _CONFIG_MATRIX[(args.obs, args.model)]
    if args.obs == "vision_propio":
        proprio_fields = _BASE_PROPIO_FIELDS
        obs_desc = "vision + EE pose (18D) + finger joints (14D)"
    elif args.obs == "vision_only":
        proprio_fields = []
        obs_desc = "vision only (no proprioception)"
    else:  # vision_propio_tactile
        proprio_fields = _BASE_PROPIO_FIELDS + _TACTILE_FIELDS
        obs_desc = "vision + EE pose + finger joints + torque + pressure"

    # --tactile-propio appends tactile fields to any obs mode (no-op for vision_propio_tactile)
    if args.tactile_propio and args.obs != "vision_propio_tactile":
        proprio_fields = proprio_fields + _TACTILE_FIELDS

    tactile_tag = "_tactile" if (args.tactile_propio and args.obs != "vision_propio_tactile") else ""

    # Camera selection
    cam_yaml = _CAMERA_YAML_BY_CONFIG[args.camera_config]
    with open(cam_yaml) as f:
        all_cameras = yaml.safe_load(f)

    _CAMERA_PRESETS = {"auto": None, "head": "head", "wrist": "wrist"}
    if args.cameras in _CAMERA_PRESETS:
        keyword = _CAMERA_PRESETS[args.cameras]
        camera_names = [c for c in all_cameras if keyword in c] if keyword else all_cameras
        cam_tag = f"_{args.cameras}only" if keyword else ""
    else:
        try:
            camera_names = json.loads(args.cameras)
        except json.JSONDecodeError:
            sys.exit(f"ERROR: --cameras must be auto|head|wrist or a JSON list, got: {args.cameras!r}")
        cam_tag = f"_{len(camera_names)}cam"

    if not camera_names:
        sys.exit(f"ERROR: --cameras '{args.cameras}' matched no cameras from {cam_yaml}")

    # Data paths
    use_tactile_data = args.tactile_propio or args.obs == "vision_propio_tactile"
    # FIXME(mark.zolotas): V2 data root is deprecated and pending migration
    # Remove this guard once V4 tactile data is available
    if use_tactile_data:
        raise NotImplementedError(
            "Tactile data still points to DEFAULT_DATA_ROOT_V2 which is no longer supported. "
            "Migrate tactile datasets to V4 before enabling this path."
        )
    default_root = DEFAULT_DATA_ROOT_V3
    data_root = args.data_root or f"{default_root}/{args.task}/{args.domain}/teleop/shards"
    run_tag = f"_{args.run_tag}" if args.run_tag else ""
    stag = _samples_tag(samples)
    ckpt_name = f"{args.task}_{args.domain}_{args.obs}_{args.model}{tactile_tag}{cam_tag}{run_tag}_{stag}"
    ckpt = f"{CKPT_ROOT}/{ckpt_name}"
    tags = json.dumps(
        ["unitree_g1", args.task, args.domain, args.obs, args.model, args.camera_config]
        + (["tactile"] if args.tactile_propio else [])
        + ([cam_tag.strip("_")] if cam_tag else [])
        + (args.wandb_tags if args.wandb_tags else [])
    )

    # Print summary
    gpus = args.gpu.split(",")
    print("=" * 40)
    print(f"Task:        {args.task}  [{args.domain} / {args.camera_config}]")
    tactile_suffix = " + tactile" if args.tactile_propio and args.obs != "vision_propio_tactile" else ""
    print(f"Obs:         {obs_desc}{tactile_suffix}")
    print(f"Cameras:     {camera_names}")
    print("Act:         EE pose (18D) + finger joints (14D) = 32D  [absolute]")
    print(f"Model:       {args.model}")
    print(f"Config:      {config}")
    print(f"Data:        {_dataset_info(data_root)}")
    print(f"Checkpoint:  {ckpt}")
    if args.steps:
        print(f"Steps:       {args.steps}  (= {samples:,} samples)")
    else:
        print(f"Samples:     {samples:,}")
    if args.batch:
        print(f"Batch:       {args.batch} per GPU")
    if args.global_batch:
        print(f"Global batch: {args.global_batch}")
    print("-" * 40)
    if args.sagemaker:
        print("Mode:        SageMaker")
        print(f"User:        {args.sm_user}")
        print(f"Queue:       {args.sm_queue} / {args.sm_instance}")
    else:
        print(f"Mode:        Local ({len(gpus)} GPUs: {args.gpu})")
    print("=" * 40)

    # Build draccus overrides shared by both launch paths
    overrides = [
        "--config_path",
        config,
        "--remote_sync",
        ckpt,
        "--wandb",
        "true",
        "--wandb_project_name",
        args.wandb_project,
        "--total_train_samples",
        str(samples),
        "--wandb_tags",
        tags,
        "--remote_sync_fixed_path",
        REMOTE_SYNC_FIXED_PATH,
        "--data.dataset_manifest",
        json.dumps([f"{data_root}/manifest.jsonl"]),
        "--data.dataset_statistics",
        json.dumps([f"{data_root}/stats.json"]),
        "--data.action_fields",
        json.dumps(ACTION_FIELDS),
        "--data.camera_names",
        json.dumps(camera_names),
    ]
    if proprio_fields:
        overrides += ["--data.proprioception_fields", json.dumps(proprio_fields)]
    else:
        overrides += ["--data.proprioception_fields", json.dumps([])]
    if args.norm_scope is not None:
        overrides += ["--data.normalization.scope", args.norm_scope]
    if args.batch:
        overrides += ["--hparams.per_gpu_batch_size", str(args.batch)]
    if args.global_batch:
        overrides += ["--hparams.global_batch_size", str(args.global_batch)]
    if args.val_samples:
        overrides += [
            "--total_val_samples",
            str(args.val_samples),
            "--data.val_dataset_manifest",
            json.dumps([f"{data_root}/manifest.jsonl"]),
            "--data.val_dataset_statistics",
            json.dumps([f"{data_root}/stats.json"]),
            "--data.val_dataset_weighting",
            json.dumps([1.0]),
        ]

    # Build full command
    if args.sagemaker:
        _verify_sagemaker_credentials(SM_PROFILE, SM_ARN)
        os.environ["AWS_PROFILE"] = SM_PROFILE
        os.environ["AWS_DEFAULT_REGION"] = SM_REGION
        cmd = [
            "uv",
            "run",
            "--group",
            "sagemaker",
            "python",
            "sagemaker/launch_training.py",
            "--sagemaker.user",
            args.sm_user,
            "--sagemaker.profile",
            SM_PROFILE,
            "--sagemaker.region",
            SM_REGION,
            "--sagemaker.arn",
            SM_ARN,
            "--sagemaker.queue_name",
            args.sm_queue,
            "--sagemaker.instance_type",
            args.sm_instance,
            "--sagemaker.priority",
            str(SM_PRIORITY),
        ] + overrides
    else:
        cmd = (
            [f"CUDA_VISIBLE_DEVICES={args.gpu}"]
            + ["uv", "run", "torchrun", f"--nproc_per_node={len(gpus)}", "--nnodes=1", "vla_foundry/main.py"]
            + overrides
        )

    if args.dry_run:
        print("\nCommand (dry-run):")
        print(" \\\n    ".join(cmd))
        return

    # For local runs, CUDA_VISIBLE_DEVICES must be an env var, not an argv token
    if not args.sagemaker:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
        cmd = cmd[1:]  # drop the display token
    else:
        env = os.environ.copy()

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
