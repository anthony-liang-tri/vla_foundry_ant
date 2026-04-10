#!/usr/bin/env python3
"""
Unified evaluation campaign orchestrator.

This script:
1. Spins up a Ray cluster with specified configuration
2. Submits evaluation jobs using ray_policy_runner.py
3. Monitors job progress
4. Collects and aggregates results
5. Generates summary report
6. Optionally tears down the cluster
"""

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import yaml
from ray.job_submission import JobStatus, JobSubmissionClient

# Use local modules instead of anzu-specific imports
from vla_foundry.tri.lbm_eval.eval_campaigns.ray_policy_runner import (
    EVAL_DOCKER_LABEL,
    TaskSpec,
    load_tasks_from_file,
)
from vla_foundry.tri.lbm_eval.eval_campaigns.success_stats import (
    calculate_confidence_interval,
    collect_multi_rollout_success_stats,
)

# Determine repository root - can be overridden via environment variable
_WORKSPACE_DIR = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
_EVAL_CAMPAIGN_ROOT = os.environ.get("EVAL_CAMPAIGN_ROOT")

if _EVAL_CAMPAIGN_ROOT:
    REPO_ROOT = Path(_EVAL_CAMPAIGN_ROOT).resolve()
elif _WORKSPACE_DIR:
    REPO_ROOT = Path(_WORKSPACE_DIR).resolve()
else:
    # Default: this file is at vla_foundry/tri/lbm_eval/eval_campaigns/run_evaluation_campaign.py
    # So repo root is 4 levels up
    REPO_ROOT = Path(__file__).resolve().parents[4]
SUCCESS_CONFIDENCE_LEVEL = 0.9
SUCCESS_METRICS_FILENAME = "success_metrics.json"


@dataclass
class CampaignConfig:
    """Configuration for an evaluation campaign."""

    # Campaign metadata
    campaign_name: str
    description: str

    # Cluster configuration
    cluster_config_file: Path
    num_workers: int
    aws_profile: str
    inference_aws_profile: str
    owner_email: str  # Required: your @tri.global email for AWS resource tagging
    cluster_name: str  # Unique cluster name (derived from owner_email if not specified)

    # Evaluation configuration
    tasks_file: Path | None
    checkpoints: list[str] | None
    task: str | None
    num_samples: int
    start_index: int
    max_samples_per_job: int

    # Docker configuration
    docker_image: str

    # Ray configuration
    launch_config_file: str | None
    launch_scenario: str
    launch_script: str
    num_flow_steps: int
    open_loop_steps: int
    device: str
    launch_cuda_visible_devices: str | None
    entrypoint_num_cpus: float | None
    entrypoint_num_gpus: float | None
    entrypoint_num_gpus: float | None
    entrypoint_memory: int | None
    jobs_per_gpu: float | None
    max_retries: int

    # Output configuration
    results_dir: Path
    save_job_logs: bool

    # Cluster management
    teardown_on_completion: bool
    teardown_on_failure: bool
    skip_cleanup: bool

    # vla_foundry source management
    vla_repo_url: str | None
    vla_repo_ref: str | None
    vla_repo_local_dir: Path | None
    vla_repo_use_remote: bool
    vla_repo_local_source: Path | None
    vla_repo_remote_dir: str | None
    vla_repo_mount_target: str = "/opt/vla_foundry"

    # Custom inference command configuration (for alternative policy repos like LBM)
    inference_script: str | None = None
    inference_script_args: str | None = None
    inference_cmd_override: str | None = None
    download_rollouts: bool = False
    download_episode_pkls: bool = False

    # Script mounting configuration
    # When True, use scripts baked into Docker image instead of mounting from cluster
    skip_mount_scripts: bool = False

    # S3 output path customization
    # Optional subfolder to insert in S3 path: {checkpoint}/evaluation/{subfolder}/{task}/rollouts/
    # Useful for organizing results from different evaluation campaigns (e.g., "oss", "stage3")
    evaluation_subfolder: str | None = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CampaignConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        vla_cfg = data.get("vla_foundry", {})

        def resolve_repo_relative(path_str: str) -> Path:
            path = Path(path_str).expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            return path

        local_source_path: Path | None = None
        if "local_source" in vla_cfg and vla_cfg["local_source"] is not None:
            local_source_path = resolve_repo_relative(vla_cfg["local_source"])

        use_remote_repo = vla_cfg.get(
            "use_remote_repo",
            not local_source_path,
        )
        if not use_remote_repo and local_source_path is None:
            local_source_path = REPO_ROOT

        owner_email = data["cluster"].get("owner_email", "")
        if not owner_email:
            raise ValueError(
                "\n" + "=" * 70 + "\n"
                "ERROR: 'owner_email' is required in the campaign config!\n"
                "=" * 70 + "\n\n"
                "AWS resources require owner tagging for cost attribution.\n"
                "Please add your email to the campaign YAML under 'cluster':\n\n"
                "  cluster:\n"
                '    owner_email: "your.name@tri.global"\n\n'
                "=" * 70
            )

        # Derive cluster name from owner email if not specified
        # This ensures each user gets their own isolated cluster
        cluster_name = data["cluster"].get("cluster_name", "")
        if not cluster_name:
            # Extract username from email (e.g., "jean.mercat@tri.global" -> "jean_mercat")
            username = owner_email.split("@")[0].replace(".", "_").replace("-", "_")
            cluster_name = f"lbm_eval_{username}"

        return cls(
            campaign_name=data["campaign_name"],
            description=data.get("description", ""),
            cluster_config_file=resolve_repo_relative(data["cluster"]["config_file"]),
            num_workers=data["cluster"]["num_workers"],
            aws_profile=data["cluster"]["aws_profile"],
            inference_aws_profile=data["evaluation"].get("inference_aws_profile", "sagemaker"),
            owner_email=owner_email,
            cluster_name=cluster_name,
            tasks_file=resolve_repo_relative(data["evaluation"]["tasks_file"])
            if "tasks_file" in data["evaluation"]
            else None,
            checkpoints=data["evaluation"].get("checkpoints"),
            task=data["evaluation"].get("task"),
            num_samples=data["evaluation"]["num_samples"],
            start_index=data["evaluation"]["start_index"],
            max_samples_per_job=data["evaluation"].get("max_samples_per_job", 20),
            docker_image=data["docker"]["image"],
            launch_config_file=data["evaluation"].get("launch_config_file"),
            launch_scenario=data["evaluation"]["launch_scenario"],
            launch_script=data["evaluation"]["launch_script"],
            num_flow_steps=data["evaluation"]["num_flow_steps"],
            open_loop_steps=data["evaluation"]["open_loop_steps"],
            device=data["evaluation"]["device"],
            launch_cuda_visible_devices=data["evaluation"].get("launch_cuda_visible_devices"),
            entrypoint_num_cpus=data["evaluation"].get("entrypoint_num_cpus"),
            entrypoint_num_gpus=data["evaluation"].get("entrypoint_num_gpus"),
            entrypoint_memory=data["evaluation"].get("entrypoint_memory"),
            jobs_per_gpu=data["evaluation"].get("jobs_per_gpu"),
            max_retries=data["evaluation"].get("max_retries", 0),
            results_dir=resolve_repo_relative(data["output"]["results_dir"]),
            save_job_logs=data["output"].get("save_job_logs", False),
            teardown_on_completion=data["cluster"].get("teardown_on_completion", False),
            teardown_on_failure=data["cluster"].get("teardown_on_failure", False),
            vla_repo_url=vla_cfg.get("repo_url", "git@github.com:TRI-ML/vla_foundry.git"),
            vla_repo_ref=vla_cfg.get("ref", "main"),
            vla_repo_local_dir=Path(vla_cfg.get("local_dir", "~/.cache/vla_foundry/ray_mount")).expanduser(),
            vla_repo_use_remote=use_remote_repo,
            vla_repo_local_source=local_source_path,
            vla_repo_remote_dir=vla_cfg.get("remote_dir", "/home/ubuntu/vla_foundry_mount"),
            vla_repo_mount_target=vla_cfg.get("mount_target", "/opt/vla_foundry"),
            skip_cleanup=data.get("skip_cleanup", False),
            # Custom inference command configuration (optional)
            inference_script=data.get("inference", {}).get("script"),
            inference_script_args=data.get("inference", {}).get("args"),
            inference_cmd_override=data.get("inference", {}).get("cmd_override"),
            # Script mounting configuration (for using Docker image scripts instead of cluster scripts)
            skip_mount_scripts=data.get("docker", {}).get("skip_mount_scripts", False),
            # S3 output path customization
            evaluation_subfolder=data.get("evaluation", {}).get("evaluation_subfolder"),
        )


class EvaluationCampaign:
    """Orchestrates an evaluation campaign."""

    STATE_FILENAME = "campaign_state.json"

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.cluster_url: str | None = None
        self.submission_ids: list[str] = []
        self.runner_env: dict[str, str] = dict(os.environ)
        self.local_vla_repo_dir: Path | None = None
        self.remote_vla_repo_dir: str | None = None
        self.success_metrics: dict[str, Any] | None = None
        self._cluster_auth: tuple[str, Path] | None = None
        # Maps vla_ref (or None for default) to (local_dir, remote_dir)
        self.vla_ref_dirs: dict[str | None, tuple[Path, str]] = {}
        # Loaded tasks from tasks_file (for per-task vla_ref support)
        self._loaded_tasks: list[TaskSpec] | None = None
        # Track completed jobs for state persistence
        self.completed: dict[str, str] = {}  # submission_id -> status
        self._last_state_save = 0.0
        self._state_save_interval = 30  # Save state every 30 seconds
        self._cluster_unreachable_count = 0
        self._max_cluster_unreachable = 10  # Max consecutive failures before aborting
        # Expanded cluster config with placeholders replaced (saved to results_dir)
        self._expanded_cluster_config_path: Path | None = None

    @property
    def _cluster_config_for_ray(self) -> Path:
        """Get the cluster config path to use for Ray CLI commands.

        Returns the expanded config (with placeholders replaced) if available,
        otherwise falls back to the original config file.
        """
        if self._expanded_cluster_config_path:
            return self._expanded_cluster_config_path
        print("  WARNING: _expanded_cluster_config_path is not set, using original config file")
        return self.config.cluster_config_file

    def _create_expanded_cluster_config(self):
        """Create expanded cluster config with placeholders replaced.

        This writes the config to results_dir/cluster_config_expanded.yaml
        and sets self._expanded_cluster_config_path.
        """
        print("  [config] Creating expanded cluster config...")
        print(f"  [config]   Source: {self.config.cluster_config_file}")
        print(f"  [config]   Cluster name: {self.config.cluster_name}")
        print(f"  [config]   Owner email: {self.config.owner_email}")

        with open(self.config.cluster_config_file) as f:
            cluster_config_content = f.read()

        # Expand placeholders
        expanded_config = cluster_config_content.replace("${OWNER_EMAIL}", self.config.owner_email)
        expanded_config = expanded_config.replace("${CLUSTER_NAME}", self.config.cluster_name)

        # Write to results dir
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self._expanded_cluster_config_path = self.config.results_dir / "cluster_config_expanded.yaml"
        with open(self._expanded_cluster_config_path, "w") as f:
            f.write(expanded_config)
        print(f"  [config] Created expanded cluster config: {self._expanded_cluster_config_path}")

        # Verify the expansion worked
        with open(self._expanded_cluster_config_path) as f:
            first_lines = f.read(500)
        if "${CLUSTER_NAME}" in first_lines:
            print("  [config] ERROR: Placeholder not expanded!")
        else:
            print("  [config] Verified: placeholders expanded correctly")

    def _get_state_file_path(self) -> Path:
        """Get the path to the campaign state file."""
        return self.config.results_dir / self.STATE_FILENAME

    def save_state(self, force: bool = False):
        """Save current campaign state to disk for recovery."""
        now = time.time()
        if not force and (now - self._last_state_save) < self._state_save_interval:
            return

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "campaign_name": self.config.campaign_name,
            "cluster_url": self.cluster_url,
            "submission_ids": self.submission_ids,
            "completed": {k: str(v) for k, v in self.completed.items()},
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "config_file": str(self.config.cluster_config_file),
        }
        state_file = self._get_state_file_path()
        try:
            # Write to temp file first, then rename for atomic write
            temp_file = state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)
            temp_file.rename(state_file)
            self._last_state_save = now
            print(f"  [state] Saved campaign state to {state_file}")
        except Exception as e:
            print(f"  [state] WARNING: Failed to save state: {e}")

    def load_state(self) -> dict[str, Any] | None:
        """Load campaign state from disk if it exists."""
        state_file = self._get_state_file_path()
        if not state_file.exists():
            return None
        try:
            with open(state_file) as f:
                state = json.load(f)
            print(f"  [state] Loaded campaign state from {state_file}")
            print(f"  [state]   Timestamp: {state.get('timestamp', 'unknown')}")
            print(f"  [state]   Submissions: {len(state.get('submission_ids', []))}")
            print(f"  [state]   Completed: {len(state.get('completed', {}))}")
            return state
        except Exception as e:
            print(f"  [state] WARNING: Failed to load state: {e}")
            return None

    def _write_cluster_info(self):
        """Write cluster info to an easily accessible file and print prominent summary."""
        if not self.cluster_url:
            return

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        info_file = self.config.results_dir / "CLUSTER_INFO.txt"

        # Extract head IP from URL
        head_ip = self.cluster_url.split("//")[-1].split(":")[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        info_content = f"""
================================================================================
                         RAY CLUSTER INFORMATION
================================================================================
  Updated:        {timestamp}
  Campaign:       {self.config.campaign_name}

  Dashboard URL:  {self.cluster_url}
  Head IP:        {head_ip}
  Results Dir:    {self.config.results_dir}

--------------------------------------------------------------------------------
  USEFUL COMMANDS
--------------------------------------------------------------------------------

  # View job status:
  curl -s "{self.cluster_url}/api/jobs/" | python3 -m json.tool

  # Port-forward dashboard (if not accessible directly):
  ray dashboard {self._cluster_config_for_ray}

  # Tear down cluster:
  AWS_PROFILE={self.config.aws_profile} ray down {self._cluster_config_for_ray} --yes

  # Attach to head node:
  ray attach {self._cluster_config_for_ray}

================================================================================
"""
        try:
            with open(info_file, "w") as f:
                f.write(info_content)
            print(info_content)
            print(f"  Cluster info saved to: {info_file}")
            print()
        except Exception as e:
            print(f"  WARNING: Failed to write cluster info file: {e}")

    def check_cluster_health(self) -> tuple[bool, str]:
        """Check if the Ray cluster is reachable and healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        if not self.cluster_url:
            return False, "No cluster URL configured"

        import requests

        try:
            # Check dashboard endpoint
            resp = requests.get(f"{self.cluster_url}/api/cluster_status", timeout=10)
            if resp.status_code == 200:
                self._cluster_unreachable_count = 0
                return True, "Cluster is healthy"
            else:
                self._cluster_unreachable_count += 1
                return False, f"Dashboard returned status {resp.status_code}"
        except requests.exceptions.ConnectionError as e:
            self._cluster_unreachable_count += 1
            return False, f"Connection error: {e}"
        except requests.exceptions.Timeout:
            self._cluster_unreachable_count += 1
            return False, "Connection timed out"
        except Exception as e:
            self._cluster_unreachable_count += 1
            return False, f"Unknown error: {e}"

    def is_cluster_lost(self) -> bool:
        """Check if we've lost connection to the cluster for too long."""
        return self._cluster_unreachable_count >= self._max_cluster_unreachable

    def run(self, skip_cluster_startup=False, resume: bool = False) -> int:
        """Execute the full evaluation campaign.

        Args:
            skip_cluster_startup: Skip the cluster startup step (use existing cluster)
            resume: Resume from saved state if available
        """
        print("=" * 80)
        print(f"Starting Evaluation Campaign: {self.config.campaign_name}")
        print(f"Description: {self.config.description}")
        print("=" * 80)
        print()

        # Check for existing state if resume requested
        saved_state = None
        if resume:
            saved_state = self.load_state()
            if saved_state:
                print()
                print("  [resume] Found saved campaign state, will resume from checkpoint")
                self.cluster_url = saved_state.get("cluster_url")
                # Restore completed jobs
                for sub_id, status_str in saved_state.get("completed", {}).items():
                    self.completed[sub_id] = status_str
                print(f"  [resume] Restored {len(self.completed)} completed job statuses")
                self._write_cluster_info()
            else:
                print("  [resume] No saved state found, starting fresh campaign")

        # Always recreate expanded cluster config so campaign YAML edits
        # (e.g. cluster name / owner email) are applied immediately.
        self._create_expanded_cluster_config()

        # Validate all vla_refs upfront before doing anything expensive
        self._validate_vla_refs()

        # Prepare local vla_foundry checkout before touching the cluster so the latest
        # code is ready to rsync as soon as the cluster becomes available.
        self.prepare_vla_foundry_mount()

        # Setup local config files (secrets.env) required by Ray file_mounts
        self.setup_local_config_files()

        try:
            # Step 1: Start cluster (unless skip requested)
            if skip_cluster_startup:
                print("[1/5] Skipping cluster startup (--skip-cluster-startup enabled)")
                print("  Using existing cluster")
                self.cluster_url = self.get_cluster_url()
                print(f"  Dashboard: {self.cluster_url}")
                print()
                self._write_cluster_info()
            else:
                self.start_cluster()

            # Ensure AWS credentials are available to downstream containers
            self.prepare_inference_credentials()

            # Stage vla_foundry repo for host mounts if configured
            self.deploy_vla_foundry_repo()

            # Stage anzu_vla_foundry files (launch_sim.sh) to all nodes
            self.deploy_anzu_repo()

            # Stop any lingering jobs from previous runs to avoid conflicts.
            if not self.config.skip_cleanup:
                self._terminate_inference_containers()
            else:
                print("  Skipping container cleanup (--skip-cleanup enabled)")

            # Step 2: Submit evaluation jobs
            self.submit_jobs()

            # Step 3: Monitor progress
            self.monitor_jobs()

            # Step 4: Collect results
            self.collect_results()

            # Step 5: Generate report
            self.generate_report()

            print()
            print("=" * 80)
            print("Campaign completed successfully!")
            print(f"Results saved to: {self.config.results_dir}")
            print("=" * 80)

            # Teardown if requested
            if self.config.teardown_on_completion:
                self.teardown_cluster()
            else:
                # Stop workers even if not doing full teardown to avoid orphaned workers
                self.stop_cluster_workers()
                print("\nHead node left running. To tear down manually:")
                print(f"  AWS_PROFILE={self.config.aws_profile} ray down {self.config.cluster_config_file}")

            return 0

        except KeyboardInterrupt:
            print("\n")
            print("=" * 80)
            print("Campaign interrupted by user (Ctrl+C)")
            print("=" * 80)
            self.save_state(force=True)
            print(f"\nState saved to: {self._get_state_file_path()}")
            print("To resume later, run with --resume flag")
            if self.config.teardown_on_failure:
                self.teardown_cluster()
            else:
                # Stop workers to avoid orphaned instances
                self.stop_cluster_workers()
            return 130  # Standard exit code for SIGINT

        except Exception as e:
            print(f"\nERROR: Campaign failed: {e}", file=sys.stderr)
            # Save state before teardown so we can analyze what happened
            self.save_state(force=True)
            print(f"\n[recovery] State saved to: {self._get_state_file_path()}")
            print(f"[recovery] Completed jobs: {len(self.completed)}/{len(self.submission_ids)}")
            if self.submission_ids:
                print("[recovery] To check job status manually:")
                print(f"  curl {self.cluster_url}/api/jobs/")
            if self.config.teardown_on_failure:
                self.teardown_cluster()
            else:
                # Stop workers to avoid orphaned instances
                self.stop_cluster_workers()
            return 1

    def setup_local_config_files(self):
        """Validate required files exist for Ray file_mounts.

        The Ray cluster config mounts these files directly from the repo:
        - secrets.env (repo root, gitignored)
        - run_inference_bundle.sh (cluster_scripts/)
        - launch_sim.sh (cluster_scripts/)
        """
        cluster_scripts_dir = REPO_ROOT / "vla_foundry" / "tri" / "lbm_eval" / "eval_campaigns" / "cluster_scripts"

        # Check cluster scripts exist
        for script_name in ["run_inference_bundle.sh", "launch_sim.sh"]:
            script_path = cluster_scripts_dir / script_name
            if not script_path.exists():
                raise FileNotFoundError(f"Required script not found: {script_path}")

        # Check secrets.env exists (create empty if not)
        secrets_path = REPO_ROOT / "secrets.env"
        if not secrets_path.exists():
            secrets_path.touch()
            print(f"[prep] Created empty secrets.env at {secrets_path}")
            print("       (Add HF_TOKEN and WANDB_API_KEY if needed)")
        else:
            print(f"[prep] Found secrets.env at {secrets_path}")

    def _wait_for_stopping_instances(self):
        """Wait for any cluster instances in 'stopping' state to become 'stopped'.

        Ray cannot properly manage instances that are transitioning. This ensures
        a clean state before ray up is called.
        """
        try:
            session = boto3.Session(profile_name=self.config.aws_profile)
            ec2 = session.client("ec2", region_name="us-east-1")

            # Find instances in 'stopping' state for this cluster
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:ray-cluster-name", "Values": [self.config.cluster_name]},
                    {"Name": "instance-state-name", "Values": ["stopping"]},
                ]
            )

            stopping_ids = []
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    stopping_ids.append(instance["InstanceId"])

            if stopping_ids:
                print(f"  Waiting for {len(stopping_ids)} instance(s) to finish stopping...")
                waiter = ec2.get_waiter("instance_stopped")
                waiter.wait(
                    InstanceIds=stopping_ids,
                    WaiterConfig={"Delay": 5, "MaxAttempts": 60},  # 5 min max
                )
                print("  All instances are now stopped")
        except Exception as e:
            print(f"  Warning: Could not check for stopping instances: {e}")

    def _terminate_stopped_head_nodes(self):
        """Terminate any stopped head nodes to force fresh creation.

        Reusing stopped head nodes with cache_stopped_nodes causes issues where
        the head node stops itself after ray up. Workers can still be reused.
        """
        try:
            session = boto3.Session(profile_name=self.config.aws_profile)
            ec2 = session.client("ec2", region_name="us-east-1")

            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:ray-cluster-name", "Values": [self.config.cluster_name]},
                    {"Name": "tag:ray-node-type", "Values": ["head"]},
                    {"Name": "instance-state-name", "Values": ["stopped"]},
                ]
            )

            head_ids = []
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    head_ids.append(instance["InstanceId"])

            if head_ids:
                print(f"  Terminating {len(head_ids)} stopped head node(s) to force fresh creation...")
                ec2.terminate_instances(InstanceIds=head_ids)
                print(f"  Terminated: {head_ids}")
        except Exception as e:
            print(f"  Warning: Could not terminate stopped head nodes: {e}")

    def start_cluster(self):
        """Start the Ray cluster."""
        print("[1/5] Starting Ray cluster...")
        print(f"  Config: {self.config.cluster_config_file}")
        print(f"  Workers: {self.config.num_workers}")
        print(f"  Owner: {self.config.owner_email}")
        print(f"  Cluster name: {self.config.cluster_name}")
        print()

        # Wait for any instances in 'stopping' state before ray up
        self._wait_for_stopping_instances()

        # Terminate stopped head nodes - reusing them causes issues
        self._terminate_stopped_head_nodes()

        env = {
            "AWS_PROFILE": self.config.aws_profile,
            "OWNER_EMAIL": self.config.owner_email,
            "CLUSTER_NAME": self.config.cluster_name,
        }

        # Ensure expanded config exists (may have been created in run())
        if not self._expanded_cluster_config_path:
            self._create_expanded_cluster_config()

        # Environment for invoking Ray CLI locally
        merged_env = {**os.environ, **env}

        cmd = [
            "ray",
            "up",
            str(self._expanded_cluster_config_path),
            "--yes",
            "--no-config-cache",  # Avoid stale cached provider config
        ]

        result = subprocess.run(
            cmd,
            env=merged_env,
            cwd=REPO_ROOT,
            capture_output=False,
        )

        if result.returncode != 0:
            raise RuntimeError("Failed to start cluster")

        # Get cluster head IP
        self.cluster_url = self.get_cluster_url()
        print("\n✓ Cluster started successfully")
        print(f"  Dashboard: {self.cluster_url}")
        print()
        self._write_cluster_info()

    def get_cluster_url(self, max_retries: int = 10, initial_delay: float = 20.0) -> str:
        """Get the Ray dashboard URL with retry logic for reused instances."""
        cmd = [
            "ray",
            "get-head-ip",
            str(self._cluster_config_for_ray),
        ]

        env = {"AWS_PROFILE": self.config.aws_profile}

        last_error = None
        delay = initial_delay
        for attempt in range(max_retries):
            result = subprocess.run(
                cmd,
                env={**os.environ, **env},
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                break

            last_error = result.stderr
            if attempt < max_retries - 1:
                print(f"  Waiting for head node registration (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
        else:
            raise RuntimeError(f"Failed to get cluster head IP: {last_error}")

        head_ip = result.stdout.strip()

        # Ensure we have just the IP, no extra text
        # The output might have multiple lines, take the last non-empty line
        if "\n" in head_ip:
            lines = [line.strip() for line in head_ip.split("\n") if line.strip()]
            head_ip = lines[-1] if lines else head_ip

        # Remove any non-IP characters
        ip_match = re.search(r"\d+\.\d+\.\d+\.\d+", head_ip)
        if ip_match:
            head_ip = ip_match.group(0)
        else:
            raise RuntimeError(f"Could not extract valid IP from: {head_ip}")

        print(f"  Head IP: {head_ip}")
        url = f"http://{head_ip}:8265"
        print(f"  Dashboard URL: {url}")
        return url

    def submit_jobs(self):
        """Submit evaluation jobs to the cluster."""
        print("[2/5] Submitting evaluation jobs...")
        print()

        runner_path = REPO_ROOT / "vla_foundry" / "tri" / "lbm_eval" / "eval_campaigns" / "ray_policy_runner.py"
        print(f"Executing runner from: {runner_path}")

        # Load tasks and resolve mount paths for each
        tasks = self._load_tasks_if_needed()

        if not tasks and self.config.tasks_file:
            # Fall back to original tasks file if no tasks loaded
            self._submit_all_jobs(runner_path, self.config.tasks_file)
        else:
            # Create a single tasks file with mount_src resolved per task
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for task in tasks:
                    # Get the mount path for this task's vla_ref
                    if task.vla_ref in self.vla_ref_dirs:
                        _, mount_src = self.vla_ref_dirs[task.vla_ref]
                    elif None in self.vla_ref_dirs:
                        _, mount_src = self.vla_ref_dirs[None]
                    else:
                        raise RuntimeError(
                            f"vla_ref '{task.vla_ref}' not found in prepared refs. "
                            f"Available refs: {list(self.vla_ref_dirs.keys())}."
                        )

                    # Write as 5-element tuple with mount_src
                    vla_ref_str = task.vla_ref if task.vla_ref else ""
                    demo_indices_str = task.demo_indices if getattr(task, "demo_indices", None) else ""
                    f.write(
                        f'("{task.name}", "{task.task_name}", "{task.checkpoint}", '
                        f'"{vla_ref_str}", "{mount_src}", "{demo_indices_str}"),\n'
                    )
                temp_tasks_file = Path(f.name)

            try:
                print(f"Submitting all {len(tasks)} task(s) in a single batch")
                self._submit_all_jobs(runner_path, temp_tasks_file)
            finally:
                temp_tasks_file.unlink(missing_ok=True)

        print(f"\n✓ Submitted {len(self.submission_ids)} total jobs")
        print()

    def _submit_all_jobs(
        self,
        runner_path: Path,
        tasks_file: Path,
    ):
        """Submit all jobs from a tasks file (mount_src embedded per task)."""
        cmd = [
            "python3",
            str(runner_path),
            "--cluster-url",
            self.cluster_url,
            "--image",
            self.config.docker_image,
            "--distribute",
            "--num-samples",
            str(self.config.num_samples),
            "--max-samples-per-job",
            str(self.config.max_samples_per_job),
            "--num-workers",
            str(self.config.num_workers),
            "--start-index",
            str(self.config.start_index),
            "--launch-scenario",
            self.config.launch_scenario,
            "--launch-script",
            self.config.launch_script,
            "--num-flow-steps",
            str(self.config.num_flow_steps),
            "--open-loop-steps",
            str(self.config.open_loop_steps),
            "--device",
            self.config.device,
            # Mount target is same for all tasks, only mount_src varies
            "--vla-foundry-mount-target",
            self.config.vla_repo_mount_target,
            "--disable-vla-foundry-auto-update",
        ]
        # Only mount scripts from cluster if not using Docker image scripts
        if not self.config.skip_mount_scripts:
            cmd.extend(
                [
                    "--mount-local-run-bundle",
                    "--cluster-repo-root",
                    "/home/ubuntu/anzu_vla_foundry",
                ]
            )
        if self.config.launch_config_file:
            cmd.extend(["--launch-config-file", self.config.launch_config_file])
        if self.config.launch_cuda_visible_devices:
            cmd.extend(
                [
                    "--launch-cuda-visible-devices",
                    self.config.launch_cuda_visible_devices,
                ]
            )
        if self.config.entrypoint_num_cpus is not None:
            cmd.extend(
                [
                    "--entrypoint-num-cpus",
                    str(self.config.entrypoint_num_cpus),
                ]
            )
        if self.config.entrypoint_num_gpus is not None:
            cmd.extend(
                [
                    "--entrypoint-num-gpus",
                    str(self.config.entrypoint_num_gpus),
                ]
            )
        if self.config.entrypoint_memory is not None:
            cmd.extend(
                [
                    "--entrypoint-memory",
                    str(self.config.entrypoint_memory),
                ]
            )
        if self.config.jobs_per_gpu is not None:
            cmd.extend(
                [
                    "--jobs-per-gpu",
                    str(self.config.jobs_per_gpu),
                ]
            )
        if self.config.max_retries > 0:
            cmd.extend(
                [
                    "--max-retries",
                    str(self.config.max_retries),
                ]
            )

        # Custom inference command configuration
        if self.config.inference_script:
            cmd.extend(["--inference-script", self.config.inference_script])
        if self.config.inference_script_args:
            cmd.extend(["--inference-script-args", self.config.inference_script_args])
        if self.config.inference_cmd_override:
            cmd.extend(["--inference-cmd-override", self.config.inference_cmd_override])

        if self.config.inference_aws_profile:
            cmd.extend(["--aws-profile", self.config.inference_aws_profile])

        # S3 output path customization
        if self.config.evaluation_subfolder:
            cmd.extend(["--evaluation-subfolder", self.config.evaluation_subfolder])

        cmd.extend(["--tasks-file", str(tasks_file)])

        # Stream output while capturing it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.runner_env,
            bufsize=1,
        )

        stdout_lines = []
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError("Failed to submit jobs")

        # Parse submission IDs from output
        for line in stdout_lines:
            if "-> submission id" in line:
                sub_id = line.split("-> submission id")[1].strip()
                self.submission_ids.append(sub_id)

    def _submit_jobs_for_ref(
        self,
        runner_path: Path,
        vla_ref: str | None,
        tasks_file: Path | None,
    ):
        """Submit jobs for a specific vla_ref (legacy method, kept for compatibility)."""
        cmd = [
            "python3",
            str(runner_path),
            "--cluster-url",
            self.cluster_url,
            "--image",
            self.config.docker_image,
            "--distribute",
            "--num-samples",
            str(self.config.num_samples),
            "--max-samples-per-job",
            str(self.config.max_samples_per_job),
            "--num-workers",
            str(self.config.num_workers),
            "--start-index",
            str(self.config.start_index),
            "--launch-scenario",
            self.config.launch_scenario,
            "--launch-script",
            self.config.launch_script,
            "--num-flow-steps",
            str(self.config.num_flow_steps),
            "--open-loop-steps",
            str(self.config.open_loop_steps),
            "--device",
            self.config.device,
        ]
        if self.config.launch_config_file:
            cmd.extend(["--launch-config-file", self.config.launch_config_file])
        if self.config.launch_cuda_visible_devices:
            cmd.extend(
                [
                    "--launch-cuda-visible-devices",
                    self.config.launch_cuda_visible_devices,
                ]
            )
        if self.config.entrypoint_num_cpus is not None:
            cmd.extend(
                [
                    "--entrypoint-num-cpus",
                    str(self.config.entrypoint_num_cpus),
                ]
            )
        if self.config.entrypoint_num_gpus is not None:
            cmd.extend(
                [
                    "--entrypoint-num-gpus",
                    str(self.config.entrypoint_num_gpus),
                ]
            )
        if self.config.entrypoint_memory is not None:
            cmd.extend(
                [
                    "--entrypoint-memory",
                    str(self.config.entrypoint_memory),
                ]
            )
        if self.config.jobs_per_gpu is not None:
            cmd.extend(
                [
                    "--jobs-per-gpu",
                    str(self.config.jobs_per_gpu),
                ]
            )
        if self.config.max_retries > 0:
            cmd.extend(
                [
                    "--max-retries",
                    str(self.config.max_retries),
                ]
            )

        # Get the correct vla_foundry mount src for this ref - NO FALLBACK
        # Each ref must be explicitly prepared via _prepare_vla_ref()
        if vla_ref not in self.vla_ref_dirs:
            raise RuntimeError(
                f"vla_ref '{vla_ref}' not found in prepared refs. "
                f"Available refs: {list(self.vla_ref_dirs.keys())}. "
                f"Ensure all refs are prepared before submitting jobs."
            )
        _, remote_dir = self.vla_ref_dirs[vla_ref]

        if remote_dir:
            cmd.extend(
                [
                    "--vla-foundry-mount-src",
                    remote_dir,
                    "--vla-foundry-mount-target",
                    self.config.vla_repo_mount_target,
                    "--disable-vla-foundry-auto-update",
                ]
            )
            # Only mount scripts from cluster if not using Docker image scripts
            if not self.config.skip_mount_scripts:
                cmd.extend(
                    [
                        "--mount-local-run-bundle",
                        "--cluster-repo-root",
                        "/home/ubuntu/anzu_vla_foundry",
                    ]
                )

        # Custom inference command configuration
        if self.config.inference_script:
            cmd.extend(["--inference-script", self.config.inference_script])
        if self.config.inference_script_args:
            cmd.extend(["--inference-script-args", self.config.inference_script_args])
        if self.config.inference_cmd_override:
            cmd.extend(["--inference-cmd-override", self.config.inference_cmd_override])

        if self.config.inference_aws_profile:
            cmd.extend(["--aws-profile", self.config.inference_aws_profile])

        # Add tasks file or checkpoints
        if tasks_file:
            cmd.extend(["--tasks-file", str(tasks_file)])
        elif self.config.checkpoints and self.config.task:
            cmd.extend(["--checkpoints"] + self.config.checkpoints)
            cmd.extend(["--task", self.config.task])
        else:
            raise ValueError("Must specify either tasks_file or (checkpoints + task)")

        # Stream output while capturing it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.runner_env,
            bufsize=1,
        )

        stdout_lines = []
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError("Failed to submit jobs")

        # Parse submission IDs from output
        for line in stdout_lines:
            if "-> submission id" in line:
                sub_id = line.split("-> submission id")[1].strip()
                self.submission_ids.append(sub_id)

    def run_subprocess(self, cmd, cwd=None, env=None):
        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")

    def prepare_inference_credentials(self):
        """Export short-lived AWS credentials for inference containers."""
        profile = self.config.inference_aws_profile
        if not profile:
            return

        print(f"[cred] Exporting AWS credentials for inference profile '{profile}'")
        try:
            session = boto3.Session(profile_name=profile)
            credentials = session.get_credentials()
            if credentials is None:
                raise RuntimeError(
                    f"Unable to load AWS credentials for profile '{profile}'. "
                    "Run 'aws sso login' for that profile and retry."
                )
            frozen = credentials.get_frozen_credentials()
        except Exception as exc:
            raise RuntimeError(f"Failed to export AWS credentials for profile '{profile}': {exc}") from exc

        region = (
            session.region_name
            or self.runner_env.get("AWS_REGION")
            or self.runner_env.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        self.runner_env["AWS_ACCESS_KEY_ID"] = frozen.access_key
        self.runner_env["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
        if frozen.token:
            self.runner_env["AWS_SESSION_TOKEN"] = frozen.token
        self.runner_env["AWS_REGION"] = region
        self.runner_env["AWS_DEFAULT_REGION"] = region
        print("[cred] AWS credentials exported for distributed inference")

    def prepare_vla_foundry_mount(self):
        """Prepare local vla_foundry directory structure for cluster sync."""
        if not self.config.vla_repo_remote_dir:
            return

        if self.config.vla_repo_use_remote:
            # For remote repos, just ensure parent directory exists
            # Actual cloning to subdirectories happens in _prepare_vla_ref()
            if not self.config.vla_repo_local_dir:
                raise RuntimeError("vla_repo_local_dir must be set when use_remote_repo is true.")
            repo_dir = self.config.vla_repo_local_dir
            repo_dir.mkdir(parents=True, exist_ok=True)
            print(f"[prep] vla_foundry parent directory ready at {repo_dir}")
            self.local_vla_repo_dir = repo_dir
            return
        else:
            repo_dir = self.config.vla_repo_local_source if self.config.vla_repo_local_source else REPO_ROOT
            repo_dir = repo_dir.resolve()
            print(f"[prep] Using local vla_foundry workspace at {repo_dir}")
            if not (repo_dir / ".git").exists():
                raise RuntimeError(f"Local vla_foundry workspace at {repo_dir} is not a git repository.")
            # Warn if workspace has uncommitted changes; the user is responsible for keeping it updated.
            try:
                status = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                )
                if status.returncode == 0 and status.stdout.strip():
                    print(
                        "[prep] WARNING: Local vla_foundry workspace has uncommitted changes; "
                        "those will be synced to the cluster."
                    )
            except FileNotFoundError:
                pass

        self.local_vla_repo_dir = repo_dir
        print(f"[prep] Local vla_foundry workspace ready at {repo_dir}")

    def deploy_vla_foundry_repo(self):
        """Ensure the prepared vla_foundry checkout is synced onto the Ray cluster."""
        if not self.config.vla_repo_remote_dir:
            return
        if not self.config.vla_repo_use_remote:
            # Using local workspace - just set the paths
            local_dir = self.config.vla_repo_local_source or REPO_ROOT
            self.vla_ref_dirs[None] = (local_dir, self.config.vla_repo_remote_dir)
            self.local_vla_repo_dir = local_dir
            self.remote_vla_repo_dir = self.config.vla_repo_remote_dir
            self.reset_remote_vla_repo_dir()
            self.sync_vla_repo_to_cluster()
            return

        # Prepare ALL refs locally first (as subdirectories of the file_mounts path)
        # This ensures file_mounts syncs everything to new workers automatically
        unique_refs = self._get_unique_vla_refs()
        if not unique_refs:
            unique_refs = [None]  # At least the default ref

        print(f"[prep] Preparing {len(unique_refs)} vla_foundry ref(s) locally")
        for ref in unique_refs:
            ref_display = ref if ref else "(campaign default)"
            print(f"[prep] Preparing ref: {ref_display}")
            self._prepare_vla_ref(ref)

        # Set the default ref's remote path for backward compatibility
        if None in self.vla_ref_dirs:
            _, self.remote_vla_repo_dir = self.vla_ref_dirs[None]
        self.local_vla_repo_dir = self.config.vla_repo_local_dir

        # Reset and sync the PARENT directory (contains all ref subdirectories)
        # This syncs to existing nodes; new workers get files via file_mounts
        self.reset_remote_vla_repo_dir()
        self.sync_vla_repo_to_cluster()

    def reset_remote_vla_repo_dir(self):
        """Clear the remote vla_foundry directory on all nodes before syncing."""
        remote_path = self.config.vla_repo_remote_dir
        if not remote_path:
            return
        print(f"[prep] Resetting remote vla_foundry directory at {remote_path} on all nodes")
        reset_script = (
            f"sudo rm -rf {remote_path} && sudo mkdir -p {remote_path} && sudo chown ubuntu:ubuntu {remote_path}"
        )
        self._run_bash_on_all_nodes(reset_script, f"Reset {remote_path}")

    def resolve_git_ref(self, repo_dir: Path, ref: str) -> str:
        candidates = []
        if not ref.startswith("origin/"):
            candidates.append(f"origin/{ref}")
        candidates.append(ref)
        for candidate in candidates:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", candidate],
                cwd=repo_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return candidate
        raise RuntimeError(f"Unable to resolve git ref '{ref}' in {repo_dir}")

    def sync_vla_repo_to_cluster(self):
        if not self.local_vla_repo_dir or not self.config.vla_repo_remote_dir:
            return
        remote_path = self.config.vla_repo_remote_dir
        print(f"[prep] Syncing vla_foundry to all cluster nodes at {remote_path}")
        env = {**os.environ, "AWS_PROFILE": self.config.aws_profile}
        rsync_source = str(self.local_vla_repo_dir)
        if not rsync_source.endswith(os.sep):
            rsync_source = rsync_source + os.sep
        rsync_cmd = [
            "ray",
            "rsync-up",
            "--all-nodes",  # Workers need vla_foundry for Docker bind mounts
            str(self._cluster_config_for_ray),
            rsync_source,
            remote_path,
        ]
        self.run_subprocess(rsync_cmd, env=env, cwd=REPO_ROOT)
        self.remote_vla_repo_dir = remote_path

    def _load_tasks_if_needed(self) -> list[TaskSpec]:
        """Load tasks from file if not already loaded."""
        if self._loaded_tasks is None:
            if self.config.tasks_file:
                self._loaded_tasks = load_tasks_from_file(self.config.tasks_file)
            else:
                self._loaded_tasks = []
        return self._loaded_tasks

    def _get_unique_vla_refs(self) -> list[str | None]:
        """Get unique vla_refs from loaded tasks, including None for default."""
        tasks = self._load_tasks_if_needed()
        refs = set()
        for task in tasks:
            refs.add(task.vla_ref)  # None means use campaign default
        # Sort with None first
        return sorted(refs, key=lambda x: (x is not None, x or ""))

    def _validate_vla_refs(self):
        """Validate that all vla_refs in tasks exist in the remote repo."""
        if not self.config.vla_repo_url or not self.config.vla_repo_use_remote:
            return

        unique_refs = self._get_unique_vla_refs()
        # Filter out None (uses campaign default) and get actual refs to validate
        refs_to_check = []
        for ref in unique_refs:
            actual_ref = ref if ref is not None else self.config.vla_repo_ref
            if actual_ref:
                refs_to_check.append((ref, actual_ref))

        if not refs_to_check:
            return

        print("[validate] Checking vla_foundry branches exist...")

        # Get all remote refs in one call
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", self.config.vla_repo_url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to query vla_foundry repo: {result.stderr}")

        # Parse remote refs (format: "<sha>\trefs/heads/<branch>" or "refs/tags/<tag>")
        remote_refs = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                ref_path = parts[1]
                # Extract branch/tag name
                if ref_path.startswith("refs/heads/"):
                    remote_refs.add(ref_path[len("refs/heads/") :])
                elif ref_path.startswith("refs/tags/"):
                    remote_refs.add(ref_path[len("refs/tags/") :])

        # Validate each ref
        missing = []
        for task_ref, actual_ref in refs_to_check:
            if actual_ref not in remote_refs:
                display = task_ref if task_ref else "(campaign default)"
                missing.append(f"  - {actual_ref} (used by: {display})")

        if missing:
            raise RuntimeError(
                f"The following vla_foundry branches/tags do not exist in "
                f"{self.config.vla_repo_url}:\n" + "\n".join(missing)
            )

        print(f"[validate] ✓ All {len(refs_to_check)} vla_foundry ref(s) validated")

    def _sanitize_ref_for_path(self, ref: str | None) -> str:
        """Convert a git ref to a safe directory name."""
        if ref is None:
            return "default"
        # Replace slashes and other unsafe chars with underscores
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", ref)

    def _prepare_vla_ref(self, ref: str | None) -> tuple[Path, str]:
        """
        Prepare a specific vla_foundry ref and return (local_dir, remote_dir).

        Args:
            ref: Git ref (branch/tag/commit) or None to use campaign default

        Returns:
            Tuple of (local_path, remote_path) for the prepared ref
        """
        if ref in self.vla_ref_dirs:
            return self.vla_ref_dirs[ref]

        if not self.config.vla_repo_remote_dir:
            raise RuntimeError("vla_repo_remote_dir not configured")

        # Determine the actual ref to use
        actual_ref = ref if ref is not None else self.config.vla_repo_ref

        # Create unique local and remote paths for this ref as SUBDIRECTORIES
        # of the file_mounts path so autoscaled workers get them automatically.
        # file_mounts syncs ~/.cache/vla_foundry/ray_mount -> /home/ubuntu/vla_foundry_mount
        # We use subdirs: ray_mount/<ref_suffix>/ -> vla_foundry_mount/<ref_suffix>/
        ref_suffix = self._sanitize_ref_for_path(ref)
        if ref is None:
            # Default ref uses "default" subdirectory
            local_dir = self.config.vla_repo_local_dir / "default"
            remote_dir = f"{self.config.vla_repo_remote_dir}/default"
        else:
            # Custom refs get their own subdirectory
            local_dir = self.config.vla_repo_local_dir / ref_suffix
            remote_dir = f"{self.config.vla_repo_remote_dir}/{ref_suffix}"

        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prep] Preparing vla_foundry ref '{actual_ref}' at {local_dir}")

        def do_fresh_clone():
            """Perform a fresh shallow clone."""
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            clone_cmd = ["git", "clone", "--depth", "1"]
            if actual_ref:
                clone_cmd.extend(["--branch", actual_ref])
            clone_cmd.extend([self.config.vla_repo_url, str(local_dir)])
            self.run_subprocess(clone_cmd)

        checkout_ref = actual_ref or "main"
        need_fresh_clone = False

        if not (local_dir / ".git").exists():
            need_fresh_clone = True
        else:
            # Try to update existing repo, fall back to fresh clone on failure
            try:
                self.run_subprocess(
                    ["git", "remote", "set-url", "origin", self.config.vla_repo_url],
                    cwd=local_dir,
                )
                self.run_subprocess(["git", "fetch", "--all", "--tags"], cwd=local_dir)
                # Try to resolve the ref - if it fails, we need a fresh clone
                resolved = self.resolve_git_ref(local_dir, checkout_ref)
            except RuntimeError as e:
                print(f"[prep] Git update failed ({e}), doing fresh clone instead")
                need_fresh_clone = True

        if need_fresh_clone:
            do_fresh_clone()
            # After fresh clone, the ref should be available as we cloned with --branch
            resolved = self.resolve_git_ref(local_dir, checkout_ref)

        if resolved.startswith("origin/"):
            self.run_subprocess(
                ["git", "checkout", "-B", checkout_ref, resolved],
                cwd=local_dir,
            )
        else:
            self.run_subprocess(["git", "checkout", resolved], cwd=local_dir)
        self.run_subprocess(["git", "reset", "--hard", resolved], cwd=local_dir)
        # Don't create .venv here - it will be copied from Docker image cache at job runtime
        # See ray_policy_runner.py build_docker_command() for the venv copy logic

        self.vla_ref_dirs[ref] = (local_dir, remote_dir)
        print(f"[prep] vla_foundry ref '{actual_ref}' ready at {local_dir}")
        return local_dir, remote_dir

    def _sync_vla_ref_to_cluster(self, ref: str | None):
        """Sync a specific vla_foundry ref to all cluster nodes."""
        if ref not in self.vla_ref_dirs:
            raise RuntimeError(f"vla_ref '{ref}' not prepared; call _prepare_vla_ref first")

        local_dir, remote_dir = self.vla_ref_dirs[ref]
        print(f"[prep] Syncing vla_foundry ref to all cluster nodes at {remote_dir}")

        # Reset remote directory on ALL nodes (not just head)
        reset_script = (
            f"sudo rm -rf {remote_dir} && sudo mkdir -p {remote_dir} && sudo chown ubuntu:ubuntu {remote_dir}"
        )
        self._run_bash_on_all_nodes(reset_script, f"Reset {remote_dir}")

        # Rsync to ALL nodes (workers need the mount too for Docker bind mounts)
        env = {**os.environ, "AWS_PROFILE": self.config.aws_profile}
        rsync_source = str(local_dir)
        if not rsync_source.endswith(os.sep):
            rsync_source = rsync_source + os.sep
        rsync_cmd = [
            "ray",
            "rsync-up",
            "--all-nodes",  # Sync to all nodes, not just head
            str(self._cluster_config_for_ray),
            rsync_source,
            remote_dir,
        ]
        self.run_subprocess(rsync_cmd, env=env, cwd=REPO_ROOT)

    def prepare_and_deploy_all_vla_refs(self):
        """Prepare and deploy all unique vla_refs found in tasks."""
        if not self.config.vla_repo_remote_dir:
            return

        unique_refs = self._get_unique_vla_refs()
        if not unique_refs:
            unique_refs = [None]  # At least prepare the default

        print(f"[prep] Found {len(unique_refs)} unique vla_foundry ref(s) to prepare")
        for ref in unique_refs:
            ref_display = ref if ref else "(campaign default)"
            print(f"[prep] Preparing ref: {ref_display}")
            self._prepare_vla_ref(ref)
            self._sync_vla_ref_to_cluster(ref)

        # For backward compatibility, set default paths
        if None in self.vla_ref_dirs:
            self.local_vla_repo_dir, self.remote_vla_repo_dir = self.vla_ref_dirs[None]

    def deploy_anzu_repo(self):
        """Sync critical anzu_vla_foundry files to all cluster nodes.

        Note: file_mounts syncs when nodes join, but this ensures all alive nodes
        (including cached/reused workers) have the latest scripts.
        """
        print("[prep] Syncing anzu_vla_foundry files to all nodes...")
        cluster_scripts = REPO_ROOT / "vla_foundry" / "tri" / "lbm_eval" / "eval_campaigns" / "cluster_scripts"

        # Sync launch_sim.sh
        self.sync_file_to_all_nodes(cluster_scripts / "launch_sim.sh", "/home/ubuntu/anzu_vla_foundry/launch_sim.sh")

        # Sync run_inference_bundle.sh
        self.sync_file_to_all_nodes(
            cluster_scripts / "run_inference_bundle.sh",
            "/home/ubuntu/anzu_vla_foundry/run_inference_bundle.sh",
        )

    def sync_file_to_all_nodes(self, local_path: Path, remote_path: str):
        """Sync a local file to all cluster nodes using base64 encoding."""
        import base64

        if not local_path.exists():
            print(f"WARNING: Local file {local_path} not found. Skipping sync.")
            return

        with open(local_path, "rb") as f:
            content = f.read()
        b64_content = base64.b64encode(content).decode("utf-8")

        # Construct command to decode and write file
        # Use single quotes for the command to avoid shell expansion issues
        cmd = (
            f"mkdir -p $(dirname {remote_path}) && "
            f"echo {b64_content} | base64 -d > {remote_path} && "
            f"chmod +x {remote_path}"
        )

        self._run_bash_on_all_nodes(cmd, f"Syncing {local_path.name}")

    def _run_bash_on_all_nodes(self, script: str, description: str):
        """Execute a bash script on every alive Ray node via remote Ray tasks."""
        python_template = textwrap.dedent(
            """
            import ray
            import subprocess
            import sys
            import time

            ray.init(address="auto")

            script = __SCRIPT__
            description = __DESCRIPTION__

            @ray.remote(num_cpus=0)
            def _run():
                result = subprocess.run(
                    ["bash", "-lc", script],
                    capture_output=True,
                    text=True,
                )
                return result.returncode, result.stdout, result.stderr

            # Wait for nodes to be ready in autoscaler
            # ray.nodes() gives us physical nodes, but we need them to be in ray.cluster_resources()
            # to be schedulable with "node:<id>" resource.
            
            max_retries = 60
            ready_nodes = []
            all_nodes = []
            
            print(f"[{description}] Waiting for nodes to be schedulable...")
            for i in range(max_retries):
                all_nodes = [n for n in ray.nodes() if n.get("Alive")]
                if not all_nodes:
                    break
                    
                resources = ray.cluster_resources()
                
                ready_nodes = []
                for node in all_nodes:
                    node_id = node["NodeID"]
                    resource_key = f"node:{node_id}"
                    # Check if the specific node resource is available
                    if resource_key in resources:
                        ready_nodes.append(node)
                
                # If all alive nodes are ready, we can proceed immediately
                if len(ready_nodes) == len(all_nodes):
                    break
                
                if i % 5 == 0:
                    print(f"[{description}] Waiting for resources... ({len(ready_nodes)}/{len(all_nodes)} ready)")
                time.sleep(1)
            
            if not ready_nodes:
                print(f"No active and schedulable Ray nodes; skipping {description}.")
                sys.exit(0)

            if len(ready_nodes) < len(all_nodes):
                print(
                    f"WARNING: Only {len(ready_nodes)}/{len(all_nodes)} nodes are "
                    f"schedulable after timeout. Skipping others."
                )

            refs = []
            for node in ready_nodes:
                resource_key = f"node:{node['NodeID']}"
                refs.append(_run.options(resources={resource_key: 0.0001}).remote())

            failed = False
            for node, ref in zip(ready_nodes, refs):
                code, out, err = ray.get(ref)
                address = node.get("NodeManagerAddress", "unknown")
                print(f"[{description}] node {address} -> exit {code}")
                if out.strip():
                    print(out.strip())
                if err.strip():
                    print(err.strip(), file=sys.stderr)
                if code != 0:
                    failed = True

            if failed:
                sys.exit(1)
            """
        ).strip()

        python_script = python_template.replace("__SCRIPT__", repr(script)).replace(
            "__DESCRIPTION__", repr(description)
        )

        # Write to a temporary file locally
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp:
            tmp.write(python_script)
            tmp_path = tmp.name

        try:
            # Upload to cluster head
            remote_path = f"/tmp/{os.path.basename(tmp_path)}"
            env = {**os.environ}
            if self.config.aws_profile:
                env["AWS_PROFILE"] = self.config.aws_profile

            rsync_cmd = [
                "ray",
                "rsync-up",
                str(self._cluster_config_for_ray),
                tmp_path,
                remote_path,
            ]
            self.run_subprocess(rsync_cmd, env=env, cwd=REPO_ROOT)

            # Execute on cluster head
            exec_cmd = [
                "ray",
                "exec",
                str(self._cluster_config_for_ray),
                f"python3 {remote_path}",
            ]
            try:
                self.run_subprocess(exec_cmd, env=env, cwd=REPO_ROOT)
            except RuntimeError as exc:
                print(f"  WARNING: Failed during '{description}': {exc}")

            # Cleanup remote file
            cleanup_cmd = [
                "ray",
                "exec",
                str(self._cluster_config_for_ray),
                f"rm {remote_path}",
            ]
            # Best effort cleanup
            subprocess.run(cleanup_cmd, env=env, cwd=REPO_ROOT, capture_output=True)

        finally:
            # Cleanup local file
            os.unlink(tmp_path)

    def monitor_jobs(self):
        """Monitor job progress until all complete."""
        print("[3/5] Monitoring job progress...")
        print()

        if not self.submission_ids:
            print("WARNING: No submission IDs found. Skipping monitoring.")
            return

        client = self._create_job_submission_client()
        pending = set(self.submission_ids)
        # Don't reset completed if we're resuming from saved state
        if not self.completed:
            self.completed = {}

        # Remove already-completed jobs from pending (for resume)
        pending -= set(self.completed.keys())

        last_status_time = time.time()
        last_health_check_time = time.time()
        status_interval = 30  # Print status every 30 seconds
        health_check_interval = 60  # Check cluster health every 60 seconds
        consecutive_errors = 0
        max_consecutive_errors = 5

        # Save initial state
        self.save_state(force=True)

        while pending:
            finished = []

            for submission_id in pending:
                try:
                    status = client.get_job_status(submission_id)
                    if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                        print(f"  {submission_id[:8]}: {status}")
                        self.completed[submission_id] = status
                        finished.append(submission_id)

                        # Save Ray job logs when requested or on failure
                        if status == JobStatus.FAILED or self.config.save_job_logs:
                            self.save_job_logs(client, submission_id)
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"  Error checking {submission_id[:8]}: {e}")

            for submission_id in finished:
                pending.remove(submission_id)

            # Save state periodically
            self.save_state()

            # Check cluster health periodically
            now = time.time()
            if now - last_health_check_time >= health_check_interval:
                is_healthy, health_msg = self.check_cluster_health()
                last_health_check_time = now
                if not is_healthy:
                    print(f"  [health] WARNING: Cluster health check failed: {health_msg}")
                    print(
                        f"  [health] Unreachable count: "
                        f"{self._cluster_unreachable_count}/{self._max_cluster_unreachable}"
                    )
                    if self.is_cluster_lost():
                        print()
                        print("=" * 80)
                        print("FATAL: Lost connection to Ray cluster!")
                        print(f"  Last cluster URL: {self.cluster_url}")
                        print(f"  Completed jobs: {len(self.completed)}/{len(self.submission_ids)}")
                        print(f"  Pending jobs: {len(pending)}")
                        print()
                        print("Campaign state has been saved. To check cluster status:")
                        print(f"  curl {self.cluster_url}/api/cluster_status")
                        print()
                        print("To check if head node exists:")
                        print(
                            f"  AWS_PROFILE={self.config.aws_profile} ray get-head-ip {self.config.cluster_config_file}"
                        )
                        print()
                        print(f"State file: {self._get_state_file_path()}")
                        print("=" * 80)
                        self.save_state(force=True)
                        raise RuntimeError(
                            f"Lost connection to Ray cluster after {self._max_cluster_unreachable} "
                            f"consecutive failures. State saved to {self._get_state_file_path()}"
                        )

            # Handle consecutive errors (might indicate cluster issues)
            if consecutive_errors >= max_consecutive_errors:
                print(f"  [monitor] WARNING: {consecutive_errors} consecutive errors - checking cluster health")
                is_healthy, health_msg = self.check_cluster_health()
                if not is_healthy:
                    print(f"  [monitor] Cluster health: {health_msg}")
                    # Try to recreate client
                    try:
                        print("  [monitor] Attempting to reconnect to job server...")
                        client = self._create_job_submission_client()
                        print("  [monitor] Reconnected successfully")
                        consecutive_errors = 0
                    except Exception as e:
                        print(f"  [monitor] Reconnection failed: {e}")

            if pending:
                # Print periodic status update
                if now - last_status_time >= status_interval:
                    print(f"  Waiting on {len(pending)} jobs... ({len(self.completed)} completed)")
                    last_status_time = now

                    # Periodically clean up Docker containers for finished jobs
                    # This allows nodes to become idle and be terminated by the autoscaler
                    try:
                        self._terminate_inference_containers()
                    except Exception as e:
                        print(f"  WARNING: Container cleanup failed: {e}")

                time.sleep(10)

        print()
        succeeded_count = sum(1 for s in self.completed.values() if str(s) == "SUCCEEDED" or s == JobStatus.SUCCEEDED)
        failed_count = sum(1 for s in self.completed.values() if str(s) == "FAILED" or s == JobStatus.FAILED)
        print("✓ All jobs completed")
        print(f"  Succeeded: {succeeded_count}")
        print(f"  Failed: {failed_count}")

        # Save final state
        self.save_state(force=True)

        # Final cleanup to terminate all remaining containers
        print()
        print("[cleanup] Final container cleanup...")
        try:
            self._terminate_inference_containers()
        except Exception as e:
            print(f"  WARNING: Final cleanup failed: {e}")
        print()

    def _terminate_inference_containers(self):
        """Ensure no previous inference Docker containers are still running."""
        print("[prep] Checking for running inference containers...")

        # Check if there are any active worker nodes first
        try:
            import ray

            # Derive Ray Client address from Dashboard URL (http://ip:8265 -> ray://ip:10001)
            ip = self.cluster_url.split("//")[-1].split(":")[0]
            ray_addr = f"ray://{ip}:10001"
            ray.init(address=ray_addr, ignore_reinit_error=True)

            nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
            ray.shutdown()

            if not nodes:
                print("[prep] No active GPU nodes detected; skipping cleanup.")
                return
        except Exception as e:
            print(f"[prep] Could not check cluster state, skipping cleanup: {e}")
            return

        docker_image = shlex.quote(self.config.docker_image)
        cleanup_script = f"""
label_containers=$(sudo docker ps -q --filter label={EVAL_DOCKER_LABEL}=1)
ancestor_containers=$(sudo docker ps -q --filter ancestor={docker_image})
bundle_containers=$(sudo docker ps -q --filter "name=run_inference_bundle")
killed=0
if [ -n "$label_containers" ]; then
  echo "Killing labeled inference containers: $label_containers"
  sudo docker kill $label_containers >/dev/null 2>&1 || true
  killed=1
fi
if [ -n "$ancestor_containers" ]; then
  echo "Killing containers running image {self.config.docker_image}: $ancestor_containers"
  sudo docker kill $ancestor_containers >/dev/null 2>&1 || true
  killed=1
fi
if [ -n "$bundle_containers" ]; then
  echo "Killing containers running run_inference_bundle: $bundle_containers"
  sudo docker kill $bundle_containers >/dev/null 2>&1 || true
  killed=1
fi
if [ "$killed" -eq 0 ]; then
  echo "No inference containers detected."
fi

# Prune stopped containers to free disk space
echo "Pruning stopped containers..."
sudo docker container prune -f >/dev/null 2>&1 || true

# Kill any orphaned GPU processes on the host (outside Docker)
echo "Checking for orphaned GPU processes..."
if command -v nvidia-smi &>/dev/null; then
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -n "$gpu_pids" ]; then
    echo "Found GPU processes: $gpu_pids"
    for pid in $gpu_pids; do
      # Check if it's a Ray process (don't kill Ray itself)
      proc_name=$(ps -p $pid -o comm= 2>/dev/null || true)
      if [ -n "$proc_name" ] && [ "$proc_name" != "raylet" ] && [ "$proc_name" != "ray" ]; then
        echo "Killing orphaned GPU process $pid ($proc_name)"
        sudo kill -9 $pid 2>/dev/null || true
      fi
    done
  else
    echo "No GPU processes found."
  fi
fi

# Clean up temporary checkpoint directories
echo "Cleaning up temporary files..."
sudo rm -rf /tmp/vla_foundry_checkpoints/* 2>/dev/null || true
sudo rm -rf /tmp/vla_foundry_runtime 2>/dev/null || true

# Report disk usage
echo "Disk usage on /tmp:"
df -h /tmp 2>/dev/null || true
"""
        self._run_bash_on_all_nodes(cleanup_script, "terminate inference containers")

    def _create_job_submission_client(
        self,
        max_attempts: int = 12,
        base_wait_seconds: int = 5,
    ) -> JobSubmissionClient:
        """Create a JobSubmissionClient with retries while the cluster boots."""
        if not self.cluster_url:
            raise RuntimeError("Cluster URL is not available for JobSubmissionClient creation.")

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return JobSubmissionClient(self.cluster_url)
            except Exception as exc:
                last_exc = exc
                wait_time = min(30, base_wait_seconds * attempt)
                print(f"  Ray job server not ready yet (attempt {attempt}/{max_attempts}): {exc}")
                time.sleep(wait_time)

        raise RuntimeError(
            f"Unable to connect to Ray job server at {self.cluster_url} after {max_attempts} attempts: {last_exc}"
        )

    def save_job_logs(self, client: JobSubmissionClient, submission_id: str):
        """Save logs for a job to the results directory."""
        try:
            logs_dir = self.config.results_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / f"{submission_id}.log"
            logs = client.get_job_logs(submission_id)

            with open(log_file, "w") as f:
                f.write(logs)

            print(f"    Logs saved to: {log_file}")
        except Exception as e:
            print(f"    Failed to save logs: {e}")

    def collect_results(self):
        """Collect results from the cluster."""
        print("[4/5] Collecting results...")
        print()

        # Create results directory
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Results directory: {self.config.results_dir}")
        self._download_cluster_artifacts(
            include_rollouts=self.config.download_rollouts,
            include_episode_pkls=self.config.download_episode_pkls,
        )

        # Save campaign metadata
        metadata = {
            "campaign_name": self.config.campaign_name,
            "description": self.config.description,
            "num_workers": self.config.num_workers,
            "num_samples": self.config.num_samples,
            "submission_ids": self.submission_ids,
            "checkpoints": self._resolve_checkpoint_paths(),
        }

        metadata_file = self.config.results_dir / "campaign_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_file}")

        success_metrics = self.compute_success_metrics()
        if success_metrics:
            self.success_metrics = success_metrics

        # Note: Artifacts are already in S3 (uploaded by inference jobs),
        # so we don't need to re-upload them here.
        print()

    def generate_report(self):
        """Generate summary report."""
        print("[5/5] Generating summary report...")
        print()

        report_file = self.config.results_dir / "campaign_report.txt"

        with open(report_file, "w") as f:
            f.write("Evaluation Campaign Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Campaign: {self.config.campaign_name}\n")
            f.write(f"Description: {self.config.description}\n\n")
            f.write("Configuration:\n")
            f.write(f"  Workers: {self.config.num_workers}\n")
            f.write(f"  Samples per task: {self.config.num_samples}\n")
            f.write(f"  Start index: {self.config.start_index}\n\n")
            f.write(f"Jobs submitted: {len(self.submission_ids)}\n")
            f.write("Submission IDs:\n")
            for sub_id in self.submission_ids:
                f.write(f"  - {sub_id}\n")
            if self.success_metrics:
                f.write("\nSuccess Metrics:\n")
                f.write(f"  Successes:     {self.success_metrics['successes']}\n")
                f.write(f"  Rollouts:      {self.success_metrics['rollouts']}\n")
                if self.success_metrics["rollouts"]:
                    success_rate = self.success_metrics["success_rate"]
                    f.write(f"  Success Rate:  {success_rate:.4f}\n")
                f.write(
                    "  Confidence Interval "
                    f"({self.success_metrics['confidence_level']}): "
                    f"[{self.success_metrics['confidence_interval']['lower']:.4f}, "
                    f"{self.success_metrics['confidence_interval']['upper']:.4f}]\n"
                )
                f.write(f"  Metrics File:  {self.success_metrics['metrics_path']}\n")

        print(f"✓ Report saved to: {report_file}")
        print()

    def teardown_cluster(self):
        """Tear down the Ray cluster."""
        print("\nTearing down cluster...")

        cmd = [
            "ray",
            "down",
            str(self._cluster_config_for_ray),
            "--yes",
        ]

        env = {"AWS_PROFILE": self.config.aws_profile}

        result = subprocess.run(
            cmd,
            env={**subprocess.os.environ, **env},
            capture_output=False,
        )

        if result.returncode != 0:
            print("WARNING: Failed to tear down cluster cleanly", file=sys.stderr)
        else:
            print("✓ Cluster torn down")

    def compute_success_metrics(self) -> dict[str, Any] | None:
        """Aggregate rollout success metrics and persist them locally/S3."""
        rollouts_root = self.config.results_dir / "rollouts" / "rollouts"
        if not rollouts_root.exists():
            print(f"  No rollouts found under {rollouts_root}; skipping success metric aggregation.")
            return None

        # Look for summaries in any checkpoint subdirectory or directly in rollouts
        # New structure: rollouts_root/{checkpoint_name}/demonstration_*/summary.yaml
        # Old structure: rollouts_root/demonstration_*/summary.yaml
        summary_globs = [
            str(rollouts_root / "*" / "demonstration_*" / "summary.yaml"),
            str(rollouts_root / "demonstration_*" / "summary.yaml"),
        ]
        success_array = collect_multi_rollout_success_stats(summary_globs)
        successes = int(success_array[0, 0])
        rollouts = int(success_array[1, 0])
        if rollouts == 0:
            print("  Found zero completed demonstrations; skipping success metric aggregation.")
            return None

        lower_bound, upper_bound = calculate_confidence_interval(
            successes=successes,
            rollouts=rollouts,
            confidence_level=SUCCESS_CONFIDENCE_LEVEL,
            interval_type="two-sided",
            scipy_interval=True,
        )
        success_rate = successes / rollouts if rollouts else 0.0
        metrics = {
            "campaign_name": self.config.campaign_name,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "successes": successes,
            "rollouts": rollouts,
            "success_rate": success_rate,
            "confidence_level": SUCCESS_CONFIDENCE_LEVEL,
            "confidence_interval": {
                "lower": lower_bound,
                "upper": upper_bound,
            },
            "summary_globs": summary_globs,
        }

        metrics_path = self.config.results_dir / SUCCESS_METRICS_FILENAME
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        metrics["metrics_path"] = str(metrics_path)

        print(f"✓ Aggregated success metrics: {successes}/{rollouts} ({success_rate:.2%})")
        print(f"  Saved to: {metrics_path}")

        self.upload_success_metrics_to_s3(metrics_path)
        return metrics

    def upload_success_metrics_to_s3(self, metrics_path: Path) -> None:
        """Upload success metrics next to each S3 checkpoint under evaluation/."""
        checkpoint_paths = self._resolve_checkpoint_paths()
        s3_paths = [path for path in checkpoint_paths if path.startswith("s3://")]
        if not s3_paths:
            print("  No S3 checkpoints detected; skipping success metric upload.")
            return

        uploaded_destinations = []
        profile = self.config.inference_aws_profile or self.config.aws_profile
        env = {**os.environ}
        if profile:
            env["AWS_PROFILE"] = profile

        for checkpoint in dict.fromkeys(s3_paths):
            destination = self._default_s3_evaluation_path(checkpoint)
            cmd = ["aws", "s3", "cp", str(metrics_path), destination]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    env=env,
                    capture_output=False,
                )
            except subprocess.CalledProcessError as exc:
                print(f"  WARNING: Failed to upload success metrics to {destination}: {exc}")
                continue
            uploaded_destinations.append(destination)

        if uploaded_destinations:
            print("  Success metrics uploaded to:")
            for dest in uploaded_destinations:
                print(f"    {dest}")

    def _resolve_checkpoint_paths(self) -> list[str]:
        """Return the list of checkpoint paths associated with this campaign."""
        if self.config.tasks_file:
            task_specs = load_tasks_from_file(self.config.tasks_file)
            return [task.checkpoint for task in task_specs]
        if self.config.checkpoints:
            return list(self.config.checkpoints)
        return []

    def _resolve_task_checkpoint_pairs(self) -> list[tuple[str, str]]:
        """Return list of (task_name, checkpoint) pairs for this campaign.

        This is needed for multi-task evaluation where the same checkpoint
        is evaluated on multiple tasks - results are stored under task-specific paths.
        """
        if self.config.tasks_file:
            task_specs = load_tasks_from_file(self.config.tasks_file)
            return [(task.task_name, task.checkpoint) for task in task_specs]
        if self.config.checkpoints:
            # For checkpoint-only mode, use empty task name (legacy behavior)
            return [("", checkpoint) for checkpoint in self.config.checkpoints]
        return []

    def _default_s3_evaluation_path(self, checkpoint_path: str) -> str:
        base = checkpoint_path.rstrip("/")
        slug = self._campaign_slug()
        return f"{base}/evaluation/{slug}/{SUCCESS_METRICS_FILENAME}"

    def _evaluation_dir_for_checkpoint(self, checkpoint_path: str) -> str:
        base = checkpoint_path.rstrip("/")
        slug = self._campaign_slug()
        return f"{base}/evaluation/{slug}"

    def _campaign_slug(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.config.campaign_name)
        slug = slug.strip("_")
        return slug or "campaign"

    def _download_cluster_artifacts(
        self,
        include_rollouts: bool = False,
        include_episode_pkls: bool = False,
    ) -> None:
        """Download evaluation artifacts from S3 for each task/checkpoint.

        The inference jobs upload their results to S3 at:
          {checkpoint}/evaluation/{task_name}/rollouts/demonstration_*/summary.yaml

        For multi-task evaluation on the same checkpoint, task_name is included
        in the path to separate results. For legacy single-task, falls back to
        {checkpoint}/evaluation/rollouts/ if task-specific path is empty.

        This method downloads those results rather than relying on worker nodes
        still being alive for rsync.
        """
        task_checkpoint_pairs = self._resolve_task_checkpoint_pairs()
        s3_pairs = [(task, cp) for task, cp in task_checkpoint_pairs if cp.startswith("s3://")]

        if not s3_pairs:
            print("  No S3 checkpoints found; skipping artifact download.")
            return

        rollouts_root = self.config.results_dir / "rollouts"
        rollouts_root.mkdir(parents=True, exist_ok=True)
        aggregated_rollouts = rollouts_root / "rollouts"
        aggregated_rollouts.mkdir(parents=True, exist_ok=True)

        env = {**os.environ}
        aws_profile = self.config.inference_aws_profile or self.config.aws_profile
        if aws_profile:
            env["AWS_PROFILE"] = aws_profile

        if include_rollouts:
            print(
                "  Downloading rollouts from S3 for "
                f"{len(s3_pairs)} task(s)... "
                f"(episode pkls: {'yes' if include_episode_pkls else 'no'})"
            )
        else:
            print(f"  Downloading rollout summaries from S3 for {len(s3_pairs)} task(s)...")

        total_downloaded = 0
        for task_name, checkpoint in s3_pairs:
            # Strip any trailing /checkpoints/* path
            s3_base = checkpoint.rstrip("/")
            if s3_base.endswith(".pt"):
                s3_base = s3_base.rsplit("/checkpoints/", 1)[0]

            # Include task name in path if provided (for multi-task evaluation)
            if task_name:
                s3_rollouts = f"{s3_base}/evaluation/{task_name}/rollouts/"
            else:
                s3_rollouts = f"{s3_base}/evaluation/rollouts/"

            # Create a unique local directory for this task's artifacts
            checkpoint_name = s3_base.split("/")[-1]
            if task_name:
                local_dir = aggregated_rollouts / f"{checkpoint_name}_{task_name}"
            else:
                local_dir = aggregated_rollouts / checkpoint_name
            local_dir.mkdir(parents=True, exist_ok=True)

            display_name = f"{task_name} ({checkpoint_name})" if task_name else checkpoint_name
            print(f"    - {display_name}: {s3_rollouts}")

            cmd: list[str] = [
                "aws",
                "s3",
                "sync",
                s3_rollouts,
                str(local_dir),
                "--quiet",
            ]
            if include_rollouts:
                if not include_episode_pkls:
                    # Episode pkls are large; skip them unless explicitly requested.
                    cmd += ["--exclude", "*.pkl"]
            else:
                # Default behavior: download only per-demonstration summaries.
                cmd += ["--exclude", "*", "--include", "demonstration_*/summary.yaml"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                stderr = result.stderr.strip()
                print(f"      WARNING: Failed to download from {s3_rollouts}: {stderr}")
            else:
                # Count downloaded demonstrations
                demos = list(local_dir.glob("demonstration_*"))
                if demos:
                    total_downloaded += len(demos)
                    print(f"      Downloaded {len(demos)} demonstration(s)")
                else:
                    print("      No demonstrations found")

        print(f"  Total demonstrations downloaded: {total_downloaded}")

    def _discover_cluster_nodes(self) -> list[tuple[str, str]]:
        """Return [('head', ip0), ('worker0', ip1), ...] for the active cluster."""
        env = {**os.environ, "AWS_PROFILE": self.config.aws_profile}
        nodes: list[tuple[str, str]] = []
        ip_pattern = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")

        head_output = self._capture_command_output(
            ["ray", "get-head-ip", str(self._cluster_config_for_ray)],
            env=env,
        )
        if head_output:
            match = ip_pattern.search(head_output)
            if match:
                nodes.append(("head", match.group(0)))

        worker_output = self._capture_command_output(
            ["ray", "get-worker-ips", str(self._cluster_config_for_ray)],
            env=env,
        )
        if worker_output:
            worker_idx = 0
            for line in worker_output.splitlines():
                match = ip_pattern.search(line)
                if match:
                    nodes.append((f"worker{worker_idx}", match.group(0)))
                    worker_idx += 1

        return nodes

    def _capture_command_output(self, cmd, env=None) -> str:
        """Run a command and return stdout, logging warnings on failure."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "no output"
            print(f"  WARNING: {' '.join(cmd)} failed ({details})")
            return ""
        return result.stdout.strip()

    def _get_cluster_auth(self) -> tuple[str, Path]:
        if self._cluster_auth is not None:
            return self._cluster_auth

        with open(self.config.cluster_config_file) as fp:
            cluster_cfg = yaml.safe_load(fp)

        auth_cfg = cluster_cfg.get("auth", {})
        ssh_user = auth_cfg.get("ssh_user", "ubuntu")
        key_path = Path(auth_cfg.get("ssh_private_key", "~/.ssh/id_rsa")).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(f"SSH private key not found at {key_path}")

        self._cluster_auth = (ssh_user, key_path)
        return self._cluster_auth

    def _rsync_node_volume(self, node_label: str, node_ip: str, destination: Path) -> bool:
        ssh_user, ssh_key = self._get_cluster_auth()
        ssh_cmd = f"ssh -i {shlex.quote(str(ssh_key))} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

        remote_root = "/var/lib/docker/volumes/anzu_vla_foundry_logs/_data/"
        rsync_cmd = [
            "rsync",
            "-az",
            "-L",
            "--rsync-path",
            "sudo -n rsync",
            "-e",
            ssh_cmd,
            f"{ssh_user}@{node_ip}:{remote_root}",
            str(destination) + "/",
        ]

        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "no output"
            print(f"      rsync failed for {node_label}: {details}")
            return False
        return True

    def _stage_node_artifacts(self, label: str, node_dir: Path) -> None:
        """Move logs + rollouts from node_dir into campaign directories."""
        rollouts_root = self.config.results_dir / "rollouts"
        rollouts_root.mkdir(parents=True, exist_ok=True)
        aggregated_rollouts = rollouts_root / "rollouts"
        aggregated_rollouts.mkdir(parents=True, exist_ok=True)

        node_rollouts = node_dir / "rollouts"
        if node_rollouts.exists():
            tar_path = rollouts_root / f"{label}.tar"
            self._create_tar_archive(node_rollouts, tar_path)
            self._move_rollout_children(node_rollouts, aggregated_rollouts, label)
        else:
            print(f"      No rollouts located on {label}")

        logs_remote = node_dir / "logs"
        if not logs_remote.exists():
            print(f"      No logs found for {label}")
            return

        # Stage logs (bazel, inference, setup) for every job
        campaign_logs = self.config.results_dir / "logs"
        campaign_logs.mkdir(parents=True, exist_ok=True)
        for job_dir in logs_remote.iterdir():
            if not job_dir.is_dir():
                continue
            for log_name in ["bazel.log", "inference.log", "setup.log"]:
                src = job_dir / log_name
                if src.exists():
                    # Prefix with node label to identify which worker ran it
                    dest = campaign_logs / f"{label}_{job_dir.name}_{log_name}"
                    shutil.copy2(src, dest)
                    print(f"      Copied {log_name} from {job_dir.name}")

        shutil.rmtree(node_dir, ignore_errors=True)

    def _create_tar_archive(self, source_dir: Path, tar_path: Path) -> None:
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "w") as archive:
            archive.add(str(source_dir), arcname="rollouts")

    def _move_rollout_children(
        self,
        source_dir: Path,
        destination_root: Path,
        node_label: str,
    ) -> None:
        entries = sorted(source_dir.iterdir(), key=lambda path: path.name)
        for entry in entries:
            target = destination_root / entry.name
            if target.exists():
                base_name = f"{node_label}_{entry.name}"
                candidate = destination_root / base_name
                suffix = 1
                while candidate.exists():
                    suffix += 1
                    candidate = destination_root / f"{base_name}_{suffix}"
                target = candidate
            shutil.move(str(entry), str(target))

        shutil.rmtree(source_dir, ignore_errors=True)

    def _upload_cluster_artifacts_to_s3(self) -> None:
        """Push collected artifacts into each checkpoint's evaluation directory."""
        rollouts_root = self.config.results_dir / "rollouts"
        if not rollouts_root.exists():
            print("  No local artifacts to upload; skipping S3 sync.")
            return

        checkpoints = [path for path in self._resolve_checkpoint_paths() if path.startswith("s3://")]
        if not checkpoints:
            print("  No S3 checkpoints detected; skipping artifact upload.")
            return

        env = {**os.environ}
        aws_profile = self.config.inference_aws_profile or self.config.aws_profile
        if aws_profile:
            env["AWS_PROFILE"] = aws_profile

        uploaded_any = False
        for checkpoint in dict.fromkeys(checkpoints):
            evaluation_dir = self._evaluation_dir_for_checkpoint(checkpoint)
            destination = f"{evaluation_dir}/artifacts/"
            print(f"  Uploading artifacts to {destination}")
            if self._sync_directory_to_s3(rollouts_root, destination, env=env):
                uploaded_any = True
                print(f"    ✓ Artifacts uploaded to {destination}")
            else:
                print(f"    WARNING: Failed to upload artifacts to {destination}")

        if uploaded_any:
            shutil.rmtree(rollouts_root, ignore_errors=True)

    def _sync_directory_to_s3(self, source: Path, destination: str, env: dict[str, str]) -> bool:
        cmd = [
            "aws",
            "s3",
            "sync",
            str(source),
            destination,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "no output"
            print(f"      aws s3 sync failed: {details}")
            return False
        return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to campaign configuration YAML file",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip termination of existing containers (useful for concurrent campaigns)",
    )
    parser.add_argument(
        "--skip-cluster-startup",
        action="store_true",
        help="Skip Ray cluster startup (use when cluster is already running)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved campaign state if available",
    )
    parser.add_argument(
        "--download-rollouts",
        action="store_true",
        help="Download full rollout artifacts from S3 (still excludes episode .pkl by default)",
    )
    parser.add_argument(
        "--download-episode-pkls",
        action="store_true",
        help="Download episode .pkl files from S3 (implies --download-rollouts)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = CampaignConfig.from_yaml(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Override config with CLI args
    if args.skip_cleanup:
        # Dataclasses are mutable by default
        config.skip_cleanup = True
    if args.download_rollouts or args.download_episode_pkls:
        config.download_rollouts = True
    if args.download_episode_pkls:
        config.download_episode_pkls = True

    # Store skip_cluster_startup flag
    skip_cluster_startup = args.skip_cluster_startup

    if args.dry_run:
        print("Configuration loaded successfully:")
        print(f"  Campaign: {config.campaign_name}")
        print(f"  Workers: {config.num_workers}")
        print(f"  Samples: {config.num_samples}")
        print(f"  Results: {config.results_dir}")
        return 0

    # Run campaign
    campaign = EvaluationCampaign(config)
    return campaign.run(
        skip_cluster_startup=skip_cluster_startup,
        resume=args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
