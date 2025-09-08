import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import boto3
import draccus

import sagemaker

# Import PreprocessParams from the main preprocessing script
from lbm2.data.preprocessing.preprocess_lbm_data import PreprocessParams
from lbm2.params.base_params import BaseParams
from sagemaker.batch_queueing.queue import Queue
from sagemaker.pytorch import PyTorch

NAME = "lbm2-preprocess"
INSTANCE_MAPPER = {
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}
QUEUE_MAPPER = {
    "us-west-2": {
        "ml.p5.48xlarge": "fss-ml-p5-48xlarge-us-west-2",
        "ml.p4de.24xlarge": "fss-ml-p4de-24xlarge-us-west-2",
        "ml.p4d.24xlarge": "fss-ml-p4d-24xlarge-us-west-2",
    },
}


def run_command(command: str) -> None:
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def remove_old_hyperparameters(path: str, expiration_days: int = 3) -> None:
    """Remove old hyperparameters files."""
    for file in Path(path).glob("hyperparameters_*.yaml"):
        if file.stat().st_mtime < time.time() - expiration_days * 24 * 60 * 60:
            file.unlink()


def get_image(user: str, profile: str = "default", region: str = "us-west-2") -> str:
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    assert account.isdigit(), f"Invalid account value: {account}"
    docker_dir = Path(__file__).parent
    algorithm_name = f"{user}-{NAME}"
    dockerfile_base = docker_dir / "Dockerfile"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin"
    )
    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} --no-cli-pager || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name} --no-cli-pager"
        ),
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


@dataclass(frozen=True)
class SageMakerRunParams(BaseParams):
    # AWS profile args
    region: str = field(default="us-west-2")
    profile: str = field(default="default")
    arn: str = field(default=None)

    # Instance args
    instance_count: int = field(default=1)
    instance_type: str = field(default="p4de")
    max_run: int = field(default=2)  # days

    # Misc
    user: str = field(default=None)
    # SageMaker queue args
    queue_name: str = field(default="ml")
    priority: int = field(default=1)


@dataclass(frozen=True)
class SageMakerPreprocessParams(PreprocessParams):
    """Combined parameters for SageMaker preprocessing runs."""

    sagemaker: SageMakerRunParams = field(default_factory=SageMakerRunParams)


def main() -> None:
    args = draccus.parse(config_class=SageMakerPreprocessParams)

    # Validate required args
    assert args.source_episodes, "--source_episodes is required"
    assert args.output_dir, "--output_dir is required"

    sm_args = args.sagemaker
    assert sm_args.instance_type in INSTANCE_MAPPER
    if sm_args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, (
            "Please specify --sagemaker.arn or set the SAGEMAKER_ARN environment variable"
        )
        object.__setattr__(sm_args, "arn", os.environ["SAGEMAKER_ARN"])  # type: ignore[attr-defined]

    # Optionally override the default preprocessing params in the image
    uuid_str = str(uuid.uuid4())
    temp_file_path = f"sagemaker/configs/hyperparameters_{uuid_str}.yaml"
    hyperparameter_sagemaker_path = f"/opt/ml/code/configs/hyperparameters_{uuid_str}.yaml"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        args_dict = draccus.parsers.encoding.encode(args)
        del args_dict["sagemaker"]
        print(args_dict)
        draccus.cfgparsing.save_config(args_dict, f)
    remove_old_hyperparameters("sagemaker/configs", expiration_days=3)

    # Build/push ECR image (Dockerfile copies the hyperparameters file we just wrote)
    image = get_image(user=sm_args.user, region=sm_args.region, profile=sm_args.profile)
    os.environ["AWS_DEFAULT_REGION"] = sm_args.region

    # Create session
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=sm_args.region))

    role = sm_args.arn
    role_name = role.split("/")[-1]
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name}")

    # Job naming
    base_job_name = f"{sm_args.user.replace('.', '-')}-{NAME}"

    def get_job_name(base: str) -> str:
        now = datetime.now()
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        job_name = "-".join([base, date_str])
        return job_name

    job_name = get_job_name(base_job_name)

    # Environment
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "WANDB_PROJECT": "lbm2",
        "SAGEMAKER_PROGRAM": "/opt/ml/code/lbm2/data/preprocessing/preprocess_lbm_data.py",
    }
    if os.path.exists("secrets.env"):
        with open("secrets.env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    environment[key.strip()] = value.strip().strip("\"'")

    # Configure estimator to run preprocessing script and submit via Queue
    estimator = PyTorch(
        entry_point="lbm2/data/preprocessing/preprocess_lbm_data.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters={"config_path": hyperparameter_sagemaker_path},
        role=role,
        image_uri=image,
        instance_count=sm_args.instance_count,
        instance_type=INSTANCE_MAPPER[sm_args.instance_type],
        job_name=job_name,
        checkpoint_local_path=None,
        distribution=None,  # no torch.distributed for preprocessing
        max_run=sm_args.max_run * 24 * 60 * 60,
        input_mode="File",
        environment=environment,
        keep_alive_period_in_seconds=5 * 60,
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{sm_args.user}@tri.global"},
        ],
    )

    queue = Queue(
        queue_name=QUEUE_MAPPER[sm_args.region][INSTANCE_MAPPER[sm_args.instance_type]].replace(
            "ml", sm_args.queue_name
        )
    )
    queue.map(
        estimator,
        inputs=[None],
        job_names=[job_name],
        priority=sm_args.priority,
        share_identifier="default",
        timeout={"attemptDurationSeconds": sm_args.max_run * 24 * 60 * 60},
    )
    print(f"Queued {job_name}")


if __name__ == "__main__":
    main()
