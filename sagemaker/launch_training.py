import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import boto3
import draccus

import sagemaker
from lbm2.params.base_params import BaseParams
from lbm2.params.train_experiment_params import TrainExperimentParams
from sagemaker.batch_queueing.queue import Queue
from sagemaker.pytorch import PyTorch

NAME = "lbm2"
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


@dataclass(frozen=True)
class SageMakerRunParams(BaseParams):
    local: bool = field(default=False)
    user: str = field(default=None)

    # AWS profile args
    region: str = field(default="us-west-2")
    profile: str = field(default="default")
    arn: str = field(default=None)

    # Instance args
    instance_count: int = field(default=1)
    instance_type: str = field(default="p4de")
    max_run: int = field(default=10)

    # SageMaker queue args
    queue_name: str = field(default="ml")
    priority: int = field(default=1)


@dataclass(frozen=True)
class SageMakerParams(TrainExperimentParams):
    sagemaker: SageMakerRunParams = field(default_factory=SageMakerRunParams)

    def __post_init__(self):
        pass


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def remove_old_hyperparameters(path, expiration_days=3):
    """Remove old hyperparameters files."""
    for file in Path(path).glob("hyperparameters_*.yaml"):
        if file.stat().st_mtime < time.time() - expiration_days * 24 * 60 * 60:
            file.unlink()


def get_image(user, profile="default", region="us-east-1"):
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


def main():
    args = draccus.parse(config_class=SageMakerParams)

    # Save hyperparameters to a yaml file. Files are deleted after 3 days.
    uuid = str(uuid4())
    temp_file_path = f"sagemaker/configs/hyperparameters_{uuid}.yaml"
    hyperparameter_sagemaker_path = f"/opt/ml/code/configs/hyperparameters_{uuid}.yaml"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        args_dict = draccus.parsers.encoding.encode(args)
        del args_dict["sagemaker"]
        print(args_dict)
        draccus.cfgparsing.save_config(args_dict, f)
    remove_old_hyperparameters("sagemaker/configs", expiration_days=3)

    # We probably want wandb logging and S3 saving for sagemaker runs
    assert args.remote_sync is not None
    assert args.wandb

    args = args.sagemaker
    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        object.__setattr__(args, "arn", os.environ["SAGEMAKER_ARN"])

    image = get_image(
        args.user,
        region=args.region,
        profile=args.profile,
    )
    os.environ["AWS_DEFAULT_REGION"] = args.region

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    if args.local:
        from sagemaker.local import LocalSession

        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split("/")[-1]
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"
    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        job_name = "-".join([base, date_str])
        return job_name

    job_name = get_job_name(base_job_name)

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "WANDB_PROJECT": "lbm2",
    }
    with open("secrets.env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                environment[key.strip()] = value.strip().strip("\"'")

    estimator = PyTorch(
        entry_point="lbm2/main.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters={"config_path": hyperparameter_sagemaker_path},
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        job_name=job_name,
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=args.max_run * 24 * 60 * 60,  # max_run days
        input_mode="FastFile",
        environment=environment,
        keep_alive_period_in_seconds=5 * 60,  # 5 minutes
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    queue = Queue(
        queue_name=QUEUE_MAPPER[args.region][INSTANCE_MAPPER[args.instance_type]].replace("ml", args.queue_name)
    )
    queue.map(
        estimator,
        inputs=[None],
        job_names=[job_name],
        priority=args.priority,
        share_identifier="default",
        timeout={"attemptDurationSeconds": args.max_run * 24 * 60 * 60},
    )
    print(f"Queued {job_name}")


if __name__ == "__main__":
    main()
