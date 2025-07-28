import os
import subprocess
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

import boto3
import draccus

import sagemaker
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
class SageMakerParams(TrainExperimentParams):
    local: bool = field(default=False)
    user: str = field(default=None)

    # AWS profile args
    region: str = field(default="us-west-2")
    profile: str = field(default="default")
    arn: str = field(default=None)
    s3_remote_sync: str = field(default=None)

    # Instance args
    instance_count: int = field(default=1)
    instance_type: str = field(default="p4de")

    # SageMaker queue args
    queue_name: str = field(default="ml")
    priority: int = field(default=1)

    def __post_init__(self):
        pass


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


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

    # Check this first to avoid waiting for Docker build.
    hyperparameters = {}
    experiment_params_fields = [f.name for f in fields(TrainExperimentParams)]
    for k, v in args:
        if k.startswith("data.") or k.startswith("distributed.") or k.startswith("hparams.") or k.startswith("model."):
            if v is None:
                continue
            if (
                k == "distributed.use_distributed"
                or k == "distributed.world_size"
                or k == "distributed.rank"
                or k == "distributed.local_rank"
                or k == "distributed.device"
                or k == "hparams.world_size"
            ):
                continue
            hyperparameters[k] = v
        if k in experiment_params_fields and v is not None:
            if k in ["data", "model", "distributed", "hparams"]:
                continue
            hyperparameters[k] = v
    hyperparameters["save_path"] = "/opt/ml/checkpoints"
    print(hyperparameters)

    # We probably want wandb logging and S3 saving for sagemaker runs
    assert hyperparameters.get("remote_sync") is not None
    assert hyperparameters.get("wandb")

    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        object.__setattr__(args, "arn", os.environ["SAGEMAKER_ARN"])

    if args.s3_remote_sync is None:
        assert "S3_REMOTE_SYNC" in os.environ, (
            "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        )
        object.__setattr__(args, "s3_remote_sync", os.environ["S3_REMOTE_SYNC"])
        object.__setattr__(args, "s3_remote_sync", args.s3_remote_sync.replace("us-east-1", args.region))

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
    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

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

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

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
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        code_location=output_s3,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        input_mode="FastFile",
        environment=environment,
        keep_alive_period_in_seconds=5 * 60,  # 30 minutes
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
        timeout={"attemptDurationSeconds": 10 * 24 * 60 * 60},
    )
    print(f"Queued {job_name}")


if __name__ == "__main__":
    main()
