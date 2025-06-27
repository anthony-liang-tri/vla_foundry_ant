import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.pytorch import PyTorch
from sagemaker.batch_queueing.queue import Queue

from dataclasses import fields
from lbm2.params.data_params import DataParams, add_data_params
from lbm2.params.distributed_params import DistributedParams, add_distributed_params
from lbm2.params.experiment_params import ExperimentParams, add_experiment_params
from lbm2.params.model_params import ModelParams, add_model_params
from lbm2.params.extra.vit_params import ViTParams, add_vit_params
from lbm2.params.extra.diffusion_params import DiffusionParams, add_diffusion_params

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

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

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


def parse_args():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")

    # AWS profile args
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument("--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var")

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))

    # SageMaker queue args
    parser.add_argument("--queue-name", type=str, default='ml')
    parser.add_argument("--priority", type=int, default=1, help="SageMaker FSS queue priority")

    add_data_params(parser)
    add_distributed_params(parser)
    add_experiment_params(parser)
    add_model_params(parser)
    add_vit_params(parser)
    add_diffusion_params(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Check this first to avoid waiting for Docker build.
    hyperparameters = {}
    for paramgroup in [DataParams, DistributedParams, ExperimentParams, ModelParams, ViTParams, DiffusionParams]:
        for i in fields(paramgroup):
            if i.name in args:
                if getattr(args, i.name) is False or getattr(args, i.name) is None:
                    continue
                elif getattr(args, i.name) is True:
                    hyperparameters[i.name.replace('_', '-')] = ""
                else:
                    hyperparameters[i.name.replace('_', '-')] = getattr(args, i.name)
    del hyperparameters['wandb']    # always log to wandb for sagemaker
    print(hyperparameters)

    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert (
            "SAGEMAKER_ARN" in os.environ
        ), "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]

    if args.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]
        args.s3_remote_sync = args.s3_remote_sync.replace("us-east-1", args.region)

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
        keep_alive_period_in_seconds=5 * 60,    # 30 minutes
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    queue = Queue(
        queue_name=QUEUE_MAPPER[args.region][INSTANCE_MAPPER[args.instance_type]].replace('ml', args.queue_name)
    )
    queued_jobs = queue.map(
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
