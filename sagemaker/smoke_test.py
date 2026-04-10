"""
SageMaker launch smoke test.

Verifies prerequisites without building an image or submitting a job.

Usage:
    uv run --group sagemaker python sagemaker/smoke_test.py --user firstname.lastname
"""

import argparse
import os
import subprocess
import sys

from vla_foundry.aws.s3_constants import DEFAULT_REGION

SM_PROFILE = "sagemaker"
SM_ARN = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"
SM_REGION = DEFAULT_REGION

_OK = "\033[32m OK\033[0m"
_FAIL = "\033[31m FAIL\033[0m"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _check(label, cmd, hint):
    r = _run(cmd)
    ok = r.returncode == 0
    print(f"  [{_OK if ok else _FAIL}] {label}")
    if not ok:
        print(f"        hint: {hint}")
        if r.stderr.strip():
            print(f"        {r.stderr.strip()[:200]}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--profile", default=SM_PROFILE)
    p.add_argument("--region", default=SM_REGION)
    args = p.parse_args()

    profile, region = args.profile, args.region
    print(f"SageMaker smoke test  (user={args.user}, profile={profile}, region={region})\n")

    passed = True

    # SSO session is valid — required for all subsequent AWS calls
    passed &= _check(
        f"AWS profile '{profile}' authenticated",
        ["aws", "--profile", profile, "sts", "get-caller-identity"],
        f"aws sso login --profile {profile}",
    )

    # Docker must be running before we attempt image build or ECR login
    passed &= _check(
        "Docker daemon running",
        ["docker", "info"],
        "sudo systemctl start docker",
    )

    # ECR login: fetch a short-lived token, pipe it to docker login against the
    # account registry hostname (not a full image URI — that was a past bug).
    acct = _run(
        [
            "aws",
            "--profile",
            profile,
            "--region",
            region,
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
            "--output",
            "text",
        ]
    )
    if acct.returncode == 0:
        registry = f"{acct.stdout.strip()}.dkr.ecr.{region}.amazonaws.com"
        token = _run(["aws", "ecr", "get-login-password", "--region", region, "--profile", profile])
        login = (
            _run(["docker", "login", "--username", "AWS", "--password-stdin", registry], input=token.stdout)
            if token.returncode == 0
            else token
        )
        # Also check stderr: some Docker versions exit 0 but fail to save credentials
        # (broken credential helper), which causes docker push to fail later.
        cred_err = "error storing credentials" in login.stderr
        ecr_ok = login.returncode == 0 and not cred_err
        print(f"  [{_OK if ecr_ok else _FAIL}] ECR login ({registry})")
        if not ecr_ok:
            msg = login.stderr.strip() or login.stdout.strip()
            print(f"        {msg[:200]}")
            if cred_err:
                print("        hint: broken credential helper — edit ~/.docker/config.json, remove 'credsStore'")
        passed &= ecr_ok

        # Verify the authenticated account matches the ARN we submit jobs under
        arn_account = SM_ARN.split(":")[4]
        arn_ok = acct.stdout.strip() == arn_account
        print(f"  [{_OK if arn_ok else _FAIL}] ARN account matches ({acct.stdout.strip()})")
        if not arn_ok:
            print(f"        SM_ARN targets {arn_account}")
        passed &= arn_ok
    else:
        print(f"  [{_FAIL}] ECR login (could not resolve account)")
        passed = False

    # sagemaker + boto3 are in the optional uv dep group, not installed by default
    passed &= _check(
        "sagemaker + boto3 importable",
        ["uv", "run", "--group", "sagemaker", "python", "-c", "import sagemaker, boto3"],
        "uv sync --group sagemaker",
    )

    # secrets.env is read unconditionally by launch_training.py at startup
    secrets_ok = os.path.exists("secrets.env")
    print(f"  [{_OK if secrets_ok else _FAIL}] secrets.env present")
    if not secrets_ok:
        print("        Create secrets.env with WANDB_API_KEY=... (copy from a teammate)")
    passed &= secrets_ok

    print()
    print("All checks passed." if passed else "One or more checks failed.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
