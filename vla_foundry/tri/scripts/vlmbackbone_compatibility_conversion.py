#!/usr/bin/env python3
"""Convert old-format checkpoints to the new backbone-abstraction format.

Renames state dict keys:
  clip.* -> vision_language_backbone._model.*

Sample usage:
python vla_foundry/tri/scripts/vlmbackbone_compatibility_conversion.py \
    --input_model_path s3://your-bucket/checkpoints/checkpoint_3.pt \
    --output_model_path s3://your-bucket/checkpoints/checkpoint_3_converted.pt
"""

import argparse

import torch

OLD_TO_NEW_PREFIXES = [
    ("clip.", "vision_language_backbone._model."),
]


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in OLD_TO_NEW_PREFIXES:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert old checkpoint format to new backbone-abstraction format.")
    parser.add_argument("--input_model_path", required=True, help="Path to the old checkpoint (local or s3://)")
    parser.add_argument(
        "--output_model_path", required=True, help="Path to save the converted checkpoint (local or s3://)"
    )
    args = parser.parse_args()

    input_path = args.input_model_path
    output_path = args.output_model_path

    # Handle S3 paths
    if input_path.startswith("s3://"):
        import subprocess
        import tempfile

        local_input = tempfile.mktemp(suffix=".pt")
        print(f"Downloading {input_path} ...")
        subprocess.run(["aws", "s3", "cp", input_path, local_input], check=True)
    else:
        local_input = input_path

    print(f"Loading checkpoint from {local_input} ...")
    checkpoint = torch.load(local_input, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        old_sd = checkpoint["state_dict"]
        new_sd = convert_state_dict(old_sd)
        renamed_count = sum(1 for old_k, new_k in zip(old_sd.keys(), new_sd.keys(), strict=False) if old_k != new_k)
        print(f"Renamed {renamed_count} / {len(old_sd)} keys in state_dict")
        checkpoint["state_dict"] = new_sd
    else:
        raise ValueError(f"Unexpected checkpoint format. Top-level keys: {list(checkpoint.keys())}")

    if output_path.startswith("s3://"):
        import subprocess
        import tempfile

        local_output = tempfile.mktemp(suffix=".pt")
        print(f"Saving checkpoint to {local_output} ...")
        torch.save(checkpoint, local_output)
        print(f"Uploading to {output_path} ...")
        subprocess.run(["aws", "s3", "cp", local_output, output_path], check=True)
    else:
        print(f"Saving checkpoint to {output_path} ...")
        torch.save(checkpoint, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
