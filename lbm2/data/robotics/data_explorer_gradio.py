#!/usr/bin/env python3
"""
Gradio-based Robotics Data Explorer

This tool provides a gradio web interface for exploring preprocessed robotics data
with interactive trajectory visualization overlaid on camera images.

Usage:
    python lbm2/data/robotics/data_explorer_gradio.py \
        --dataset-path /path/to/processed/dataset/ \
        --max-samples 100 \
        --port 7860

Features:
    - Modern web-based interface using Gradio
    - Interactive sliders and dropdowns
    - Real-time trajectory overlay on camera images
    - 3D trajectory visualization
    - Multiple camera support
    - Gripper state visualization
    - Sample metadata display
"""

import argparse

import fsspec
import yaml

from lbm2.data.robotics.gradio_dataloader import RoboticsDataLoader
from lbm2.data.robotics.gradio_interface import GradioDataExplorer
from lbm2.params.data_params import LBMDataParams


def main():
    parser = argparse.ArgumentParser(description="Gradio-based Robotics Data Explorer")

    parser.add_argument("--dataset-path", required=True, help="Path to processed robotics dataset")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of samples to load (-1 for all)")
    parser.add_argument("--max-shards", type=int, default=-1, help="Maximum number of shards to load (-1 for all)")
    parser.add_argument("--trajectory-length", type=int, default=50, help="Number of trajectory points to display")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to run Gradio interface on (auto-select if not specified)"
    )
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    parser.add_argument(
        "--use-dataloader",
        action="store_true",
        default=True,
        help="Use dataloader pipeline for loading (default: True, use --no-use-dataloader for direct file loading)",
    )

    args = parser.parse_args()

    if args.dataset_path.endswith("/"):
        args.dataset_path = args.dataset_path[:-1]

    print("🔍 Loading robotics data...")

    # Load data using direct file access
    config_path = "lbm2/config_presets/data/lbm_data_params.yaml"
    with fsspec.open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Override fields for the data explorer
    config_dict.update(
        {
            "num_workers": 0,
            "seed": 42,
            "processor": "google/paligemma-3b-pt-224",
            "add_action_token": False,
            "seq_len": 512,
            "dataset_statistics": [f"{args.dataset_path}/dataset_statistics.json"],
            "dataset_manifest": [f"{args.dataset_path}/manifest.jsonl"],
            "normalization": {"enabled": False},
        }
    )

    params = LBMDataParams.from_dict(config_dict)

    # Create data loader and load samples
    data_loader = RoboticsDataLoader(
        params, max_samples=args.max_samples, max_shards=args.max_shards, use_dataloader=False
    )
    samples = data_loader.load_samples_auto()

    if not samples:
        print("❌ No samples loaded!")
        return

    print(f"✅ Loaded {len(samples)} samples")

    # Create explorer
    print("🚀 Starting Gradio data explorer...")
    explorer = GradioDataExplorer(samples, trajectory_length=args.trajectory_length)

    # Create and launch interface
    interface = explorer.create_interface()

    print("📊 Features:")
    print("  - Interactive sample navigation with slider")
    print("  - Camera selection dropdown")
    print("  - Real-time trajectory overlay on images")
    print("  - 3D trajectory visualization")
    print("  - Gripper state indicators")
    print("  - Sample metadata display")
    print("  - Configurable shard and sample limits for faster loading")
    print("  - Choice between dataloader pipeline and direct file loading")

    # Handle port selection
    launch_kwargs = {"share": args.share, "show_error": True}

    if args.port is not None:
        launch_kwargs["server_port"] = args.port
        print(f"🌐 Launching web interface on port {args.port}")
    else:
        print("🌐 Launching web interface on auto-selected port")

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
