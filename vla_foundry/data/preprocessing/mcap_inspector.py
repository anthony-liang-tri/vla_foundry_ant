#!/usr/bin/env python3
"""
MCAP Topic Inspector

Inspect topics and message types in MCAP files. Supports local and S3 paths.

Usage:
    uv run python mcap_inspector.py <mcap_path>
    uv run python mcap_inspector.py <mcap_path> --sample
    uv run python mcap_inspector.py <mcap_path> --sample --topic /joint_states

Examples:
    uv run python mcap_inspector.py /local/path/recording.mcap
    uv run python mcap_inspector.py s3://bucket/path/recording.mcap
    uv run python mcap_inspector.py s3://bucket/path/recording.mcap --sample
"""

import argparse
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader

from vla_foundry.file_utils import copy_to_temp_file


def _print_section(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def inspect_mcap(mcap_path: str, show_sample: bool = False, topic_filter: str | None = None) -> None:
    """
    Inspect MCAP file and print topic information.

    Args:
        mcap_path: Local or S3 path to MCAP file
        show_sample: If True, print sample message from each topic
        topic_filter: If provided, only show this specific topic
    """
    is_s3 = mcap_path.startswith("s3://")

    if is_s3:
        print(f"Downloading from S3: {mcap_path}")
        with copy_to_temp_file(mcap_path) as local_path:
            _inspect_local(local_path, show_sample, topic_filter)
    else:
        _inspect_local(mcap_path, show_sample, topic_filter)


def _inspect_local(local_path: str, show_sample: bool, topic_filter: str | None) -> None:
    """
    Inspect local MCAP file.

    Args:
        local_path: Path to local MCAP file
        show_sample: If True, print sample message from each topic
        topic_filter: If provided, only show this specific topic

    Raises:
        FileNotFoundError: If the MCAP file does not exist
    """
    path_obj = Path(local_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"MCAP file not found: {local_path}")

    with AnyReader([path_obj]) as reader:
        # Build topic dictionary
        topics = {
            conn.topic: conn.msgtype
            for conn in reader.connections
            if topic_filter is None or conn.topic == topic_filter
        }

        # Print topic summary
        _print_section("TOPICS")
        for topic, msgtype in topics.items():
            print(f"{topic}: {msgtype}")

        print(f"\nTotal: {len(topics)} topics")

        if show_sample:
            _print_sample_messages(reader, topics, topic_filter)


def _print_sample_messages(reader, topics: dict, topic_filter: str | None) -> None:
    """
    Print one sample message from each topic.

    Args:
        reader: AnyReader instance
        topics: Dictionary of topic -> msgtype
        topic_filter: If provided, only show this specific topic
    """
    _print_section("SAMPLE MESSAGES")

    seen = set()
    for conn, _timestamp, rawdata in reader.messages():
        # Skip if filtering to different topic or already seen this topic
        if (topic_filter is not None and conn.topic != topic_filter) or conn.topic in seen:
            continue
        seen.add(conn.topic)

        msg = reader.deserialize(rawdata, conn.msgtype)
        print(f"\n--- {conn.topic} ({conn.msgtype}) ---")
        _print_message_attributes(msg)

        # Stop once we have seen all topics
        if topic_filter is None and len(seen) >= len(topics):
            return


def _print_message_attributes(msg) -> None:
    """
    Print message attributes in a readable format.

    Skips private attributes (underscore-prefixed) and callable methods.
    For large sequences (>10 elements), prints type and length instead of
    full content to keep output readable.
    """
    # Use vars() for data attributes; fall back to dir() for objects without __dict__
    try:
        attributes = vars(msg).keys()
    except TypeError:
        attributes = [a for a in dir(msg) if not a.startswith("_")]

    for attr in attributes:
        # Skip private/dunder attributes
        if attr.startswith("_"):
            continue

        try:
            val = getattr(msg, attr)
        except AttributeError:
            continue

        # Skip methods
        if callable(val):
            continue

        # Truncate large sequences for readability
        if hasattr(val, "__len__") and len(val) > 10:
            print(f"  {attr}: [{type(val).__name__}, len={len(val)}]")
        else:
            print(f"  {attr}: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect MCAP file topics and messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mcap_path", help="Path to MCAP file (local or s3://)")
    parser.add_argument("--sample", action="store_true", help="Show sample message from each topic")
    parser.add_argument("--topic", type=str, default=None, help="Filter to specific topic")

    args = parser.parse_args()

    try:
        inspect_mcap(args.mcap_path, args.sample, args.topic)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
