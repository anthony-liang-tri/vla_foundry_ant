"""Tests for protobuf compatibility across the project.

These tests ensure that the installed protobuf version is compatible with the
generated protobuf files in the grpc-workspace package, preventing runtime
import errors that can occur when there's a version mismatch.
"""

import re
from pathlib import Path

import pytest


def get_protobuf_version():
    """Get the installed protobuf version."""
    import google.protobuf

    return google.protobuf.__version__


def get_generated_protobuf_version():
    """Parse the protobuf version from generated files."""
    # Go up to project root (tests/dependencies/inference -> project root)
    project_root = Path(__file__).parent.parent.parent.parent
    proto_dir = project_root / "packages" / "grpc-workspace" / "src" / "grpc_workspace" / "proto"

    for proto_file in proto_dir.glob("*_pb2.py"):
        with open(proto_file, "r") as f:
            content = f.read()
            # Look for "# Protobuf Python Version: X.Y.Z" comment
            match = re.search(r"# Protobuf Python Version: (\d+\.\d+\.\d+)", content)
            if match:
                return match.group(1)

    return None


def test_protobuf_imports():
    """Test that all generated protobuf files can be imported successfully."""
    from grpc_workspace.proto import GetPolicyMetadata_pb2, PolicyReset_pb2, PolicyStep_pb2, health_pb2

    assert PolicyStep_pb2 is not None
    assert PolicyReset_pb2 is not None
    assert GetPolicyMetadata_pb2 is not None
    assert health_pb2 is not None


def test_protobuf_version_compatibility():
    """Test that protobuf version is compatible with generated files."""
    installed_version = get_protobuf_version()
    generated_version = get_generated_protobuf_version()

    major_version = int(installed_version.split(".")[0])
    if generated_version and int(generated_version.split(".")[0]) >= 5 and major_version < 5:
        pytest.fail(
            f"Protobuf version mismatch: generated code expects >= {generated_version} "
            f"but installed version is {installed_version}. "
            f"Generated files need runtime_version module which is not available."
        )

    # If we reach here, versions should be compatible


def test_grpc_workspace_can_be_imported():
    """Test that grpc_workspace package and protobuf modules can be imported without errors."""
    import grpc_workspace
    from grpc_workspace.lbm_policy_client import LbmPolicyClient
    from grpc_workspace.lbm_policy_server import LbmPolicyServer
    from grpc_workspace.proto import PolicyStep_pb2

    assert grpc_workspace is not None
    assert PolicyStep_pb2 is not None
    assert LbmPolicyServer is not None
    assert LbmPolicyClient is not None


def test_protobuf_version_matches_generated_code():
    """Test that installed protobuf version is compatible with generated code."""
    installed_version = get_protobuf_version()
    generated_version = get_generated_protobuf_version()

    assert generated_version is not None, "Could not determine protobuf version from generated files"

    installed_major = int(installed_version.split(".")[0])
    generated_major = int(generated_version.split(".")[0])

    if generated_major >= 5 and installed_major < 5:
        pytest.fail(
            f"Protobuf version mismatch: generated code expects >= {generated_version} "
            f"but installed version is {installed_version}. "
            f"Either upgrade protobuf or regenerate protobuf files."
        )


def test_runtime_version_availability():
    """Test runtime_version availability based on protobuf version."""
    installed_version = get_protobuf_version()
    major_version = int(installed_version.split(".")[0])

    if major_version >= 5:
        from google.protobuf import runtime_version

        assert runtime_version is not None
    else:
        generated_version = get_generated_protobuf_version()
        if generated_version and int(generated_version.split(".")[0]) >= 5:
            pytest.fail(
                f"Generated protobuf files expect version >= {generated_version} "
                f"but installed version {installed_version} does not support runtime_version. "
                f"Regenerate protobuf files with current protobuf version."
            )
