from pathlib import Path

import grpc_tools.protoc


def build_proto(dest_dir: str | None | Path = None):
    """
    Generate Protobuf and gRPC messages for known Proto files.

    Args:
        dest_dir:
            Destination directory for the auto-generated Protobuf and gRPC
            message files to be placed within. By default, this path will
            point to the source directory under 'grpc_workspace/proto' in order
            to overwrite the generated Protobuf and gRPC files there when
            updating the Proto files themselves.
    """
    if dest_dir:
        dest_dir = Path(dest_dir)
        if not dest_dir.is_dir():
            raise ValueError(f"The supplied 'dest_dir' is not a directory or does not exist: {dest_dir}")

    proto_names = ["health", "GetPolicyMetadata", "PolicyReset", "PolicyStep"]
    for proto_name in proto_names:
        # Location of *.proto files.
        proto_path = Path(f"packages/grpc-workspace/src/grpc_workspace/proto/{proto_name}.proto").absolute().resolve()

        src_dir = proto_path.parent
        # Directory where we would like to output resulting python files.
        if not dest_dir:
            dest_dir = src_dir

        # Build python and gRPC files.
        grpc_tools.protoc.main(
            [
                "protoc",
                "--proto_path=" + str(src_dir),
                "--python_out=" + str(dest_dir),
                "--grpc_python_out=" + str(dest_dir),
                f"{proto_name}.proto",
            ]
        )

        # For imports to work in both lbm and anzu (with bazel) we need to
        # specify the absolute path from the nearest ancestor on the path.
        # TODO: Determine better way to handle this.
        file_to_patch = str(dest_dir) + f"/{proto_name}_pb2_grpc.py"
        with open(file_to_patch, "r") as file:
            lines = file.readlines()

        find_str = f"import {proto_name}_pb2 as {proto_name}__pb2"
        repl_str = f"from grpc_workspace.proto import {proto_name}_pb2 as {proto_name}__pb2\n"

        with open(file_to_patch, "w") as file:
            for line in lines:
                if line.strip().startswith(find_str):
                    file.write(repl_str)
                else:
                    file.write(line)


if __name__ == "__main__":
    build_proto()
