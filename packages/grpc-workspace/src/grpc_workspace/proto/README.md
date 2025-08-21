# Protobuf and gRPC Artifact Generation

After modifying any Protobuf (.proto) file, regenerate the Python artifacts and commit the changes:

```
# From project root.
uv run python packages/grpc-workspace/src/grpc_workspace/proto/build_proto.py
git add .
git commit -m "Update generated Protobuf and gRPC artifacts"
```
