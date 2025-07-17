import json
import os

import webdataset as wds
from tqdm import trange


def create_synthetic_json_shards(output_dir, num_shards=4, samples_per_shard=8192, tokens_per_sample=2048):
    os.makedirs(output_dir, exist_ok=True)
    manifest_data = []

    for shard_id in range(num_shards):
        shard_name = f"shard-{shard_id:08d}"
        shard_path = os.path.join(output_dir, shard_name + ".tar")
        sink = wds.ShardWriter(shard_path, maxcount=samples_per_shard)

        for i in trange(samples_per_shard, desc=f"Shard {shard_id}", leave=False):
            global_index = shard_id * samples_per_shard + i
            fake_tokens = list(range(1, tokens_per_sample + 1))  # Example token IDs
            sink.write({"__key__": f"{global_index:08d}", "json": {"input_ids": fake_tokens}})

        sink.close()
        manifest_data.append({"shard": shard_name, "num_sequences": samples_per_shard})

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in manifest_data:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Created {num_shards} JSON-based tar shards in: {output_dir}")
    return output_dir


# Example usage:
create_synthetic_json_shards("shards_json", num_shards=4)
