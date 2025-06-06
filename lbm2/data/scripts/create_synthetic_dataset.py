#!/usr/bin/env python3
import os
import json
import argparse
import boto3
import tempfile
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from botocore.exceptions import ClientError
from lbm2.data.datasets import SyntheticDataset, SyntheticDatasetUntokenizedText


def create_dataset(seq_len, vocab_size, dataset_size):
    """Create a synthetic dataset with the given parameters."""
    # return SyntheticDataset(seq_len, vocab_size, dataset_size)
    return SyntheticDatasetUntokenizedText(seq_len, vocab_size, dataset_size)

def save_dataset_to_webdataset(dataset, samples_per_shard=1000):
    """
    Save dataset to WebDataset format (tar files with individual samples)
    with data stored as JSON
    """
    dataloader = DataLoader(dataset, batch_size=1)
    temp_dir = tempfile.mkdtemp()
    
    # Create pattern for sharded dataset
    base_filename = os.path.join(temp_dir, "shard-%06d.tar")
    
    # Create a WebDataset sink (writer)
    sink = wds.ShardWriter(base_filename, maxcount=samples_per_shard)
    
    # Keep track of samples in each shard
    current_shard = 0
    samples_in_current_shard = 0
    manifest_data = []

    for i, (seq,) in enumerate(dataloader):
        # Track which shard we're on
        shard_num = i // samples_per_shard
        if shard_num > current_shard:
            # We've moved to a new shard, record the previous one
            manifest_data.append({
                "shard": f"shard-{current_shard:06d}",
                "num_sequences": samples_in_current_shard
            })
            current_shard = shard_num
            samples_in_current_shard = 0
        
        samples_in_current_shard += 1

        # Write the sample to the sink
        try:
            seq_out = seq.squeeze(0).tolist()
        except AttributeError:
            seq_out = seq
        sink.write({
            "__key__": f"{i:08d}",
            "json": seq_out
        })
    
    # Close the sink to ensure all data is written
    sink.close()
    
        # Add the last shard to the manifest
    if samples_in_current_shard > 0:
        manifest_data.append({
            "shard": f"shard-{current_shard:06d}",
            "num_sequences": samples_in_current_shard
        })
    
    # Write the manifest file
    manifest_path = os.path.join(temp_dir, "manifest.jsonl")
    with open(manifest_path, 'w') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + "\n")
    
    # Get list of all tar files
    shard_files = [f for f in sorted(os.listdir(temp_dir)) if f.endswith('.tar')]
    return temp_dir, shard_files, "manifest.jsonl"


def upload_to_s3(local_dir, file_list, manifest_file, s3_path):
    """Upload the WebDataset shards and manifest to S3."""
    # Parse the S3 path
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with 's3://'")
    
    s3_path = s3_path[5:]  # Remove the 's3://' prefix
    bucket_name = s3_path.split('/')[0]
    prefix = '/'.join(s3_path.split('/')[1:])
    
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    
    # Upload files to S3
    s3_client = boto3.client('s3')
    uploaded_urls = []
    
    try:
        for filename in file_list:
            local_path = os.path.join(local_dir, filename)
            s3_key = prefix + filename
            
            print(f"Uploading {filename} to s3://{bucket_name}/{s3_key}...")
            s3_client.upload_file(local_path, bucket_name, s3_key)
            uploaded_urls.append(f"s3://{bucket_name}/{s3_key}")
        
        # Upload manifest file
        manifest_path = os.path.join(local_dir, manifest_file)
        manifest_key = prefix + manifest_file
        
        print(f"Uploading {manifest_file} to s3://{bucket_name}/{manifest_key}...")
        s3_client.upload_file(manifest_path, bucket_name, manifest_key)
        manifest_url = f"s3://{bucket_name}/{manifest_key}"
        
        print(f"Successfully uploaded {len(file_list)} shards and manifest to S3")
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False, [], ""
    
    return True, uploaded_urls, manifest_url


def main():
    """Parse arguments and create and save the dataset."""
    parser = argparse.ArgumentParser(description='Create a synthetic dataset in WebDataset format (JSON) and upload to S3')
    parser.add_argument('--seq-len', type=int, default=20, help='Sequence length')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--dataset-size', type=int, default=100, help='Dataset size')
    parser.add_argument('--samples-per-shard', type=int, default=512, 
                        help='Number of samples per WebDataset shard')
    parser.add_argument('--s3-path', type=str, required=True, 
                        help='S3 path (s3://bucket-name/path/to/folder)')
    
    args = parser.parse_args()
    
    # Create the dataset
    print(f"Creating synthetic dataset with sequence length {args.seq_len}, "
          f"vocabulary size {args.vocab_size}, and {args.dataset_size} samples...")
    dataset = create_dataset(args.seq_len, args.vocab_size, args.dataset_size)
    
    # Save the dataset in WebDataset format with JSON
    print(f"Converting to WebDataset format (JSON) with {args.samples_per_shard} samples per shard...")
    temp_dir, shard_files, manifest_file = save_dataset_to_webdataset(
        dataset, 
        samples_per_shard=args.samples_per_shard
    )
    
    # Upload to S3
    print(f"Uploading WebDataset shards and manifest to {args.s3_path}...")
    success, uploaded_urls, manifest_url = upload_to_s3(temp_dir, shard_files, manifest_file, args.s3_path)
    
    if success:        
        # Clean up temporary directory
        for file in shard_files:
            os.remove(os.path.join(temp_dir, file))
        os.remove(os.path.join(temp_dir, manifest_file))
        os.rmdir(temp_dir)
        print("Temporary files cleaned up")
    else:
        print("Failed to upload dataset to S3")


if __name__ == '__main__':
    main()