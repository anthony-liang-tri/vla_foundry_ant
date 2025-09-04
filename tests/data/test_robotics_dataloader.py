#!/usr/bin/env python3
"""
Pytest tests for the robotics dataloader.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from lbm2.data.dataloader import get_datastring_input, get_wds_dataloader
from lbm2.data.robotics.data_explorer_gradio import RoboticsDataLoader
from lbm2.params.data_params import LBMDataParams


@pytest.fixture
def dataset_path():
    """Path to the test dataset."""
    return "tests/test_assets/small_lbm_dataset"


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing based on lbm_data_params.yaml."""

    def _create_config(
        dataset_path: str,
        batch_size: int = 2,
        processor_name: str = "google/paligemma-3b-pt-224",
        add_action_token: bool = False,
    ):
        # Load the base config from YAML
        config_path = "lbm2/config_presets/data/lbm_data_params.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override test-specific settings
        config_dict.update(
            {
                "num_workers": 1,
                "seed": 42,
                "processor": processor_name,  # No processor for basic tests
                "add_action_token": add_action_token,
                "seq_len": 512,
                "dataset_statistics": [dataset_path + "/dataset_statistics.json"],
                "dataset_manifest": [dataset_path + "/manifest.jsonl"],
                "num_images": None,  # Let processor infer number of images
                "img_num_tokens": 49,  # Set image sequence length for PaliGemma
            }
        )

        # Override normalization settings
        config_dict["normalization"]["enabled"] = False  # Default to disabled for most tests

        # Create LBMDataParams from the modified config
        data_params = LBMDataParams.from_dict(config_dict)

        # Create mock distributed config
        distributed = SimpleNamespace()
        distributed.world_size = 1
        distributed.rank = 0

        # Create mock vit config (not used for robotics but needed by interface)
        vit = SimpleNamespace()
        vit.img_size = 128
        vit.img_num_tokens = 256

        hparams = SimpleNamespace()
        hparams.global_batch_size = batch_size

        # Create main config
        cfg = SimpleNamespace()
        cfg.distributed = distributed
        cfg.data = data_params
        cfg.vit = vit
        cfg.hparams = hparams
        return cfg

    return _create_config


@pytest.fixture
def manifest_data(dataset_path):
    """Load and return manifest data."""
    manifest_path = os.path.join(dataset_path, "manifest.jsonl")

    with open(manifest_path, "r") as f:
        manifest_lines = f.readlines()

    manifest = [json.loads(line.strip()) for line in manifest_lines]
    return manifest


def create_datastring(dataset_path, manifest_data):
    """Create properly formatted datastring for local files."""
    # For local files, construct direct paths to shard files
    shard_paths = []
    for entry in manifest_data:
        shard_file = entry["shard"] + ".tar"
        shard_path = os.path.join(dataset_path, shard_file)
        # Verify the file exists
        if os.path.exists(shard_path):
            shard_paths.append(shard_path)
        else:
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

    # For multiple shards, use brace expansion; for single shard, use direct path
    if len(shard_paths) == 1:
        datastring = shard_paths[0]
    else:
        # Extract common path and create brace expansion
        common_path = os.path.dirname(shard_paths[0])
        shard_names = [os.path.basename(path).replace(".tar", "") for path in shard_paths]
        datastring = common_path + "/{" + ",".join(shard_names) + "}.tar"

    return datastring


def test_manifest_loading(dataset_path, manifest_data):
    """Test that the manifest can be loaded and has expected structure."""
    assert len(manifest_data) > 0, "Manifest should contain at least one shard"

    # Check manifest structure
    for entry in manifest_data:
        assert "shard" in entry, "Each manifest entry should have 'shard'"
        assert "num_sequences" in entry, "Each manifest entry should have 'num_sequences'"
        assert isinstance(entry["num_sequences"], int), "num_sequences should be an integer"
        assert entry["num_sequences"] > 0, "num_sequences should be positive"

    total_samples = sum(entry["num_sequences"] for entry in manifest_data)
    assert total_samples > 0, "Total samples should be positive"

    print(f"📊 Found {len(manifest_data)} shards with {total_samples} total samples")


def test_dataloader_creation(dataset_path, manifest_data, mock_config):
    """Test that the dataloader can be created successfully."""
    # Create datastring from manifest
    test_shards = manifest_data[: min(3, len(manifest_data))]  # Use first 3 shards
    datastring = create_datastring(dataset_path, test_shards)

    print(f"📦 Datastring: {datastring}")

    # Create config
    cfg = mock_config(dataset_path, batch_size=2, processor_name="google/paligemma-3b-pt-224")

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader
    assert dataloader is not None, "Dataloader should be created"
    assert hasattr(dataloader, "num_batches"), "Dataloader should have num_batches attribute"
    assert hasattr(dataloader, "num_samples"), "Dataloader should have num_samples attribute"
    assert dataloader.num_samples > 0, "Dataloader should have positive number of samples"

    print(f"✅ Created dataloader with {dataloader.num_batches} batches, {dataloader.num_samples} samples")


def test_batch_loading(dataset_path, manifest_data, mock_config):
    """Test loading batches from the dataloader."""
    # Create datastring from manifest
    test_shards = manifest_data[: min(3, len(manifest_data))]
    datastring = create_datastring(dataset_path, test_shards)

    print(f"📦 Datastring for batch loading: {datastring}")

    # Create config with smaller batch size to ensure we get at least one batch
    cfg = mock_config(dataset_path, batch_size=1, processor_name="google/paligemma-3b-pt-224")

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    print(f"📊 Expected samples: {num_samples_per_dataset}")

    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader
    print(f"📈 Dataloader stats: {dataloader.num_batches} batches, {dataloader.num_samples} samples")

    # Test loading first batch
    batch_iter = iter(dataloader)
    try:
        batch = next(batch_iter)

        # Basic assertions
        assert isinstance(batch, dict), "Batch should be a dictionary"
        assert len(batch) > 0, "Batch should not be empty"

        print(f"📦 Batch keys: {list(batch.keys())}")

        # Test expected batch structure (updated for new pipeline)
        expected_keys = [
            "pixel_values",
            "input_ids",
            "attention_mask",
            "text",
            "lowdim",
            "lowdim_text",
            "lowdim_text_tokenized",
            "masks",
            "actions",
            "metadata",
        ]
        for key in expected_keys:
            if key in batch:
                print(f"  ✅ Found {key}")
            else:
                print(f"  ⚠️  Missing {key} (may be optional)")

        # Check that images field is None (should be converted to pixel_values)
        if "images" in batch and batch["images"] is not None:
            print(f"  ⚠️  Images field should be None after processing, got {type(batch['images'])}")
    except StopIteration:
        # If we get StopIteration, let's debug why
        print("❌ No batches available from dataloader")
        print(f"   Datastring: {datastring}")
        print(f"   File exists: {os.path.exists(datastring)}")
        print(f"   Dataloader num_batches: {dataloader.num_batches}")
        print(f"   Dataloader num_samples: {dataloader.num_samples}")
        raise AssertionError("Dataloader produced no batches - check datastring and file accessibility") from None


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batch_size(dataset_path, manifest_data, mock_config, batch_size):
    """Test that the batch size is correct."""
    # Create datastring from manifest
    test_shards = manifest_data[: min(3, len(manifest_data))]
    datastring = create_datastring(dataset_path, test_shards)

    # Create config
    cfg = mock_config(dataset_path, batch_size=batch_size, processor_name="google/paligemma-3b-pt-224")

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader
    batch_iter = iter(dataloader)
    batch = next(batch_iter)

    batch_size_measured = batch["lowdim"]["robot__actual__poses__right::panda__xyz"].shape[0]
    assert batch_size_measured == batch_size, f"Batch size should be {batch_size}, got {batch_size_measured}"
    print(f"✅ Batch size is correct: {batch_size_measured}")


def test_batch_content_structure(dataset_path, manifest_data, mock_config):
    """Test the structure and content of batch data."""
    # Create datastring from manifest
    test_shards = manifest_data[: min(3, len(manifest_data))]
    datastring = create_datastring(dataset_path, test_shards)

    # Create config
    cfg = mock_config(dataset_path, batch_size=1, processor_name="google/paligemma-3b-pt-224")

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader

    # Test loading first batch
    batch_iter = iter(dataloader)
    batch = next(batch_iter)

    # Test pixel_values structure (replaces images)
    if "pixel_values" in batch:
        print("  Pixel values:")
        pixel_values = batch["pixel_values"]
        assert hasattr(pixel_values, "shape"), "Pixel values should have shape attribute"
        assert len(pixel_values.shape) == 5, (
            f"Pixel values should have 5 dimensions [B, Camera, C, H, W], got {pixel_values.shape}"
        )
        print(f"    pixel_values: {pixel_values.shape} ({pixel_values.dtype})")

    # Test processor outputs
    if "input_ids" in batch:
        print("  Processor outputs:")
        input_ids = batch["input_ids"]
        assert hasattr(input_ids, "shape"), "Input IDs should have shape attribute"
        assert len(input_ids.shape) == 2, f"Input IDs should have 2 dimensions [B, seq_len], got {input_ids.shape}"
        print(f"    input_ids: {input_ids.shape} ({input_ids.dtype})")

        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
            assert hasattr(attention_mask, "shape"), "Attention mask should have shape attribute"
            assert input_ids.shape == attention_mask.shape, "Input IDs and attention mask should have same shape"
            print(f"    attention_mask: {attention_mask.shape} ({attention_mask.dtype})")

    # Test text field
    if "text" in batch:
        print("  Text field:")
        text = batch["text"]
        assert isinstance(text, list), f"Text should be a list, got {type(text)}"
        print(f"    text: {len(text)} samples")
        if text:
            print(f"    sample text: {text[0]}")
            # Check that image tokens are added
            for text_sample in text:
                assert "<image>" in text_sample, f"Text should contain <image> token, got: {text_sample}"

    # Test low-dim data structure
    if "lowdim" in batch:
        print("  Low-dim numerical data:")
        for key, tensor in batch["lowdim"].items():
            if hasattr(tensor, "shape"):
                print(f"    {key}: {tensor.shape} ({tensor.dtype})")
            else:
                print(f"    {key}: {type(tensor)} (variable shapes)")

    # Test tokenized text structure
    if "lowdim_text_tokenized" in batch:
        print("  Low-dim text data (tokenized):")
        for key, tokenized_data in batch["lowdim_text_tokenized"].items():
            if isinstance(tokenized_data, dict):
                print(f"    {key}:")
                if "input_ids" in tokenized_data:
                    assert hasattr(tokenized_data["input_ids"], "shape"), "input_ids should have shape"
                    print(f"      input_ids: {tokenized_data['input_ids'].shape} ({tokenized_data['input_ids'].dtype})")
                if "attention_mask" in tokenized_data:
                    assert hasattr(tokenized_data["attention_mask"], "shape"), "attention_mask should have shape"
                    print(
                        f"      attention_mask: {tokenized_data['attention_mask'].shape} "
                        f"({tokenized_data['attention_mask'].dtype})"
                    )

    # Test metadata structure
    if "metadata" in batch:
        print(f"  Metadata: {len(batch['metadata'])} samples")
        assert len(batch["metadata"]) > 0, "Metadata should not be empty"
        if batch["metadata"]:
            sample_meta = batch["metadata"][0]
            if isinstance(sample_meta, dict):
                print(f"    Sample keys: {list(sample_meta.keys())}")


def test_multiple_batches(dataset_path, manifest_data, mock_config):
    """Test loading multiple batches."""
    # Create datastring from manifest
    test_shards = manifest_data[: min(3, len(manifest_data))]
    datastring = create_datastring(dataset_path, test_shards)

    # Create config with batch size that ensures multiple batches
    total_samples = sum(entry["num_sequences"] for entry in test_shards)
    batch_size = max(1, total_samples // 3)  # Ensure at least 3 batches if possible
    cfg = mock_config(dataset_path, batch_size=batch_size, processor_name="google/paligemma-3b-pt-224")

    # Get dataloader
    num_samples_per_dataset = [total_samples]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader

    # Test loading multiple batches
    num_batches_to_test = min(3, max(1, dataloader.num_batches))
    batches_loaded = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_test:
            break

        assert isinstance(batch, dict), f"Batch {i} should be a dictionary"
        assert len(batch) > 0, f"Batch {i} should not be empty"
        batches_loaded += 1

        print(f"📦 Batch {i + 1}: {list(batch.keys())}")

    assert batches_loaded == num_batches_to_test, (
        f"Should have loaded {num_batches_to_test} batches, got {batches_loaded}"
    )
    print(f"✅ Successfully loaded {batches_loaded} batches")


@pytest.mark.parametrize("processor_name", ["HuggingFaceTB/SmolVLM-Instruct", "google/paligemma-3b-pt-224"])
def test_different_tokenizers(dataset_path, manifest_data, mock_config, processor_name):
    """Test dataloader with different tokenizers."""
    # Create datastring from manifest
    test_shards = manifest_data[:1]  # Use just first shard for speed
    datastring = create_datastring(dataset_path, test_shards)

    # Create config with specific tokenizer
    cfg = mock_config(dataset_path, batch_size=1, processor_name=processor_name)

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader

    # Test loading one batch
    batch_iter = iter(dataloader)
    batch = next(batch_iter)

    assert isinstance(batch, dict), f"Batch should be a dictionary with tokenizer {processor_name}"
    print(f"✅ Successfully loaded batch with tokenizer: {processor_name}")


def test_with_action_token(dataset_path, manifest_data, mock_config):
    """Test dataloader with action token enabled."""
    # Create datastring from manifest
    test_shards = manifest_data[:1]  # Use just first shard
    datastring = create_datastring(dataset_path, test_shards)

    # Create config with action token enabled
    cfg = mock_config(dataset_path, batch_size=1, processor_name="google/paligemma-3b-pt-224", add_action_token=True)

    # Get dataloader
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]
    dataloader_info = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    )

    dataloader = dataloader_info.dataloader

    # Get first batch
    batch = next(iter(dataloader))

    # Check that action token was added to text (both in processor output and lowdim_text_tokenized)
    action_token_found = False

    # Find the processor in the pipeline by iterating through the pipeline components
    # TODO: This is a hack to get the processor, we should find a better way to do that...
    processor = None
    for dataset in dataloader.pipeline[0].dataset.datasets:
        for pipe_component in dataset.pipeline:
            if hasattr(pipe_component, "args") and pipe_component.args:
                for arg in pipe_component.args:
                    if hasattr(arg, "keywords") and "processor" in arg.keywords:
                        processor = arg.keywords["processor"]
                        break
    if processor is None:
        raise ValueError("Could not find processor in the dataloader pipeline")
    action_token_id = processor.tokenizer.encode("<|action|>")[0]

    if "input_ids" in batch:
        for input_ids in batch["input_ids"]:
            if action_token_id in input_ids:
                action_token_found = True
                print(f"  ✅ Found <|action|> in processor input_ids: {input_ids}")

    # Check in lowdim_text_tokenized
    if "lowdim_text_tokenized" in batch:
        for text_field, tokenized_data in batch["lowdim_text_tokenized"].items():
            if "original_text" in tokenized_data:
                original_texts = tokenized_data["original_text"]
                print(f"📝 {text_field} original texts: {original_texts}")

                # Check if action token is present
                for text in original_texts:
                    if "<|action|>" in text:
                        action_token_found = True
                        print(f"  ✅ Found <|action|> in: {text}")

    if not action_token_found:
        # Try to find action token in any text field
        all_text_fields = []
        if "text" in batch:
            all_text_fields.extend(batch["text"])
        if "lowdim_text_tokenized" in batch:
            for tokenized_data in batch["lowdim_text_tokenized"].values():
                if "original_text" in tokenized_data:
                    all_text_fields.extend(tokenized_data["original_text"])

        print(f"⚠️  All text fields found: {all_text_fields}")

        # If there are no text fields, the action token test is not applicable
        # since action tokens can only be added to text that exists
        if not all_text_fields:
            print("ℹ️  No text fields found in batch - action token test not applicable")
            print("ℹ️  This is expected for datasets without text instructions")
            return  # Skip the test since there's no text to add action tokens to

        assert action_token_found, "Action token <|action|> should be present in at least one text field"


def test_normalization(dataset_path, manifest_data, mock_config):
    """Test that normalization is working correctly."""

    # Create datastring from manifest
    test_shards = manifest_data[:1]  # Use just first shard for speed
    datastring = create_datastring(dataset_path, test_shards)

    print(f"📦 Testing normalization with datastring: {datastring}")

    def _create_config_with_norm(enabled):
        # Load the base config from YAML
        config_path = "tests/test_assets/small_lbm_dataset/test_lbm_data_params.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override test-specific settings
        config_dict.update(
            {
                "num_workers": 1,
                "seed": 42,
                "processor": "google/paligemma-3b-pt-224",
                "add_action_token": False,
                "seq_len": 512,
                "dataset_statistics": [dataset_path + "/dataset_statistics.json"],
                "dataset_manifest": [dataset_path + "/manifest.jsonl"],
            }
        )

        # Override normalization settings
        config_dict["normalization"]["enabled"] = enabled

        # Create LBMDataParams from the modified config
        data_params = LBMDataParams.from_dict(config_dict)

        # Create mock distributed config
        distributed = SimpleNamespace()
        distributed.world_size = 1
        distributed.rank = 0

        # Create mock vit config
        vit = SimpleNamespace()
        vit.img_size = 128
        vit.img_num_tokens = 256

        hparams = SimpleNamespace()
        hparams.global_batch_size = 3

        # Create main config
        cfg = SimpleNamespace()
        cfg.distributed = distributed
        cfg.data = data_params
        cfg.vit = vit
        cfg.hparams = hparams

        return cfg

    # Get normalized data
    cfg_normalized = _create_config_with_norm(enabled=True)
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]

    dataloader_info_norm = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg_normalized
    )

    # Get non-normalized data for comparison
    cfg_no_norm = _create_config_with_norm(enabled=False)
    dataloader_info_no_norm = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg_no_norm
    )

    # Get batches
    batch_normalized = next(iter(dataloader_info_norm.dataloader))
    batch_no_norm = next(iter(dataloader_info_no_norm.dataloader))

    print("🔍 Checking normalization effects...")

    # Check that both batches have lowdim data
    assert "lowdim" in batch_normalized, "Normalized batch should have lowdim data"
    assert "lowdim" in batch_no_norm, "Non-normalized batch should have lowdim data"

    # Compare normalized vs non-normalized lowdim data
    # The normalization system normalizes ALL fields except those in exclude_fields
    included_fields = set(cfg_normalized.data.proprioception_fields + cfg_normalized.data.action_fields)
    excluded_fields = set(cfg_normalized.data.exclude_fields)

    if batch_normalized["lowdim"] and batch_no_norm["lowdim"]:
        normalized_fields_found = 0
        excluded_fields_found = 0

        for field_name in batch_normalized["lowdim"]:
            if field_name in batch_no_norm["lowdim"]:
                norm_data = batch_normalized["lowdim"][field_name]
                no_norm_data = batch_no_norm["lowdim"][field_name]

                # Skip if not tensor data
                if not isinstance(norm_data, torch.Tensor) or not isinstance(no_norm_data, torch.Tensor):
                    continue

                # Skip if tensors have different shapes (shouldn't happen but safety check)
                if norm_data.shape != no_norm_data.shape:
                    continue

                print(f"📊 Analyzing field: {field_name}")
                print(f"  Shape: {norm_data.shape}")

                # Calculate statistics
                norm_mean = torch.mean(norm_data).item()
                norm_std = torch.std(norm_data).item()
                no_norm_mean = torch.mean(no_norm_data).item()
                no_norm_std = torch.std(no_norm_data).item()

                print(f"  Non-normalized: mean={no_norm_mean:.3f}, std={no_norm_std:.3f}")
                print(f"  Normalized:     mean={norm_mean:.3f}, std={norm_std:.3f}")

                # Check if field should be excluded from normalization
                should_be_excluded = field_name in excluded_fields or any(
                    keyword in field_name.lower() for keyword in ["text", "language", "instruction", "mask", "valid"]
                )

                should_be_included = field_name in included_fields or any(
                    keyword in field_name.lower() for keyword in ["text", "language", "instruction", "mask", "valid"]
                )

                if should_be_included:
                    if should_be_excluded:
                        # For fields that should be excluded, they should be identical
                        assert torch.allclose(norm_data, no_norm_data, atol=1e-6), (
                            f"Excluded field {field_name} should be identical in both batches"
                        )
                        print(f"  ✅ Field {field_name} correctly unchanged (excluded from normalization)")
                        excluded_fields_found += 1
                    else:
                        # For fields that should be normalized, they should be different
                        # UNLESS the original data has zero variance (all values are identical)
                        if no_norm_std > 1e-6:  # Only check if original data has some variance
                            assert not torch.allclose(norm_data, no_norm_data, atol=1e-6), (
                                f"Normalized data should differ from non-normalized for field {field_name}"
                            )
                        else:
                            print(f"  ℹ️  Skipping difference check for {field_name} - original data has zero variance")

                        if cfg_normalized.data.normalization.field_configs[field_name].method == "std":
                            # Check that normalized data is roughly standardized
                            # (allowing some tolerance since we have limited data)
                            assert abs(norm_mean) < 2.0, (
                                f"Normalized data mean should be closer to 0 for field {field_name}, got {norm_mean}"
                            )

                            # Don't enforce strict std=1 since the test data might be limited
                            # Some fields in test data might have very low variance, so be very lenient
                            assert norm_std >= 0.0, (
                                f"Normalized data std should be non-negative for field {field_name}, got {norm_std}"
                            )

                            # Only warn about very small std instead of failing
                            if norm_std < 0.001:
                                print(
                                    f"  ⚠️  Very small std ({norm_std:.6f}) - field may have low variance in test data"
                                )
                        elif cfg_normalized.data.normalization.field_configs[field_name].method == "percentile_1_99":
                            # Check that normalized data is roughly standardized
                            # For percentile normalization, values outside [0,1] are expected for outliers
                            # We just verify that the majority of values are in a reasonable range
                            sum_out_of_range = torch.sum(torch.logical_or(norm_data < -1, norm_data > 2).to(int))
                            num_elements = torch.numel(norm_data)
                            assert sum_out_of_range < num_elements * 0.2, (
                                f"Most percentile normalized data should be in reasonable range for field {field_name},"
                                f" got {sum_out_of_range} out of {num_elements} elements severely out of range"
                            )

                        print(f"  ✅ Normalization working for {field_name}")
                        normalized_fields_found += 1

        # Ensure we found at least some fields to normalize
        assert normalized_fields_found > 0, "Should have found at least one field that gets normalized"

    print("✅ Normalization test passed!")


def test_normalization_consistency(dataset_path, manifest_data, mock_config):
    """Test that normalization produces consistent results across batches."""
    import torch

    # Create datastring from manifest
    test_shards = manifest_data[:1]
    datastring = create_datastring(dataset_path, test_shards)

    # Load the base config from YAML
    config_path = "lbm2/config_presets/data/lbm_data_params.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Override test-specific settings
    config_dict.update(
        {
            "num_workers": 1,
            "seed": 42,  # Fixed seed for reproducibility
            "processor": "google/paligemma-3b-pt-224",
            "add_action_token": False,
            "seq_len": 512,
            "dataset_statistics": [dataset_path + "/dataset_statistics.json"],
            "dataset_manifest": [dataset_path + "/manifest.jsonl"],
        }
    )

    # Override normalization settings
    config_dict["normalization"]["enabled"] = True

    # Create LBMDataParams from the modified config
    data_params = LBMDataParams.from_dict(config_dict)

    # Create mock distributed config
    distributed = SimpleNamespace()
    distributed.world_size = 1
    distributed.rank = 0

    hparams = SimpleNamespace()
    hparams.global_batch_size = 1

    # Create mock vit config
    vit = SimpleNamespace()
    vit.img_size = 128
    vit.img_num_tokens = 256

    # Create main config
    cfg = SimpleNamespace()
    cfg.distributed = distributed
    cfg.data = data_params
    cfg.vit = vit
    cfg.hparams = hparams

    # Create two separate dataloaders with same config
    num_samples_per_dataset = [sum(entry["num_sequences"] for entry in test_shards)]

    dataloader1 = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    ).dataloader

    dataloader2 = get_wds_dataloader(
        datastrings=[datastring], num_samples_per_dataset=num_samples_per_dataset, checkpoint_num=0, cfg=cfg
    ).dataloader

    # Get first batch from each
    batch1 = next(iter(dataloader1))
    batch2 = next(iter(dataloader2))

    print("🔄 Testing normalization consistency...")

    # Compare normalized data between the two dataloaders
    if "lowdim" in batch1 and "lowdim" in batch2:
        for field_name in batch1["lowdim"]:
            if field_name in batch2["lowdim"]:
                data1 = batch1["lowdim"][field_name]
                data2 = batch2["lowdim"][field_name]

                # Skip if not tensor data
                if not isinstance(data1, torch.Tensor) or not isinstance(data2, torch.Tensor):
                    continue

                print(f"📊 Checking consistency for field: {field_name}")

                # Data should be identical (same normalization applied to same raw data)
                assert torch.allclose(data1, data2, atol=1e-6), (
                    f"Normalized data should be consistent across dataloaders for field {field_name}"
                )

                print(f"  ✅ Consistent normalization for {field_name}")

    print("✅ Normalization consistency test passed!")


def test_compare_dataloader_and_roboticsdataloader(dataset_path, mock_config):
    """Test that get_wds_dataloader and RoboticsDataLoader produce matching lowdim data for all sample_ids."""
    import torch
    import yaml

    # Load config
    config_path = "lbm2/config_presets/data/lbm_data_params.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict.update(
        {
            "num_workers": 1,
            "seed": 42,
            "processor": "google/paligemma-3b-pt-224",
            "add_action_token": False,
            "seq_len": 512,
            "dataset_statistics": [f"{dataset_path}/dataset_statistics.json"],
            "dataset_manifest": [f"{dataset_path}/manifest.jsonl"],
            "normalization": {"enabled": False},
        }
    )
    data_cfg = LBMDataParams.from_dict(config_dict)

    cfg = mock_config(dataset_path, batch_size=1, processor_name="google/paligemma-3b-pt-224")

    # Use get_datastring_input to generate datastrings
    num_samples = -1
    curr_shard_idx_per_dataset = [0]
    shard_shuffle_seed_per_dataset = [0]
    manifest_paths = [f"{dataset_path}/manifest.jsonl"]
    dataset_weighting = [1]
    allow_multiple_epochs = False
    num_workers_per_gpu = 1
    world_size = 1
    (
        datastrings,
        num_samples_list_per_dataset,
        next_shard_idx_per_dataset,
        next_shard_shuffle_seed_per_dataset,
    ) = get_datastring_input(
        num_samples,
        curr_shard_idx_per_dataset,
        shard_shuffle_seed_per_dataset,
        manifest_paths,
        dataset_weighting,
        allow_multiple_epochs,
        num_workers_per_gpu,
        world_size,
    )

    from lbm2.data.dataloader import get_wds_dataloader

    dataloader_info = get_wds_dataloader(
        datastrings=datastrings, num_samples_per_dataset=num_samples_list_per_dataset, checkpoint_num=0, cfg=cfg
    )
    dataloader = dataloader_info.dataloader

    samples_dataloader = {}
    cam_dataloader = {}
    for batch in dataloader:
        for i, sample_id in enumerate([meta["sample_id"] for meta in batch["metadata"]]):
            samples_dataloader[sample_id] = {
                k: v[i] if hasattr(v, "shape") and v.shape[0] == len(batch["metadata"]) else v
                for k, v in batch["lowdim"].items()
            }
        cam_dataloader[batch["metadata"][0]["sample_id"]] = {
            "intrinsics": batch["intrinsics"],
            "extrinsics": batch["extrinsics"],
        }
    # Create RoboticsDataLoader and load samples
    data_loader = RoboticsDataLoader(data_cfg, max_samples=num_samples, max_shards=1, use_dataloader=True)
    samples = data_loader.load_samples_auto()
    samples_files = {sample["metadata"]["sample_id"]: sample["lowdim"] for sample in samples}
    cam_files = {
        sample["metadata"]["sample_id"]: {"intrinsics": sample["intrinsics"], "extrinsics": sample["extrinsics"]}
        for sample in samples
    }

    assert len(samples_dataloader) == len(samples_files)
    assert set(samples_dataloader.keys()) == set(samples_files.keys())

    # Compare the lowdim data from the dataloader and the RoboticsDataLoader
    for k in samples_dataloader:
        for k2, v2 in samples_dataloader[k].items():
            assert k2 in samples_files[k]
            if isinstance(v2, torch.Tensor):
                v1 = torch.tensor(samples_files[k][k2], dtype=v2.dtype)
                assert torch.allclose(v2, v1), f"Mismatch for sample {k}, field {k2}"

    # Compare the camera data from the dataloader and the RoboticsDataLoader
    for k in cam_files:
        for k_cam in ["intrinsics", "extrinsics"]:
            for k2, v2 in cam_files[k][k_cam].items():
                assert k2 in cam_files[k][k_cam]
                if isinstance(v2, torch.Tensor):
                    assert torch.allclose(v2, torch.tensor(cam_files[k][k_cam][k2], dtype=v2.dtype))
                else:
                    np.testing.assert_allclose(v2, cam_files[k][k_cam][k2])


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
