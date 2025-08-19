import torch
from PIL import Image

from lbm2.data.processor import get_processor
from lbm2.params.train_experiment_params import load_experiment_params_from_yaml


class TestProcessorPaliGemma:
    def test_processor_paligemma(self):
        params = load_experiment_params_from_yaml("tests/params/dummy_configs/dummy_vlm_config.yaml")
        object.__setattr__(params.data, "processor", "google/paligemma-3b-pt-224")
        paligemma_processor = get_processor(params.data)

        # Create a dummy image
        image = Image.open("tests/shared/chonky_cat.png")
        if image.mode == "RGBA":
            image = image.convert("RGB")

        sample = {"image": image, "text": "<image> What is in this image?"}

        num_img_tokens = paligemma_processor.image_seq_length
        result = paligemma_processor(
            images=sample["image"],
            text=sample["text"],
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            padding_value=0,
            max_length=params.data.seq_len + num_img_tokens,
        )

        # Test that all expected keys are present
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "pixel_values" in result

        # Test input_ids shape and values
        assert result["input_ids"].dim() == 2, "input_ids should be 2D tensor"
        assert result["input_ids"].shape[0] == 1, "Batch size should be 1"
        assert result["input_ids"].shape[1] == params.data.seq_len + num_img_tokens, (
            "Sequence length should be 256 for next token prediction"
        )
        assert result["input_ids"].dtype == torch.long, "input_ids should be long tensor"
        # Check that input_ids contains valid token IDs (non-negative integers)
        assert (result["input_ids"] >= 0).all(), "All input_ids should be non-negative"
        # Check for padding tokens (should be 0 at the end due to padding)
        assert (result["input_ids"][0, -10:] == 0).any(), "Should have padding tokens at the end"

        # Test attention_mask shape and values
        assert result["attention_mask"].dim() == 2, "attention_mask should be 2D tensor"
        assert result["attention_mask"].shape == result["input_ids"].shape, (
            "attention_mask should match input_ids shape"
        )
        assert result["attention_mask"].dtype == torch.long, "attention_mask should be long tensor"
        # Attention mask should only contain 0s and 1s
        assert torch.all((result["attention_mask"] == 0) | (result["attention_mask"] == 1)), (
            "attention_mask should only contain 0s and 1s"
        )
        # Should have 1s for actual tokens and 0s for padding
        assert result["attention_mask"][0, 0] == 1, "First token should have attention"
        assert (result["attention_mask"][0, -10:] == 0).any(), "Should have 0s for padding tokens"

        # Test pixel_values shape and values
        assert result["pixel_values"].dim() == 4, "pixel_values should be 4D tensor (batch, channels, height, width)"
        assert result["pixel_values"].shape[0] == 1, "Batch size should be 1"
        assert result["pixel_values"].shape[1] == 3, "Should have 3 color channels (RGB)"
        assert result["pixel_values"].shape[2] == 224, "Height should be 224 (standard for PaliGemma)"
        assert result["pixel_values"].shape[3] == 224, "Width should be 224 (standard for PaliGemma)"
        assert result["pixel_values"].dtype == torch.float32, "pixel_values should be float32"
        # Check that pixel values are normalized (typically in range [-1, 1] or [0, 1])
        assert result["pixel_values"].min() >= -2.0, "Pixel values should be reasonably normalized (min check)"
        assert result["pixel_values"].max() <= 2.0, "Pixel values should be reasonably normalized (max check)"
        # Check that pixel values are not all the same (should have variation)
        assert result["pixel_values"].std() > 0.1, "Pixel values should have reasonable variation"


# Implement this after the Stable Diffusion is merged in
# class TestProcessorStableDiffusion:
