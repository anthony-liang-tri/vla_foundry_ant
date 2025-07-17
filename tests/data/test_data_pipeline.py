import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from lbm2.data.pipelines.text import TextPipeline, filter_lt_seqlen
from lbm2.data.pipelines.text_untokenized import TextUntokenizedPipeline, batch_tokenize
from lbm2.data.pipelines.image_caption import ImageCaptionPipeline, filter_no_caption_or_no_image
from lbm2.data.pipelines import create_wds_pipeline, FiniteDataPipeline
from lbm2.data.dataloader import get_wds_dataloader, get_datastring_input
from lbm2.params.train_experiment_params import load_params_from_yaml


class TestTextPipeline:
    """Test TextPipeline for pre-tokenized text data."""
    
    def create_mock_data_configs(self):
        """Create mock data configurations."""
        mock_config = Mock()
        mock_config.seq_len = 128
        mock_config.seed = 42
        return mock_config
    
    def test_filter_lt_seqlen_valid(self):
        """Test filter_lt_seqlen with valid sequence length."""
        seq_len = 10
        valid_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Length 11 > 10
        assert filter_lt_seqlen(seq_len, valid_sequence) == True
    
    def test_filter_lt_seqlen_invalid(self):
        """Test filter_lt_seqlen with invalid sequence length."""
        seq_len = 10
        invalid_sequence = [1, 2, 3, 4, 5]  # Length 5 <= 10
        with patch('logging.warning') as mock_warning:
            assert filter_lt_seqlen(seq_len, invalid_sequence) == False
            mock_warning.assert_called_once()
    
    def test_text_pipeline_creation(self):
        """Test TextPipeline initialization and pipeline creation."""
        data_configs = self.create_mock_data_configs()
        batch_size = 4
        
        pipeline = TextPipeline("text", data_configs, batch_size)
        
        assert pipeline.modality == "text"
        assert pipeline.data_configs == data_configs
        assert pipeline.batch_size == batch_size
            
    @pytest.mark.parametrize("seq_len,batch_size", [
        (64, 2),
        (128, 4),
        (256, 8),
    ])
    def test_text_pipeline_different_configs(self, seq_len, batch_size):
        """Test TextPipeline with different configurations."""
        data_configs = Mock()
        data_configs.seq_len = seq_len
        data_configs.seed = 42
        
        pipeline = TextPipeline("text", data_configs, batch_size)
        pipeline_steps = pipeline.create_pipeline("dummy_datastring", 0)
        
        assert pipeline.batch_size == batch_size

    @pytest.mark.parametrize("param_config_path", ["tests/shared/dummy_text_config.yaml"])
    def test_text_dataloader_actual_datastring_with_mixing(self, param_config_path):
        params = load_params_from_yaml(param_config_path)
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples = 50_000,
            curr_shard_idx_per_dataset = [0,0],
            shard_shuffle_seed_per_dataset = [123,123],
            manifest_paths = params.data.dataset_manifest,
            dataset_weighting = params.data.dataset_weighting,
            allow_multiple_epochs = True,
            num_workers_per_gpu = 1,
            world_size = 8,
        )
                
        # Assert datastring outputs
        assert isinstance(datastrings, list)
        assert len(datastrings) == len(params.data.dataset_manifest)
        assert all(isinstance(ds, str) for ds in datastrings)

        # Assert num_samples_per_dataset
        assert isinstance(num_samples_per_dataset, list)
        assert len(num_samples_per_dataset) == len(params.data.dataset_manifest)
        assert all(isinstance(n, int) and n > 0 for n in num_samples_per_dataset)

        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)

        # Assert dataloader properties
        assert dataloader is not None
        assert hasattr(dataloader, 'dataloader')
        assert dataloader.dataloader is not None

        # Test iteration and batch structure
        batch_count = 0
        for batch in dataloader.dataloader:
            # Assert batch structure
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert isinstance(batch["input_ids"], torch.Tensor)
            
            # Assert batch dimensions
            assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]
            assert batch["input_ids"].shape[1] == params.data.seq_len + 1  # +1 for next token prediction
            
            # Assert data types
            assert batch["input_ids"].dtype == torch.long
            
            # Assert no NaN or infinite values
            assert not torch.isnan(batch["input_ids"]).any()
            assert not torch.isinf(batch["input_ids"]).any()
            
            batch_count += 1
            if batch_count >= 3:  # Only test first few batches to avoid long test times
                break

        # Assert we got at least some batches
        assert batch_count > 0

    def test_text_batch_consistency(self):
        """Test that text batches maintain consistency across dataloader instances."""
        params = load_params_from_yaml('tests/shared/dummy_text_config.yaml')
        
        # Create two identical dataloaders with same seed
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples=50_000,
            curr_shard_idx_per_dataset=[0, 0],
            shard_shuffle_seed_per_dataset=[42, 42],  # Fixed seed for consistency
            manifest_paths=params.data.dataset_manifest,
            dataset_weighting=params.data.dataset_weighting,
            allow_multiple_epochs=True,
            num_workers_per_gpu=1,
            world_size=8,
        )
                        
        dataloader1 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        dataloader2 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        
        # Get first batch from each dataloader
        batch1 = next(iter(dataloader1.dataloader))
        batch2 = next(iter(dataloader2.dataloader))
        
        # Assert consistent batch structure
        assert isinstance(batch1, dict)
        assert isinstance(batch2, dict)
        assert "input_ids" in batch1 and "input_ids" in batch2
        
        # Assert tensor properties are consistent
        assert batch1["input_ids"].dtype == batch2["input_ids"].dtype
        assert batch1["input_ids"].shape == batch2["input_ids"].shape
        
        # Assert both batches have valid content
        assert batch1["input_ids"].shape[0] > 0  # Non-empty batch
        assert batch2["input_ids"].shape[0] > 0
        assert batch1["input_ids"].shape[1] == params.data.seq_len + 1
        assert batch2["input_ids"].shape[1] == params.data.seq_len + 1
        
        # Assert no NaN or infinite values in either batch
        assert not torch.isnan(batch1["input_ids"]).any()
        assert not torch.isinf(batch1["input_ids"]).any()
        assert not torch.isnan(batch2["input_ids"]).any()
        assert not torch.isinf(batch2["input_ids"]).any()
        
        # Assert valid token ranges
        assert (batch1["input_ids"] >= 0).all()
        assert (batch2["input_ids"] >= 0).all()
        

class TestTextUntokenizedPipeline:
    """Test TextUntokenizedPipeline for raw text data."""
    
    def create_mock_data_configs(self):
        """Create mock data configurations."""
        mock_config = Mock()
        mock_config.seq_len = 128
        mock_config.seed = 42
        mock_config.tokenizer = "gpt2"
        return mock_config
    
    @patch('lbm2.data.pipelines.text_untokenized.get_tokenizer')
    def test_text_untokenized_pipeline_creation(self, mock_tokenizer):
        """Test TextUntokenizedPipeline initialization."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.add_special_tokens = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        data_configs = self.create_mock_data_configs()
        batch_size = 4
        
        pipeline = TextUntokenizedPipeline("text_untokenized", data_configs, batch_size)
        
        assert pipeline.modality == "text_untokenized"
        assert pipeline.data_configs == data_configs
        assert pipeline.batch_size == batch_size
        assert pipeline.tokenizer == mock_tokenizer_instance
        
        # Verify pad token was added
        mock_tokenizer_instance.add_special_tokens.assert_called_once_with({'pad_token': '[PAD]'})
    
    def test_batch_tokenize(self):
        """Test batch_tokenize function."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        }
        
        # Test data
        batch = [["Hello world", "Test text"]]
        seq_len = 3
        
        input_ids, attention_mask = batch_tokenize(batch, mock_tokenizer, seq_len)
        
        # Verify tokenizer was called with correct parameters
        mock_tokenizer.assert_called_once_with(
            ["Hello world", "Test text"],
            padding='max_length',
            truncation=True,
            max_length=seq_len + 1,
            return_tensors='pt'
        )
        
        # Verify outputs
        assert torch.equal(input_ids, torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
        assert torch.equal(attention_mask, torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
    
    def test_batch_tokenize_with_bytes(self):
        """Test batch_tokenize with bytes input."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Test with bytes input
        batch = [[b"Hello world"]]
        seq_len = 10
        
        input_ids, attention_mask = batch_tokenize(batch, mock_tokenizer, seq_len)
        
        # Verify bytes were decoded
        mock_tokenizer.assert_called_once_with(
            ["Hello world"],  # Should be decoded from bytes
            padding='max_length',
            truncation=True,
            max_length=seq_len + 1,
            return_tensors='pt'
        )
    
    @patch('lbm2.data.pipelines.text_untokenized.get_tokenizer')
    def test_tokenize_wrapper(self, mock_tokenizer):
        """Test tokenize_wrapper method."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        data_configs = self.create_mock_data_configs()
        pipeline = TextUntokenizedPipeline("text_untokenized", data_configs, 4)
        
        # Mock batch_tokenize
        with patch('lbm2.data.pipelines.text_untokenized.batch_tokenize') as mock_batch_tokenize:
            mock_input_ids = torch.tensor([[1, 2, 3]])
            mock_attention_mask = torch.tensor([[1, 1, 1]])
            mock_batch_tokenize.return_value = (mock_input_ids, mock_attention_mask)
            
            batch = [["test text"]]
            result = pipeline.tokenize_wrapper(batch)
            
            # Verify batch_tokenize was called
            mock_batch_tokenize.assert_called_once_with(batch, mock_tokenizer_instance, data_configs.seq_len)
            
            # Verify output format
            assert "input_ids" in result
            assert "attention_mask" in result
            assert torch.equal(result["input_ids"], mock_input_ids)
            assert torch.equal(result["attention_mask"], mock_attention_mask)

    @pytest.mark.parametrize("param_config_path", ["tests/shared/dummy_text_untokenized_config.yaml"])
    def test_text_untokenized_dataloader_actual_datastring(self, param_config_path):
        params = load_params_from_yaml(param_config_path)
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples = 100_000,
            curr_shard_idx_per_dataset = [0],
            shard_shuffle_seed_per_dataset = [123],
            manifest_paths = params.data.dataset_manifest,
            dataset_weighting = params.data.dataset_weighting,
            allow_multiple_epochs = True,
            num_workers_per_gpu = 1,
            world_size = 8,
        )

        # Assert datastring structure
        assert isinstance(datastrings, list)
        assert len(datastrings) == 1  # Single dataset
        assert isinstance(datastrings[0], str)
        
        # Assert shard count - should have many shards for 100k samples across 8 workers
        shard_pattern = datastrings[0].split("{")[1].split("}")[0]
        shard_list = shard_pattern.split(",")
        assert len(shard_list) >= 8  # At least one shard per worker
        assert len(shard_list) % 8 == 0  # Divisible by world_size for even distribution
        
        # Assert sample count
        assert isinstance(num_samples_per_dataset, list)
        assert len(num_samples_per_dataset) == 1

        # Test dataloader creation
        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        
        # Assert dataloader properties
        assert dataloader is not None
        assert hasattr(dataloader, 'dataloader')
        assert dataloader.dataloader is not None
        
        # Test batch iteration
        batch_count = 0
        for batch in dataloader.dataloader:
            # Assert batch structure for text_untokenized
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            
            # Assert tensor properties
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["attention_mask"], torch.Tensor)
            
            # Assert data types
            assert batch["input_ids"].dtype == torch.long
            assert batch["attention_mask"].dtype == torch.long
            
            # Assert tensor shapes match
            assert batch["input_ids"].shape == batch["attention_mask"].shape
            assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]
            
            # Assert sequence length (+1 for next token prediction)
            assert batch["input_ids"].shape[1] == params.data.seq_len + 1
            
            # Assert batch size is reasonable
            batch_size = batch["input_ids"].shape[0]
            assert batch_size > 0
            
            # Assert no NaN or infinite values
            assert not torch.isnan(batch["input_ids"]).any()
            assert not torch.isinf(batch["input_ids"]).any()
            assert not torch.isnan(batch["attention_mask"]).any()
            assert not torch.isinf(batch["attention_mask"]).any()
            
            # Assert token values are valid
            assert (batch["input_ids"] >= 0).all()  # No negative token IDs
            assert (batch["attention_mask"] >= 0).all()  # No negative attention values
            assert (batch["attention_mask"] <= 1).all()  # Attention mask should be 0 or 1
            
            # Assert attention mask structure (should have 1s for real tokens, 0s for padding)
            for i in range(batch_size):
                attention_row = batch["attention_mask"][i]
                # Find first zero (start of padding)
                zeros = (attention_row == 0).nonzero(as_tuple=True)[0]
                if len(zeros) > 0:
                    first_zero = zeros[0].item()
                    # All tokens before first zero should be 1
                    assert (attention_row[:first_zero] == 1).all()
                    # All tokens after first zero should be 0
                    assert (attention_row[first_zero:] == 0).all()
            
            batch_count += 1
            if batch_count >= 2:  # Test first couple batches
                break
        
        # Assert we successfully got batches
        assert batch_count > 0, "No batches were produced by the dataloader"

    def test_text_untokenized_batch_consistency(self):
        """Test that text_untokenized batches maintain consistency across dataloader instances."""
        params = load_params_from_yaml('tests/shared/dummy_text_untokenized_config.yaml')
        
        # Create two identical dataloaders with same seed
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples=50_000,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[42],  # Fixed seed for consistency
            manifest_paths=params.data.dataset_manifest,
            dataset_weighting=params.data.dataset_weighting,
            allow_multiple_epochs=True,
            num_workers_per_gpu=1,
            world_size=8,
        )
        
        dataloader1 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        dataloader2 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        
        # Get first batch from each dataloader
        batch1 = next(iter(dataloader1.dataloader))
        batch2 = next(iter(dataloader2.dataloader))
        
        # Assert consistent batch structure for text_untokenized
        assert isinstance(batch1, dict)
        assert isinstance(batch2, dict)
        assert "input_ids" in batch1 and "input_ids" in batch2
        assert "attention_mask" in batch1 and "attention_mask" in batch2
        
        # Assert tensor properties are consistent
        assert batch1["input_ids"].dtype == batch2["input_ids"].dtype
        assert batch1["attention_mask"].dtype == batch2["attention_mask"].dtype
        assert batch1["input_ids"].shape == batch2["input_ids"].shape
        assert batch1["attention_mask"].shape == batch2["attention_mask"].shape
        
        # Assert both batches have valid content
        assert batch1["input_ids"].shape[0] > 0  # Non-empty batch
        assert batch2["input_ids"].shape[0] > 0
        assert batch1["input_ids"].shape[1] == params.data.seq_len + 1
        assert batch2["input_ids"].shape[1] == params.data.seq_len + 1
        
        # Assert tensor shapes match between input_ids and attention_mask
        assert batch1["input_ids"].shape == batch1["attention_mask"].shape
        assert batch2["input_ids"].shape == batch2["attention_mask"].shape
        
        # Assert no NaN or infinite values in either batch
        assert not torch.isnan(batch1["input_ids"]).any()
        assert not torch.isinf(batch1["input_ids"]).any()
        assert not torch.isnan(batch1["attention_mask"]).any()
        assert not torch.isinf(batch1["attention_mask"]).any()
        assert not torch.isnan(batch2["input_ids"]).any()
        assert not torch.isinf(batch2["input_ids"]).any()
        assert not torch.isnan(batch2["attention_mask"]).any()
        assert not torch.isinf(batch2["attention_mask"]).any()
        
        # Assert valid token ranges
        assert (batch1["input_ids"] >= 0).all()
        assert (batch2["input_ids"] >= 0).all()
        assert (batch1["attention_mask"] >= 0).all() and (batch1["attention_mask"] <= 1).all()
        assert (batch2["attention_mask"] >= 0).all() and (batch2["attention_mask"] <= 1).all()
        
        # Assert attention mask structure consistency
        for batch in [batch1, batch2]:
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                attention_row = batch["attention_mask"][i]
                # Should have some attention (not all zeros)
                assert attention_row.sum() > 0, f"Empty attention mask for sample {i}"
                
                # Find padding pattern (1s followed by 0s)
                zeros = (attention_row == 0).nonzero(as_tuple=True)[0]
                if len(zeros) > 0:
                    first_zero = zeros[0].item()
                    # All tokens before first zero should be 1
                    assert (attention_row[:first_zero] == 1).all()
                    # All tokens after first zero should be 0
                    assert (attention_row[first_zero:] == 0).all()


class TestImageCaptionPipeline:
    """Test ImageCaptionPipeline for image-caption data."""
    
    def create_mock_configs(self):
        """Create mock configurations."""
        data_config = Mock()
        data_config.seq_len = 128
        data_config.seed = 42
        data_config.processor = "google/paligemma-3b-pt-224"
        return data_config
    
    def test_filter_no_caption_or_no_image(self):
        """Test filter_no_caption_or_no_image function."""
        # Valid sample with both image and caption
        valid_sample = {"txt": "A cat", "jpg": b"image_data"}
        assert filter_no_caption_or_no_image(valid_sample) == True
        
        # Sample with only image, no caption
        no_caption_sample = {"jpg": b"image_data"}
        assert filter_no_caption_or_no_image(no_caption_sample) == False
        
        # Sample with only caption, no image
        no_image_sample = {"txt": "A cat"}
        assert filter_no_caption_or_no_image(no_image_sample) == False
        
        # Sample with different image formats
        png_sample = {"txt": "A cat", "png": b"image_data"}
        assert filter_no_caption_or_no_image(png_sample) == True
        
        jpeg_sample = {"txt": "A cat", "jpeg": b"image_data"}
        assert filter_no_caption_or_no_image(jpeg_sample) == True
        
        webp_sample = {"txt": "A cat", "webp": b"image_data"}
        assert filter_no_caption_or_no_image(webp_sample) == True
    
    @patch('lbm2.data.pipelines.image_caption.get_processor')
    def test_image_caption_pipeline_creation(self, mock_get_processor):
        """Test ImageCaptionPipeline initialization."""
        mock_processor = Mock()
        mock_get_processor.return_value = mock_processor
        
        data_config = self.create_mock_configs()
        batch_size = 4
        
        pipeline = ImageCaptionPipeline("image_caption", data_config, batch_size)
        
        assert pipeline.modality == "image_caption"
        assert pipeline.data_configs == data_config
        assert pipeline.batch_size == batch_size
        assert pipeline.processor == mock_processor
        
        # Verify processor was initialized correctly
        mock_get_processor.assert_called_once_with(data_config)
        
    @pytest.mark.parametrize("image_format", ["jpg", "png", "jpeg", "webp"])
    def test_image_caption_filter_different_formats(self, image_format):
        """Test filter function with different image formats."""
        sample = {"txt": "A description", image_format: b"image_data"}
        assert filter_no_caption_or_no_image(sample) == True

    @pytest.mark.parametrize("param_config_path", ["tests/shared/dummy_vlm_config.yaml"])
    def test_vlm_dataloader_actual_datastring(self, param_config_path):
        params = load_params_from_yaml(param_config_path)
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples = 10_000,
            curr_shard_idx_per_dataset = [0],
            shard_shuffle_seed_per_dataset = [123],
            manifest_paths = params.data.dataset_manifest,
            dataset_weighting = params.data.dataset_weighting,
            allow_multiple_epochs = True,
            num_workers_per_gpu = 1,
            world_size = 8,
        )

        # Assert datastring structure for VLM data
        assert isinstance(datastrings, list)
        assert len(datastrings) == 1  # Single dataset
        assert isinstance(datastrings[0], str)
        
        # Assert shard structure for image-caption data
        shard_pattern = datastrings[0].split("{")[1].split("}")[0]
        shard_list = shard_pattern.split(",")
        assert len(shard_list) >= 8  # At least one shard per worker
        assert len(shard_list) % 8 == 0  # Divisible by world_size

        # Assert sample count
        assert isinstance(num_samples_per_dataset, list)
        assert len(num_samples_per_dataset) == 1
        
        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)

        # Assert dataloader properties
        assert dataloader is not None
        assert hasattr(dataloader, 'dataloader')
        assert dataloader.dataloader is not None
        
        # Test VLM batch structure
        batch_count = 0
        for batch in dataloader.dataloader:
            # Assert VLM-specific batch structure
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "pixel_values" in batch
            
            # Assert tensor types
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["attention_mask"], torch.Tensor)
            assert isinstance(batch["pixel_values"], torch.Tensor)
            
            # Assert data types
            assert batch["input_ids"].dtype == torch.long
            assert batch["attention_mask"].dtype == torch.long
            assert batch["pixel_values"].dtype == torch.float32
            
            # Assert tensor dimensions
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            
            # Text tensors should be 2D: [batch_size, seq_len]
            assert batch["input_ids"].dim() == 2
            assert batch["attention_mask"].dim() == 2
            assert batch["input_ids"].shape == batch["attention_mask"].shape
            
            # Image tensor should be 4D: [batch_size, channels, height, width]
            assert batch["pixel_values"].dim() == 4
            pixel_shape = batch["pixel_values"].shape
            assert pixel_shape[0] == batch_size  # Same batch size
            assert pixel_shape[1] == 3  # RGB channels
            
            # Assert sequence length matches config (+1 for next token prediction)
            assert seq_len == params.data.seq_len + 1
            
            # Assert reasonable batch and image sizes
            assert batch_size > 0
            assert pixel_shape[2] > 0 and pixel_shape[3] > 0  # Valid image dimensions
            
            # Assert no NaN or infinite values
            assert not torch.isnan(batch["input_ids"]).any()
            assert not torch.isinf(batch["input_ids"]).any()
            assert not torch.isnan(batch["attention_mask"]).any()
            assert not torch.isinf(batch["attention_mask"]).any()
            assert not torch.isnan(batch["pixel_values"]).any()
            assert not torch.isinf(batch["pixel_values"]).any()
            
            # Assert token values are valid
            assert (batch["input_ids"] >= 0).all()  # No negative token IDs
            assert (batch["attention_mask"] >= 0).all()
            assert (batch["attention_mask"] <= 1).all()  # Binary attention mask
                        
            # Assert attention mask structure
            for i in range(batch_size):
                attention_row = batch["attention_mask"][i]
                # Should have some attention (not all zeros)
                assert attention_row.sum() > 0, f"Empty attention mask for sample {i}"
                
                # Find padding pattern (1s followed by 0s)
                zeros = (attention_row == 0).nonzero(as_tuple=True)[0]
                if len(zeros) > 0:
                    first_zero = zeros[0].item()
                    # All tokens before first zero should be 1
                    assert (attention_row[:first_zero] == 1).all()
                    # All tokens after first zero should be 0
                    assert (attention_row[first_zero:] == 0).all()
            
            # Test image diversity (not all images should be identical)
            if batch_size > 1:
                # Check that not all images are identical
                first_image = batch["pixel_values"][0]
                identical_count = 0
                for i in range(1, batch_size):
                    if torch.allclose(first_image, batch["pixel_values"][i], atol=1e-6):
                        identical_count += 1
                
                # Allow some identical images but not all
                assert identical_count < batch_size - 1, "All images appear to be identical"
            
            batch_count += 1
            if batch_count >= 2:  # Test first couple batches
                break
        
        # Assert we successfully got batches
        assert batch_count > 0, "No batches were produced by the VLM dataloader"

    def test_vlm_batch_consistency(self):
        params = load_params_from_yaml('tests/shared/dummy_vlm_config.yaml')
        
        # Create two identical dataloaders
        datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples=10_000,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[42],  # Fixed seed
            manifest_paths=params.data.dataset_manifest,
            dataset_weighting=params.data.dataset_weighting,
            allow_multiple_epochs=True,
            num_workers_per_gpu=1,
            world_size=8,
        )
                    
        dataloader1 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        dataloader2 = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num=0, cfg=params)
        
        # Get first batch from each
        batch1 = next(iter(dataloader1.dataloader))
        batch2 = next(iter(dataloader2.dataloader))
        
        # Assert consistent shapes (content may differ due to randomness)
        assert batch1["input_ids"].shape == batch2["input_ids"].shape
        assert batch1["attention_mask"].shape == batch2["attention_mask"].shape
        assert batch1["pixel_values"].shape == batch2["pixel_values"].shape


class TestPipelineCreation:
    """Test pipeline creation factory function."""
    
    def create_mock_cfg(self, modality):
        """Create mock configuration for different modalities."""
        cfg = Mock()
        cfg.data = Mock()
        cfg.data.seq_len = 128
        cfg.data.seed = 42
        
        if modality == "text_untokenized":
            cfg.data.tokenizer = "gpt2"
        elif modality == "image_caption":
            cfg.data.processor = "google/paligemma-3b-pt-224"
            cfg.vit = Mock()
            cfg.vit.image_size = 224
        
        return cfg
    
    @pytest.mark.parametrize("modality", ["text", "text_untokenized", "image_caption"])
    def test_create_wds_pipeline(self, modality):
        """Test create_wds_pipeline factory function."""
        cfg = self.create_mock_cfg(modality)
        datastring = "dummy_datastring"
        batch_size = 4
        checkpoint_num = 0
        
        with patch('lbm2.data.pipelines.text_untokenized.get_tokenizer') if modality == "text_untokenized" else \
             patch('lbm2.data.pipelines.image_caption.get_processor') if modality == "image_caption" else \
             patch('builtins.print'):  # Dummy patch for text modality
            
            pipeline = create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, cfg)
            
            assert isinstance(pipeline, FiniteDataPipeline)
    
    def test_create_wds_pipeline_unsupported_modality(self):
        """Test create_wds_pipeline with unsupported modality."""
        cfg = self.create_mock_cfg("text")
        
        with pytest.raises(ValueError, match="unsupported_modality webdataset pipeline not supported"):
            create_wds_pipeline("dummy", "unsupported_modality", 4, 0, cfg)


class TestFiniteDataPipeline:
    """Test FiniteDataPipeline wrapper."""
    
    def test_finite_data_pipeline_iter_with_limit(self):
        """Test FiniteDataPipeline with sample limit."""
        # Create mock data
        mock_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Create pipeline with limit
        pipeline = FiniteDataPipeline()
        pipeline.nsamples = 5
        pipeline.iterator = lambda: iter(mock_data)
        
        # Test iteration
        result = list(pipeline.__iter__())
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5
    
    def test_finite_data_pipeline_iter_no_limit(self):
        """Test FiniteDataPipeline without sample limit."""
        # Create mock data
        mock_data = [1, 2, 3, 4, 5]
        
        # Create pipeline without limit
        pipeline = FiniteDataPipeline()
        pipeline.nsamples = 0
        pipeline.iterator = lambda: iter(mock_data)
        
        # Test iteration
        result = list(pipeline.__iter__())
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5


class TestIntegrationWithRealConfig:
    """Integration tests using real configuration files."""
    def test_image_caption_pipeline_with_real_config(self):
        """Test ImageCaptionPipeline with real VLM config."""
        try:
            params = load_params_from_yaml('tests/shared/dummy_vlm_config.yaml')
            
            with patch('lbm2.data.pipelines.image_caption.get_processor') as mock_get_processor:
                mock_processor = Mock()
                mock_get_processor.return_value = mock_processor
                
                pipeline = ImageCaptionPipeline(
                    "image_caption", 
                    params.data, 
                    4, 
                )
                
                assert pipeline.modality == "image_caption"
                assert pipeline.batch_size == 4
                
        except FileNotFoundError:
            pytest.skip("Real config file not available")