"""Unit tests for dataset loading"""
import pytest
import torch
from src.data.datasets import DatasetConfig, DATASET_CONFIGS


class TestDatasetConfig:
    """Test dataset configuration"""
    
    def test_default_config(self):
        """Test default dataset configuration"""
        config = DatasetConfig()
        assert config.name == "wikitext"
        assert config.max_length == 512
        assert config.tokenizer_name == "gpt2"
    
    def test_custom_config(self):
        """Test custom dataset configuration"""
        config = DatasetConfig(
            name="custom",
            max_length=1024,
            tokenizer_name="facebook/opt-125m",
            streaming=True
        )
        assert config.name == "custom"
        assert config.max_length == 1024
        assert config.streaming is True
    
    def test_validation(self):
        """Test configuration validation"""
        config = DatasetConfig(max_length=512)
        assert config.max_length == 512
        assert config.min_length == 10
        assert config.max_length > config.min_length


class TestPredefinedConfigs:
    """Test predefined dataset configurations"""
    
    def test_wikitext_config(self):
        """Test WikiText configuration"""
        config = DATASET_CONFIGS["wikitext"]
        assert config.name == "wikitext"
        assert config.subset == "wikitext-2-raw-v1"
        assert config.text_column == "text"
    
    def test_c4_config(self):
        """Test C4 configuration"""
        config = DATASET_CONFIGS["c4"]
        assert config.name == "c4"
        assert config.subset == "en"
        assert config.streaming is True
    
    def test_all_configs_valid(self):
        """Test that all predefined configs are valid"""
        for name, config in DATASET_CONFIGS.items():
            assert isinstance(config, DatasetConfig)
            assert config.name is not None
            assert config.text_column is not None
            assert config.max_length > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
