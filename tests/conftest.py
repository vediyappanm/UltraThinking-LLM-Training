"""Pytest configuration and fixtures"""
import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_model_config():
    """Small model config for testing"""
    from src.models.architecture import ModelConfig
    return ModelConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        intermediate_size=512,
        flash_attention=False,
        gradient_checkpointing=False
    )


@pytest.fixture
def tiny_ultrathink_config():
    """Tiny ULTRATHINK config for testing"""
    from src.models.ultrathink import UltraThinkConfig
    from src.models.architecture import ModelConfig
    
    model_config = ModelConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        intermediate_size=512,
        flash_attention=False,
        gradient_checkpointing=False
    )
    
    return UltraThinkConfig(
        model_config=model_config,
        enable_dre=False,
        enable_constitutional=False,
        enable_moe=False,
        enable_multimodal=False,
        enable_rlhf=False
    )


@pytest.fixture
def sample_batch(device):
    """Generate a sample batch for testing"""
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    
    return {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, device=device, dtype=torch.long),
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return tmp_path


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip GPU tests if no GPU available
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    if not torch.cuda.is_available():
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
