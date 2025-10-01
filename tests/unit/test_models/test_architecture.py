"""Unit tests for core architecture components"""
import pytest
import torch
import torch.nn as nn
from src.models.architecture import (
    ModelConfig,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    GroupedQueryAttention,
    apply_rotary_pos_emb,
    rotate_half
)


class TestRMSNorm:
    """Test RMSNorm layer"""
    
    def test_initialization(self):
        """Test RMSNorm initialization"""
        norm = RMSNorm(hidden_size=768)
        assert norm.weight.shape == (768,)
        assert torch.allclose(norm.weight, torch.ones(768))
    
    def test_forward_pass(self):
        """Test RMSNorm forward pass"""
        norm = RMSNorm(hidden_size=768)
        x = torch.randn(2, 16, 768)
        output = norm(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_normalization(self):
        """Test that RMSNorm actually normalizes"""
        norm = RMSNorm(hidden_size=768, eps=1e-6)
        x = torch.randn(2, 16, 768) * 100  # Large scale
        output = norm(x)
        
        # Check variance is close to 1
        variance = output.pow(2).mean(-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=1e-5)


class TestRotaryPositionalEmbedding:
    """Test Rotary Position Embeddings"""
    
    def test_initialization(self):
        """Test RoPE initialization"""
        rope = RotaryPositionalEmbedding(dim=64, max_position_embeddings=2048)
        assert rope.dim == 64
        assert rope.max_position_embeddings == 2048
        assert rope.inv_freq.shape == (32,)  # dim // 2
    
    def test_forward_pass(self):
        """Test RoPE forward pass"""
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(2, 8, 16, 64)  # batch, heads, seq, dim
        
        cos, sin = rope(x, seq_len=16)
        
        assert cos.shape == (1, 1, 16, 64)
        assert sin.shape == (1, 1, 16, 64)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()
    
    def test_caching(self):
        """Test that RoPE caches cos/sin values"""
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(2, 8, 16, 64)
        
        cos1, sin1 = rope(x, seq_len=16)
        cos2, sin2 = rope(x, seq_len=16)
        
        # Should use cached values
        assert torch.equal(cos1, cos2)
        assert torch.equal(sin1, sin2)


class TestRotaryFunctions:
    """Test rotary embedding utility functions"""
    
    def test_rotate_half(self):
        """Test rotate_half function"""
        x = torch.tensor([[1., 2., 3., 4.]])
        rotated = rotate_half(x)
        expected = torch.tensor([[-3., -4., 1., 2.]])
        assert torch.allclose(rotated, expected)
    
    def test_apply_rotary_pos_emb(self):
        """Test applying rotary embeddings to Q and K"""
        q = torch.randn(2, 8, 16, 64)  # batch, heads, seq, dim
        k = torch.randn(2, 8, 16, 64)
        cos = torch.randn(1, 1, 16, 64)
        sin = torch.randn(1, 1, 16, 64)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()


class TestSwiGLU:
    """Test SwiGLU activation"""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            n_embd=768,
            intermediate_size=3072,
            mlp_bias=False
        )
    
    def test_initialization(self, config):
        """Test SwiGLU initialization"""
        swiglu = SwiGLU(config)
        assert swiglu.hidden_size == 768
        assert swiglu.intermediate_size == 3072
        assert swiglu.gate_proj.weight.shape == (3072, 768)
        assert swiglu.up_proj.weight.shape == (3072, 768)
        assert swiglu.down_proj.weight.shape == (768, 3072)
    
    def test_forward_pass(self, config):
        """Test SwiGLU forward pass"""
        swiglu = SwiGLU(config)
        x = torch.randn(2, 16, 768)
        output = swiglu(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_no_bias(self, config):
        """Test that bias is not used when mlp_bias=False"""
        swiglu = SwiGLU(config)
        assert swiglu.gate_proj.bias is None
        assert swiglu.up_proj.bias is None
        assert swiglu.down_proj.bias is None


class TestGroupedQueryAttention:
    """Test Grouped Query Attention"""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            n_embd=768,
            n_head=12,
            n_kv_head=4,  # GQA with 3 groups
            attention_bias=False,
            attention_dropout=0.0
        )
    
    def test_initialization(self, config):
        """Test GQA initialization"""
        gqa = GroupedQueryAttention(config)
        
        assert gqa.num_heads == 12
        assert gqa.num_kv_heads == 4
        assert gqa.num_kv_groups == 3
        assert gqa.head_dim == 64  # 768 / 12
        
        # Check projection sizes
        assert gqa.q_proj.weight.shape == (768, 768)  # num_heads * head_dim
        assert gqa.k_proj.weight.shape == (256, 768)  # num_kv_heads * head_dim
        assert gqa.v_proj.weight.shape == (256, 768)
    
    def test_forward_pass(self, config):
        """Test GQA forward pass"""
        gqa = GroupedQueryAttention(config)
        x = torch.randn(2, 16, 768)  # batch, seq, hidden
        
        output, attn_weights = gqa(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_with_attention_mask(self, config):
        """Test GQA with attention mask"""
        gqa = GroupedQueryAttention(config)
        x = torch.randn(2, 16, 768)
        attention_mask = torch.ones(2, 16, dtype=torch.bool)
        attention_mask[:, 8:] = False  # Mask last half
        
        output, _ = gqa(x, attention_mask=attention_mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_causal_attention(self, config):
        """Test causal (autoregressive) attention"""
        gqa = GroupedQueryAttention(config)
        x = torch.randn(2, 16, 768)
        
        output, attn_weights = gqa(x, is_causal=True)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestModelConfig:
    """Test model configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ModelConfig()
        
        assert config.vocab_size == 100352
        assert config.n_positions == 8192
        assert config.n_embd == 4096
        assert config.n_layer == 64
        assert config.activation == "swiglu"
        assert config.norm_type == "rmsnorm"
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ModelConfig(
            vocab_size=50000,
            n_embd=512,
            n_layer=6,
            n_head=8
        )
        
        assert config.vocab_size == 50000
        assert config.n_embd == 512
        assert config.n_layer == 6
        assert config.n_head == 8
    
    def test_gqa_compatibility(self):
        """Test that GQA heads are compatible"""
        config = ModelConfig(n_head=12, n_kv_head=4)
        assert config.n_head % config.n_kv_head == 0
        
        # This should work - 12 heads, 4 KV heads = 3 groups
        gqa = GroupedQueryAttention(config)
        assert gqa.num_kv_groups == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
