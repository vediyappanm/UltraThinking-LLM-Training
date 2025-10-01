"""Unit tests for Mixture of Experts components"""
import pytest
import torch
import torch.nn as nn
from src.models.moe_advanced import (
    ExpertConfig,
    Expert,
    TopKRouter,
    MoELayer,
    LoadBalancingLoss
)


class TestExpertConfig:
    """Test expert configuration"""
    
    def test_default_config(self):
        """Test default expert configuration"""
        config = ExpertConfig()
        assert config.num_knowledge_experts == 8
        assert config.num_skill_experts == 8
        assert config.num_meta_experts == 4
        assert config.top_k == 2
    
    def test_custom_config(self):
        """Test custom expert configuration"""
        config = ExpertConfig(
            num_knowledge_experts=16,
            top_k=4,
            expert_capacity=256
        )
        assert config.num_knowledge_experts == 16
        assert config.top_k == 4
        assert config.expert_capacity == 256


class TestExpert:
    """Test individual expert module"""
    
    def test_initialization(self):
        """Test expert initialization"""
        expert = Expert(hidden_dim=768, intermediate_dim=3072)
        assert expert.fc1.weight.shape == (3072, 768)
        assert expert.fc2.weight.shape == (768, 3072)
    
    def test_forward_pass(self):
        """Test expert forward pass"""
        expert = Expert(hidden_dim=768, intermediate_dim=3072)
        x = torch.randn(4, 16, 768)
        output = expert(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTopKRouter:
    """Test Top-K routing mechanism"""
    
    @pytest.fixture
    def router(self):
        return TopKRouter(hidden_dim=768, num_experts=8, top_k=2)
    
    def test_initialization(self, router):
        """Test router initialization"""
        assert router.num_experts == 8
        assert router.top_k == 2
        assert router.gate.weight.shape == (8, 768)
    
    def test_forward_pass(self, router):
        """Test router forward pass"""
        x = torch.randn(4, 16, 768)
        expert_weights, expert_indices = router(x)
        
        batch, seq, top_k = expert_weights.shape
        assert expert_weights.shape == (4, 16, 2)
        assert expert_indices.shape == (4, 16, 2)
        
        # Check weights sum to 1
        assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(4, 16), atol=1e-5)
        
        # Check indices are valid
        assert (expert_indices >= 0).all()
        assert (expert_indices < 8).all()
    
    def test_routing_distribution(self, router):
        """Test that routing distributes across experts"""
        x = torch.randn(32, 128, 768)  # Larger batch
        _, expert_indices = router(x)
        
        # Count expert assignments
        unique_experts = torch.unique(expert_indices)
        
        # Should use multiple experts (not just one)
        assert len(unique_experts) > 2


class TestMoELayer:
    """Test complete MoE layer"""
    
    @pytest.fixture
    def config(self):
        return ExpertConfig(
            num_knowledge_experts=8,
            num_skill_experts=0,  # Only knowledge experts for simplicity
            top_k=2,
            expert_capacity=128
        )
    
    def test_initialization(self, config):
        """Test MoE layer initialization"""
        moe = MoELayer(hidden_dim=768, config=config)
        assert len(moe.knowledge_experts) == 8
        assert moe.router.num_experts == 8
    
    def test_forward_pass(self, config):
        """Test MoE layer forward pass"""
        moe = MoELayer(hidden_dim=768, config=config)
        x = torch.randn(2, 16, 768)
        
        output, aux_loss = moe(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert aux_loss >= 0  # Load balancing loss should be non-negative
    
    def test_load_balancing(self, config):
        """Test that load balancing loss encourages distribution"""
        moe = MoELayer(hidden_dim=768, config=config)
        x = torch.randn(32, 128, 768)  # Large batch
        
        _, aux_loss = moe(x)
        
        # Load balancing loss should be present
        assert aux_loss > 0
        assert aux_loss < 1.0  # Reasonable range
    
    def test_expert_capacity(self, config):
        """Test expert capacity constraints"""
        config.expert_capacity = 16  # Small capacity
        moe = MoELayer(hidden_dim=768, config=config)
        
        x = torch.randn(64, 128, 768)  # Large input
        output, _ = moe(x)
        
        # Should still produce output without errors
        assert output.shape == x.shape


class TestLoadBalancingLoss:
    """Test load balancing loss computation"""
    
    def test_balanced_routing(self):
        """Test loss with perfectly balanced routing"""
        # Perfectly uniform routing
        expert_counts = torch.ones(8) * 100  # Each expert gets 100 tokens
        expert_probs = torch.ones(8) * 0.125  # Each expert has 1/8 probability
        
        loss = LoadBalancingLoss()(expert_counts, expert_probs, num_experts=8)
        
        # Should be close to 0 for perfect balance
        assert loss < 0.01
    
    def test_imbalanced_routing(self):
        """Test loss with imbalanced routing"""
        # Imbalanced: most tokens go to first expert
        expert_counts = torch.tensor([700., 100., 100., 0., 0., 0., 0., 0.])
        expert_probs = torch.tensor([0.7, 0.1, 0.1, 0.1, 0., 0., 0., 0.])
        
        loss = LoadBalancingLoss()(expert_counts, expert_probs, num_experts=8)
        
        # Should have significant loss
        assert loss > 0.1


class TestMoEIntegration:
    """Integration tests for MoE system"""
    
    def test_gradient_flow(self):
        """Test that gradients flow through MoE"""
        config = ExpertConfig(num_knowledge_experts=4, top_k=2)
        moe = MoELayer(hidden_dim=128, config=config)
        
        x = torch.randn(2, 8, 128, requires_grad=True)
        output, aux_loss = moe(x)
        
        # Compute loss and backward
        loss = output.sum() + aux_loss
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert moe.router.gate.weight.grad is not None
        
        # Check gradients are reasonable
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_different_batch_sizes(self):
        """Test MoE with different batch sizes"""
        config = ExpertConfig(num_knowledge_experts=8, top_k=2)
        moe = MoELayer(hidden_dim=256, config=config)
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 32, 256)
            output, aux_loss = moe(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
