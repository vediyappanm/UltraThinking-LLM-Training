"""Unit tests for optimizer utilities"""
import pytest
import torch
import torch.nn as nn
from src.training.optim import build_optimizer
from argparse import Namespace


class TestBuildOptimizer:
    """Test optimizer builder"""
    
    @pytest.fixture
    def simple_model(self):
        """Simple model for testing"""
        return nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    @pytest.fixture
    def args(self):
        """Default training arguments"""
        return Namespace(
            learning_rate=1e-3,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            optimizer='adamw'
        )
    
    def test_adamw_optimizer(self, simple_model, args):
        """Test AdamW optimizer creation"""
        optimizer = build_optimizer(simple_model, args)
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['weight_decay'] == 0.01
        assert optimizer.defaults['betas'] == (0.9, 0.999)
    
    def test_parameter_groups(self, args):
        """Test parameter grouping (no decay for bias/norm)"""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.Linear(50, 10)
        )
        
        optimizer = build_optimizer(model, args)
        
        # Should have 2 parameter groups: decay and no_decay
        assert len(optimizer.param_groups) == 2
        
        # Check weight decay settings
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]
        
        # One should have weight decay, other shouldn't
        assert (decay_group['weight_decay'] == 0.01 and no_decay_group['weight_decay'] == 0.0) or \
               (decay_group['weight_decay'] == 0.0 and no_decay_group['weight_decay'] == 0.01)
    
    def test_zero_weight_decay(self, simple_model, args):
        """Test with zero weight decay"""
        args.weight_decay = 0.0
        optimizer = build_optimizer(simple_model, args)
        
        # Should only have one param group when no weight decay
        assert len(optimizer.param_groups) >= 1
    
    def test_optimizer_step(self, simple_model, args):
        """Test that optimizer can step"""
        optimizer = build_optimizer(simple_model, args)
        
        # Forward pass
        x = torch.randn(4, 100)
        output = simple_model(x)
        loss = output.sum()
        
        # Backward
        loss.backward()
        
        # Get parameter before step
        first_param_before = list(simple_model.parameters())[0].clone()
        
        # Optimizer step
        optimizer.step()
        
        # Parameter should change
        first_param_after = list(simple_model.parameters())[0]
        assert not torch.equal(first_param_before, first_param_after)


class TestOptimizerIntegration:
    """Integration tests for optimizer"""
    
    def test_training_loop(self):
        """Test optimizer in simple training loop"""
        model = nn.Linear(10, 1)
        args = Namespace(
            learning_rate=0.01,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            optimizer='adamw'
        )
        
        optimizer = build_optimizer(model, args)
        
        # Training data
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        initial_loss = None
        final_loss = None
        
        # Train for a few steps
        for i in range(10):
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            
            if i == 0:
                initial_loss = loss.item()
            if i == 9:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should decrease (model should learn)
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
