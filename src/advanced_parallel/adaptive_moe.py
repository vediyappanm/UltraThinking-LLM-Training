"""
Adaptive MoE (Dynamic Expert Routing)
Let experts grow/shrink and self-balance load during training
Improves efficiency & generalization over static MoE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveMoEConfig:
    """Configuration for adaptive MoE"""
    num_experts: int = 64
    hidden_size: int = 4096
    intermediate_size: int = 14336
    top_k: int = 2
    
    # Adaptive routing
    enable_dynamic_routing: bool = True
    routing_temperature: float = 1.0
    routing_temperature_decay: float = 0.999
    
    # Expert management
    enable_expert_pruning: bool = True
    prune_threshold: float = 0.01  # Prune experts used < 1%
    enable_expert_growing: bool = True
    grow_threshold: float = 0.95  # Grow when utilization > 95%
    
    # Load balancing
    load_balance_weight: float = 0.01
    adaptive_load_balance: bool = True
    target_expert_usage: float = 1.0 / 64  # Uniform distribution
    
    # Capacity
    capacity_factor: float = 1.25
    adaptive_capacity: bool = True
    min_capacity_factor: float = 1.0
    max_capacity_factor: float = 2.0


class AdaptiveRouter(nn.Module):
    """
    Adaptive router with dynamic temperature and load balancing
    Learns to route tokens efficiently while maintaining balance
    """
    
    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config
        
        # Router network
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Adaptive parameters
        self.register_buffer('temperature', torch.tensor(config.routing_temperature))
        self.register_buffer('expert_usage', torch.zeros(config.num_experts))
        self.register_buffer('expert_capacity_factors', torch.ones(config.num_experts) * config.capacity_factor)
        
        # Statistics
        self.register_buffer('routing_history', torch.zeros(config.num_experts, 100))
        self.history_idx = 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Adaptive routing with dynamic load balancing
        
        Args:
            hidden_states: [batch*seq, hidden_size]
            training: Whether in training mode
        
        Returns:
            expert_indices: [batch*seq, top_k]
            expert_weights: [batch*seq, top_k]
            aux_losses: Dictionary of auxiliary losses
        """
        batch_seq, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch*seq, num_experts]
        
        # Apply temperature scaling
        if self.config.enable_dynamic_routing:
            router_logits = router_logits / self.temperature
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.config.top_k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Update expert usage statistics
        if training:
            self._update_expert_usage(routing_probs)
            
            # Adaptive temperature decay
            if self.config.enable_dynamic_routing:
                self.temperature *= self.config.routing_temperature_decay
                self.temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(
            router_logits,
            routing_probs,
            top_k_indices,
            batch_seq,
        )
        
        return top_k_indices, top_k_weights, aux_losses
    
    def _update_expert_usage(self, routing_probs: torch.Tensor):
        """Update expert usage statistics"""
        # Average routing probability per expert
        expert_probs = routing_probs.mean(dim=0)
        
        # Exponential moving average
        alpha = 0.01
        self.expert_usage = (1 - alpha) * self.expert_usage + alpha * expert_probs
        
        # Update routing history
        self.routing_history[:, self.history_idx] = expert_probs
        self.history_idx = (self.history_idx + 1) % 100
    
    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        routing_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing"""
        aux_losses = {}
        
        # Expert usage (fraction of tokens routed to each expert)
        expert_counts = torch.zeros(self.config.num_experts, device=routing_probs.device)
        for k in range(self.config.top_k):
            expert_counts.scatter_add_(0, top_k_indices[:, k], torch.ones_like(top_k_indices[:, k], dtype=torch.float32))
        
        expert_usage = expert_counts / (num_tokens * self.config.top_k)
        
        # Load balance loss
        if self.config.adaptive_load_balance:
            # Adaptive: penalize deviation from target usage
            target_usage = self.config.target_expert_usage
            load_balance_loss = torch.var(expert_usage) * self.config.num_experts
        else:
            # Standard: encourage uniform distribution
            load_balance_loss = torch.var(expert_usage)
        
        aux_losses['load_balance_loss'] = load_balance_loss * self.config.load_balance_weight
        
        # Router z-loss (encourage smaller logits)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        aux_losses['router_z_loss'] = router_z_loss * 0.001
        
        # Entropy regularization (encourage diversity)
        routing_entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=-1).mean()
        aux_losses['routing_entropy'] = routing_entropy
        
        # Expert usage stats
        aux_losses['expert_usage'] = expert_usage
        aux_losses['expert_usage_std'] = torch.std(expert_usage)
        
        return aux_losses
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get expert usage and routing statistics"""
        return {
            'expert_usage': self.expert_usage.cpu().numpy(),
            'temperature': self.temperature.item(),
            'routing_history': self.routing_history.cpu().numpy(),
            'capacity_factors': self.expert_capacity_factors.cpu().numpy(),
        }


class DynamicExpertPool(nn.Module):
    """
    Dynamic expert pool that can grow/shrink experts
    Prunes underutilized experts and adds new ones when needed
    """
    
    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config
        
        # Initial expert pool
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(config.num_experts)
        ])
        
        # Expert metadata
        self.register_buffer('expert_active', torch.ones(config.num_experts, dtype=torch.bool))
        self.register_buffer('expert_usage_count', torch.zeros(config.num_experts))
        
        self.num_active_experts = config.num_experts
    
    def _create_expert(self) -> nn.Module:
        """Create a single expert (FFN)"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route tokens through experts
        
        Args:
            hidden_states: [batch*seq, hidden_size]
            expert_indices: [batch*seq, top_k]
            expert_weights: [batch*seq, top_k]
        
        Returns:
            output: [batch*seq, hidden_size]
        """
        batch_seq, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(len(self.experts)):
            if not self.expert_active[expert_idx]:
                continue
            
            # Find tokens routed to this expert
            mask = (expert_indices == expert_idx)
            if not mask.any():
                continue
            
            # Get token indices and weights
            token_indices = mask.nonzero(as_tuple=True)[0]
            weights = expert_weights[mask]
            
            # Update usage count
            self.expert_usage_count[expert_idx] += token_indices.size(0)
            
            # Process through expert
            expert_input = hidden_states[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weighted add to output
            output[token_indices] += expert_output * weights.unsqueeze(-1)
        
        return output
    
    def prune_experts(self, usage_threshold: float = 0.01):
        """Prune underutilized experts"""
        if not self.config.enable_expert_pruning:
            return
        
        total_usage = self.expert_usage_count.sum()
        if total_usage == 0:
            return
        
        expert_usage_ratio = self.expert_usage_count / total_usage
        
        # Mark experts below threshold as inactive
        for expert_idx in range(len(self.experts)):
            if expert_usage_ratio[expert_idx] < usage_threshold:
                self.expert_active[expert_idx] = False
                logger.info(f"Pruned expert {expert_idx} (usage: {expert_usage_ratio[expert_idx]:.4f})")
        
        self.num_active_experts = self.expert_active.sum().item()
    
    def grow_experts(self, num_new_experts: int = 1):
        """Add new experts to the pool"""
        if not self.config.enable_expert_growing:
            return
        
        for _ in range(num_new_experts):
            new_expert = self._create_expert()
            self.experts.append(new_expert)
            
            # Expand metadata buffers
            self.expert_active = torch.cat([self.expert_active, torch.tensor([True])])
            self.expert_usage_count = torch.cat([self.expert_usage_count, torch.tensor([0.0])])
        
        self.num_active_experts += num_new_experts
        logger.info(f"Added {num_new_experts} new experts (total: {len(self.experts)})")


class AdaptiveMoELayer(nn.Module):
    """
    Complete adaptive MoE layer with dynamic routing and expert management
    """
    
    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config
        
        # Adaptive router
        self.router = AdaptiveRouter(config)
        
        # Dynamic expert pool
        self.expert_pool = DynamicExpertPool(config)
        
        # Management counters
        self.step_count = 0
        self.prune_interval = 1000
        self.grow_check_interval = 500
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with adaptive routing
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            output: [batch, seq, hidden_size]
            aux_losses: Auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_flat = hidden_states.reshape(-1, hidden_size)
        
        # Route tokens
        expert_indices, expert_weights, aux_losses = self.router(hidden_flat, self.training)
        
        # Process through experts
        output_flat = self.expert_pool(hidden_flat, expert_indices, expert_weights)
        
        # Reshape output
        output = output_flat.reshape(batch_size, seq_len, hidden_size)
        
        # Periodic expert management
        if self.training:
            self.step_count += 1
            
            # Prune underutilized experts
            if self.step_count % self.prune_interval == 0:
                self.expert_pool.prune_experts(self.config.prune_threshold)
            
            # Check if we need to grow experts
            if self.step_count % self.grow_check_interval == 0:
                self._check_and_grow_experts(aux_losses)
        
        return output, aux_losses
    
    def _check_and_grow_experts(self, aux_losses: Dict[str, Any]):
        """Check if we need to add more experts"""
        if not self.config.enable_expert_growing:
            return
        
        # Get expert usage
        expert_usage = aux_losses.get('expert_usage')
        if expert_usage is None:
            return
        
        # Check if any expert is overloaded
        max_usage = expert_usage.max().item()
        if max_usage > self.config.grow_threshold:
            # Add new experts
            num_new = max(1, int(self.config.num_experts * 0.1))  # Add 10% more
            self.expert_pool.grow_experts(num_new)
            logger.info(f"Growing expert pool: max usage {max_usage:.2f} > threshold {self.config.grow_threshold}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        router_stats = self.router.get_expert_statistics()
        
        return {
            'router': router_stats,
            'num_experts': len(self.expert_pool.experts),
            'num_active_experts': self.expert_pool.num_active_experts,
            'expert_usage_count': self.expert_pool.expert_usage_count.cpu().numpy(),
        }
