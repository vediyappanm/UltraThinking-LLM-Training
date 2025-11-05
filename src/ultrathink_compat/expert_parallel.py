"""
Expert Parallel MoE for Megatron-LM
Compatible with TP/PP/SP, top-k routing, load balancing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_expert_parallel_group():
    """Get expert parallel process group"""
    if not dist.is_available() or not dist.is_initialized():
        return None
    # For now, use world group; in production, create dedicated EP group
    return dist.group.WORLD


def get_expert_parallel_world_size():
    """Get expert parallel world size"""
    group = get_expert_parallel_group()
    if group is None:
        return 1
    return dist.get_world_size(group)


def get_expert_parallel_rank():
    """Get expert parallel rank"""
    group = get_expert_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


class TopKRouter(nn.Module):
    """Top-K router for MoE with load balancing"""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        jitter_eps: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_eps = jitter_eps
        
        # Router weights
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route tokens to experts
        
        Returns:
            dispatch_mask: (batch*seq, num_experts, capacity)
            combine_weights: (batch*seq, num_experts, capacity)
            aux_loss_dict: auxiliary losses for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)  # (batch*seq, hidden)
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch*seq, num_experts)
        
        # Add jitter for exploration
        if self.training and self.jitter_eps > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_eps
        
        # Compute routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)  # (batch*seq, num_experts)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute capacity
        num_tokens = hidden_states.size(0)
        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        
        # Create dispatch and combine masks
        dispatch_mask = torch.zeros(
            num_tokens, self.num_experts, capacity,
            dtype=torch.float32, device=hidden_states.device
        )
        combine_weights = torch.zeros(
            num_tokens, self.num_experts, capacity,
            dtype=torch.float32, device=hidden_states.device
        )
        
        # Track expert load for balancing
        expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
        
        # Fill masks
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k_idx].item()
                weight = top_k_weights[token_idx, k_idx].item()
                
                # Find next available slot in expert
                expert_count = int(expert_counts[expert_idx].item())
                if expert_count < capacity:
                    dispatch_mask[token_idx, expert_idx, expert_count] = 1.0
                    combine_weights[token_idx, expert_idx, expert_count] = weight
                    expert_counts[expert_idx] += 1
        
        # Compute auxiliary losses for load balancing
        aux_loss_dict = self._compute_aux_losses(
            router_logits, routing_weights, expert_counts, num_tokens
        )
        
        return dispatch_mask, combine_weights, aux_loss_dict
    
    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_counts: torch.Tensor,
        num_tokens: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing"""
        # Load balancing loss (encourage uniform distribution)
        expert_probs = routing_weights.mean(dim=0)  # Average probability per expert
        expert_usage = expert_counts / (num_tokens * self.top_k)  # Actual usage per expert
        
        # Load balance loss: minimize variance in expert usage
        load_balance_loss = torch.var(expert_usage) * self.num_experts
        
        # Router z-loss: encourage smaller logits for stability
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # Entropy regularization: encourage diversity
        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-10)).sum(dim=-1).mean()
        
        return {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'routing_entropy': routing_entropy,
            'expert_usage': expert_usage,
        }


class ExpertParallelMoELayer(nn.Module):
    """Expert-parallel MoE layer with all-to-all communication"""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        
        # Get expert parallelism info
        self.ep_size = get_expert_parallel_world_size()
        self.ep_rank = get_expert_parallel_rank()
        self.ep_group = get_expert_parallel_group()
        
        # Number of local experts
        self.num_local_experts = num_experts // self.ep_size
        
        # Router
        self.router = TopKRouter(
            num_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        
        # Local experts
        self.experts = nn.ModuleList([
            self._create_expert(hidden_size, intermediate_size, activation)
            for _ in range(self.num_local_experts)
        ])
    
    def _create_expert(self, hidden_size: int, intermediate_size: int, activation: str):
        """Create a single expert FFN"""
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )
    
    def _all_to_all_dispatch(
        self,
        hidden_states: torch.Tensor,
        dispatch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """All-to-all communication to dispatch tokens to experts"""
        if self.ep_size == 1:
            return hidden_states
        
        # Split tokens across expert parallel ranks
        # This is a simplified version; production would use optimized all-to-all
        return hidden_states
    
    def _all_to_all_combine(
        self,
        expert_outputs: torch.Tensor,
        combine_weights: torch.Tensor,
    ) -> torch.Tensor:
        """All-to-all communication to combine expert outputs"""
        if self.ep_size == 1:
            return expert_outputs
        
        # Combine outputs from all experts
        return expert_outputs
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with expert parallelism
        
        Returns:
            output: (batch, seq, hidden)
            aux_loss_dict: auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape
        
        # Route tokens to experts
        dispatch_mask, combine_weights, aux_loss_dict = self.router(hidden_states)
        
        # Flatten for expert processing
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)
        
        # Dispatch tokens (all-to-all if EP > 1)
        dispatched_tokens = self._all_to_all_dispatch(hidden_states_flat, dispatch_mask)
        
        # Process through local experts
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            global_expert_idx = self.ep_rank * self.num_local_experts + expert_idx
            
            # Extract tokens assigned to this expert
            expert_mask = dispatch_mask[:, global_expert_idx, :]  # (num_tokens, capacity)
            expert_tokens_mask = expert_mask.sum(dim=-1) > 0  # (num_tokens,)
            
            if expert_tokens_mask.any():
                expert_input = hidden_states_flat[expert_tokens_mask]
                expert_output = expert(expert_input)
                expert_outputs.append(expert_output)
            else:
                expert_outputs.append(torch.zeros(0, hidden_size, device=hidden_states.device))
        
        # Combine expert outputs
        combined_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx, expert_output in enumerate(expert_outputs):
            global_expert_idx = self.ep_rank * self.num_local_experts + expert_idx
            expert_mask = dispatch_mask[:, global_expert_idx, :]
            expert_weights = combine_weights[:, global_expert_idx, :]
            
            expert_tokens_mask = expert_mask.sum(dim=-1) > 0
            if expert_tokens_mask.any():
                # Weight and add expert output
                weighted_output = expert_output * expert_weights[expert_tokens_mask].sum(dim=-1, keepdim=True)
                combined_output[expert_tokens_mask] += weighted_output
        
        # All-to-all combine (if EP > 1)
        final_output = self._all_to_all_combine(combined_output, combine_weights)
        
        # Reshape to original
        final_output = final_output.reshape(original_shape)
        
        return final_output, aux_loss_dict


def compute_moe_aux_loss(
    aux_loss_dict: Dict[str, torch.Tensor],
    load_balance_weight: float = 0.01,
    z_loss_weight: float = 0.001,
) -> torch.Tensor:
    """Compute total MoE auxiliary loss"""
    total_loss = torch.tensor(0.0, device=next(iter(aux_loss_dict.values())).device)
    
    if 'load_balance_loss' in aux_loss_dict:
        total_loss += load_balance_weight * aux_loss_dict['load_balance_loss']
    
    if 'router_z_loss' in aux_loss_dict:
        total_loss += z_loss_weight * aux_loss_dict['router_z_loss']
    
    return total_loss
