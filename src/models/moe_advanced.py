"""
Advanced Mixture of Experts (MoE³) Architecture
Implements hierarchical, consultative experts with domain specialization
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# Safe import for torch.compile disable
try:
    from torch._dynamo import disable as torchdynamo_disable
except Exception:  # pragma: no cover
    def torchdynamo_disable(fn):
        return fn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of experts in the hierarchy"""
    KNOWLEDGE = "knowledge"  # Domain knowledge experts
    SKILL = "skill"  # Task-specific skill experts  
    META = "meta"  # Meta-reasoning experts
    SAFETY = "safety"  # Safety and alignment experts


@dataclass
class ExpertConfig:
    """Configuration for expert modules"""
    num_knowledge_experts: int = 64
    num_skill_experts: int = 32
    num_meta_experts: int = 16
    num_safety_experts: int = 8
    expert_capacity: float = 1.25  # Capacity factor for load balancing
    expert_dropout: float = 0.1
    
    # Load balancing loss weights (Switch Transformers approach)
    load_balance_weight: float = 0.01  # Primary load balancing loss
    z_loss_weight: float = 0.001  # Router logit regularization
    importance_weight: float = 0.01  # Routing diversity loss
    entropy_reg_weight: float = 1.0  # NUCLEAR FIX: Direct entropy regularization
    aux_loss_weight: float = 0.01  # Legacy - kept for compatibility
    
    top_k: int = 2  # Number of experts to route to
    moe_dtype: torch.dtype = torch.float32
    expert_parallelism: bool = True
    consultative_attention: bool = True
    hierarchical_routing: bool = True


class NoisyTopKRouter(nn.Module):
    """Noisy Top-K routing with load balancing"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Router network - use float32
        # Router gate with proper initialization
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=torch.float32)
        # Initialize with small uniform weights to encourage balanced routing
        with torch.no_grad():
            self.gate.weight.uniform_(-0.01, 0.01)
        
        # Learnable noise parameters
        self.noise_linear = nn.Linear(hidden_dim, num_experts, dtype=torch.float32)
        # Initialize noise layer with small weights
        with torch.no_grad():
            self.noise_linear.weight.uniform_(-0.005, 0.005)  # Even smaller for noise
        
    @torchdynamo_disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route to top-k experts with load balancing
        Returns: (dispatch_mask, combine_weights, aux_losses)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [B*S, H]

        # Preserve original dtype of activations; router math in fp32
        original_dtype = hidden_states.dtype

        # Disable AMP autocast inside router to avoid Float/Half addmm mismatches
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Cast activations to fp32 for stable routing
            hs_fp32 = hidden_states_flat.to(dtype=torch.float32)

            # Router logits in fp32
            logits = self.gate(hs_fp32)  # [B*S, E], fp32 weights by construction

            # Add noise during training for exploration (fp32)
            if training and self.noise_std > 0:
                noise_logits = self.noise_linear(hs_fp32)
                noise = torch.randn_like(logits) * F.softplus(noise_logits)
                logits = logits + noise * self.noise_std

            # Softmax/topk in fp32
            scores = F.softmax(logits, dim=-1)
            
            # Top-k selection (ensure k doesn't exceed number of experts)
            actual_top_k = min(self.top_k, self.num_experts)
            top_k_scores, top_k_indices = torch.topk(scores, actual_top_k, dim=-1)

            # Renormalize and cast back to original dtype for downstream use
            top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
            top_k_scores = top_k_scores.to(original_dtype)
        
        # Create dispatch mask [B*S, E, K] - use actual_top_k
        dispatch_mask = torch.zeros(
            batch_size * seq_len, self.num_experts, actual_top_k,
            dtype=torch.bool, device=hidden_states.device
        )
        
        # Create combine weights [B*S, E, K] - use actual_top_k
        combine_weights = torch.zeros(
            batch_size * seq_len, self.num_experts, actual_top_k,
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # Fill dispatch mask and weights
        for k in range(actual_top_k):
            expert_idx = top_k_indices[:, k]  # [B*S]
            dispatch_mask[torch.arange(batch_size * seq_len), expert_idx, k] = True
            combine_weights[torch.arange(batch_size * seq_len), expert_idx, k] = top_k_scores[:, k]
        
        # Compute auxiliary losses for load balancing
        aux_losses = self._compute_aux_losses(scores, dispatch_mask)
        
        # Reshape back
        dispatch_mask = dispatch_mask.view(batch_size, seq_len, self.num_experts, actual_top_k)
        combine_weights = combine_weights.view(batch_size, seq_len, self.num_experts, actual_top_k)
        
        return dispatch_mask, combine_weights, aux_losses
    
    def _compute_aux_losses(
        self,
        scores: torch.Tensor,
        dispatch_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute load balancing auxiliary losses"""
        # Load balancing loss (encourage equal expert usage)
        # Cast to float32 for numerical stability in loss computation
        expert_usage = dispatch_mask.float().sum(dim=(0, 2))  # [E]
        expert_usage = expert_usage / (expert_usage.sum() + 1e-10)
        
        # Ideal uniform distribution
        uniform_distribution = torch.ones_like(expert_usage) / self.num_experts
        load_loss = F.kl_div(
            torch.log(expert_usage + 1e-10),
            uniform_distribution,
            reduction='batchmean'
        ).float()
        
        # Importance loss (encourage diversity in routing)
        importance = scores.mean(dim=0).float()  # [E]
        if importance.numel() < 2:
            importance_loss = torch.tensor(0.0, device=importance.device, dtype=torch.float32)
        else:
            importance_loss = torch.var(importance, unbiased=False) / (torch.mean(importance) ** 2 + 1e-10)
        
        # NUCLEAR FIX: Direct entropy regularization to force balanced routing
        # Calculate entropy of routing distribution
        routing_entropy = -torch.sum(scores * torch.log(scores + 1e-10), dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=scores.device))
        
        # Entropy regularization loss - encourage high entropy (uniform distribution)
        entropy_reg_loss = (max_entropy - routing_entropy) * 10.0  # Strong penalty for low entropy
        
        # Z-loss (encourage router confidence)
        log_z = torch.logsumexp(scores, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        
        return {
            'load_loss': load_loss,
            'importance_loss': importance_loss,
            'z_loss': z_loss,
            'entropy_reg_loss': entropy_reg_loss,
            'expert_usage': expert_usage,
            'routing_entropy': -torch.sum(expert_usage * torch.log(expert_usage + 1e-10))
        }


class Expert(nn.Module):
    """Individual expert module"""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        specialization: Optional[str] = None
    ):
        super().__init__()
        
        self.specialization = specialization
        
        # Expert MLP
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        # Specialization-specific components
        if specialization:
            self.specialization_layer = nn.Linear(hidden_dim, hidden_dim)
            self.specialization_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert forward pass"""
        # Main expert computation
        hidden = self.fc1(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        
        # Apply specialization if present
        if self.specialization:
            specialized = self.specialization_layer(output)
            specialized = self.specialization_norm(specialized)
            output = output + specialized
        
        return output


class CrossExpertAttention(nn.Module):
    """Attention mechanism for expert consultation"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Allow experts to attend to each other"""
        if len(expert_outputs) == 0:
            return torch.zeros_like(expert_outputs[0])
        
        # Stack expert outputs [num_experts, batch, seq, hidden]
        stacked = torch.stack(expert_outputs, dim=0)
        num_experts, batch_size, seq_len, hidden_dim = stacked.shape
        
        # Reshape for attention [batch, num_experts * seq, hidden]
        reshaped = stacked.permute(1, 0, 2, 3).reshape(
            batch_size, num_experts * seq_len, hidden_dim
        )
        
        # Compute attention
        Q = self.q_proj(reshaped).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(reshaped).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(reshaped).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply weights if provided
        if weights is not None:
            # Expand weights to match attention shape
            weights_expanded = weights.unsqueeze(1).expand_as(scores)
            scores = scores * weights_expanded
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, hidden_dim)
        
        # Output projection and reshape back
        output = self.o_proj(attn_output)
        output = output.view(batch_size, num_experts, seq_len, hidden_dim)
        output = output.mean(dim=1)  # Average over experts
        
        return output


class HierarchicalMoE(nn.Module):
    """Hierarchical Mixture of Experts with multiple levels"""
    
    def __init__(self, config: ExpertConfig, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        
        # Knowledge experts (domain-specific)
        self.knowledge_experts = nn.ModuleList([
            Expert(
                hidden_dim, intermediate_dim,
                dropout=config.expert_dropout,
                specialization=f"knowledge_{i}"
            )
            for i in range(config.num_knowledge_experts)
        ])
        
        # Skill experts (task-specific)
        self.skill_experts = nn.ModuleList([
            Expert(
                hidden_dim, intermediate_dim,
                dropout=config.expert_dropout,
                specialization=f"skill_{i}"
            )
            for i in range(config.num_skill_experts)
        ])
        
        # Meta experts (reasoning and planning)
        self.meta_experts = nn.ModuleList([
            Expert(
                hidden_dim, intermediate_dim,
                dropout=config.expert_dropout,
                specialization=f"meta_{i}"
            )
            for i in range(config.num_meta_experts)
        ])
        
        # Safety experts (alignment and safety)
        self.safety_experts = nn.ModuleList([
            Expert(
                hidden_dim, intermediate_dim,
                dropout=config.expert_dropout,
                specialization="safety"
            )
            for i in range(config.num_safety_experts)
        ])
        
        # Routers for each level
        self.knowledge_router = NoisyTopKRouter(
            hidden_dim, config.num_knowledge_experts, config.top_k
        )
        self.skill_router = NoisyTopKRouter(
            hidden_dim, config.num_skill_experts, config.top_k
        )
        self.meta_router = NoisyTopKRouter(
            hidden_dim, config.num_meta_experts, config.top_k
        )
        self.safety_router = NoisyTopKRouter(
            hidden_dim, config.num_safety_experts, min(config.top_k, config.num_safety_experts)
        )
        
        # Cross-expert attention for consultation
        if config.consultative_attention:
            self.cross_expert_attention = CrossExpertAttention(hidden_dim)
        
        # Hierarchical combination
        if config.hierarchical_routing:
            self.hierarchy_combiner = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.expert_dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
    def _route_and_compute(
        self,
        hidden_states: torch.Tensor,
        experts: nn.ModuleList,
        router: NoisyTopKRouter,
        expert_type: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Route to experts and compute output"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions
        dispatch_mask, combine_weights, aux_losses = router(hidden_states, self.training)
        
        # Initialize output
        expert_output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx, expert in enumerate(experts):
            # Get tokens routed to this expert
            expert_mask = dispatch_mask[:, :, expert_idx, :].any(dim=-1)  # [B, S]
            
            if expert_mask.any():
                # Get input for this expert
                expert_input = hidden_states[expert_mask]
                
                # Compute expert output
                expert_result = expert(expert_input.unsqueeze(0)).squeeze(0)
                
                # Get weights for this expert
                expert_weights = combine_weights[:, :, expert_idx, :].sum(dim=-1)  # [B, S]
                expert_weights = expert_weights[expert_mask]
                
                # Weighted accumulation
                expert_output[expert_mask] += expert_result * expert_weights.unsqueeze(-1)
        
        # Add auxiliary losses with type prefix
        prefixed_losses = {f"{expert_type}_{k}": v for k, v in aux_losses.items()}
        
        return expert_output, prefixed_losses
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_levels: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through hierarchical MoE"""
        
        all_aux_losses = {}
        level_outputs = {}
        
        # Level 1: Knowledge experts
        knowledge_output, knowledge_losses = self._route_and_compute(
            hidden_states, self.knowledge_experts, self.knowledge_router, "knowledge"
        )
        level_outputs['knowledge'] = knowledge_output
        all_aux_losses.update(knowledge_losses)
        
        # Level 2: Skill experts (can see knowledge expert output)
        skill_input = hidden_states + 0.5 * knowledge_output  # Residual connection
        skill_output, skill_losses = self._route_and_compute(
            skill_input, self.skill_experts, self.skill_router, "skill"
        )
        level_outputs['skill'] = skill_output
        all_aux_losses.update(skill_losses)
        
        # Level 3: Meta experts (can see both knowledge and skill)
        meta_input = hidden_states + 0.3 * knowledge_output + 0.3 * skill_output
        meta_output, meta_losses = self._route_and_compute(
            meta_input, self.meta_experts, self.meta_router, "meta"
        )
        level_outputs['meta'] = meta_output
        all_aux_losses.update(meta_losses)
        
        # Level 4: Safety experts (see all previous levels)
        safety_input = hidden_states + 0.2 * (knowledge_output + skill_output + meta_output)
        safety_output, safety_losses = self._route_and_compute(
            safety_input, self.safety_experts, self.safety_router, "safety"
        )
        level_outputs['safety'] = safety_output
        all_aux_losses.update(safety_losses)
        
        # Consultative attention between expert outputs
        if self.config.consultative_attention and hasattr(self, 'cross_expert_attention'):
            expert_outputs_list = [knowledge_output, skill_output, meta_output, safety_output]
            consulted_output = self.cross_expert_attention(expert_outputs_list)
            level_outputs['consulted'] = consulted_output
        
        # Hierarchical combination
        if self.config.hierarchical_routing:
            combined = torch.cat([
                knowledge_output,
                skill_output,
                meta_output,
                safety_output
            ], dim=-1)
            final_output = self.hierarchy_combiner(combined)
        else:
            # Simple weighted combination
            final_output = (
                0.3 * knowledge_output +
                0.3 * skill_output +
                0.2 * meta_output +
                0.2 * safety_output
            )
        
        # Add residual connection
        final_output = hidden_states + final_output
        
        # Calculate expert utilization metrics
        expert_utilization = {}
        total_routing_entropy = 0.0
        
        for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
            if f"{expert_type}_expert_usage" in all_aux_losses:
                usage = all_aux_losses[f"{expert_type}_expert_usage"]
                entropy = all_aux_losses[f"{expert_type}_routing_entropy"]
                
                # Per-expert utilization percentages
                expert_utilization[f"{expert_type}_usage_pct"] = (usage * 100).tolist()
                
                # Load variance (how balanced the routing is)
                if usage.numel() > 1:
                    expert_utilization[f"{expert_type}_load_variance"] = float(torch.var(usage, unbiased=False))
                else:
                    expert_utilization[f"{expert_type}_load_variance"] = 0.0
                
                # Top expert concentration (what % goes to most used expert)
                expert_utilization[f"{expert_type}_top_expert_pct"] = float(torch.max(usage) * 100)
                
                # Routing entropy (higher = more balanced)
                expert_utilization[f"{expert_type}_entropy"] = float(entropy)
                total_routing_entropy += entropy
        
        # Overall MoE health metrics
        expert_utilization['total_routing_entropy'] = float(total_routing_entropy)
        expert_utilization['avg_routing_entropy'] = float(total_routing_entropy / 4)  # 4 expert types
        
        # Prepare info dictionary
        info = {
            'aux_losses': all_aux_losses,
            'level_outputs': level_outputs if return_all_levels else None,
            'num_experts_used': {
                'knowledge': self.config.num_knowledge_experts,
                'skill': self.config.num_skill_experts,
                'meta': self.config.num_meta_experts,
                'safety': self.config.num_safety_experts
            },
            'expert_utilization': expert_utilization
        }
        
        return final_output, info


class MoELayer(nn.Module):
    
    def __init__(
        self,
        config: ExpertConfig,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim or hidden_dim * 4
        
        # Layer normalization
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.post_moe_norm = nn.LayerNorm(hidden_dim)
        
        # Hierarchical MoE
        self.moe = HierarchicalMoE(config, hidden_dim, self.intermediate_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.expert_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through MoE layer"""
        
        # Normalize input
        normed_hidden_states = self.input_norm(hidden_states)
        
        # MoE forward pass
        moe_output, moe_info = self.moe(normed_hidden_states)
        
        # Store MoE info for later access
        self.last_moe_info = moe_info
        
        # Apply dropout
        moe_output = self.dropout(moe_output)
        
        # Post normalization
        output = self.post_moe_norm(moe_output)
        
        # Calculate auxiliary loss
        aux_loss = None
        if return_aux_loss and moe_info.get('aux_losses'):
            aux_losses = moe_info['aux_losses']
            
            # Combine all auxiliary losses with proper weighting
            total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
            
            # Load balancing losses (Switch Transformers approach)
            load_balance_weight = getattr(self.config, 'load_balance_weight', 0.01)
            z_loss_weight = getattr(self.config, 'z_loss_weight', 0.001)
            importance_weight = getattr(self.config, 'importance_weight', 0.01)
            
            for key, loss in aux_losses.items():
                if 'load_loss' in key:
                    # Primary load balancing loss - encourages uniform expert usage
                    total_aux_loss += load_balance_weight * loss
                elif 'z_loss' in key:
                    # Z-loss - prevents router logits from becoming too large
                    total_aux_loss += z_loss_weight * loss
                elif 'importance_loss' in key:
                    # Importance loss - encourages diversity in routing decisions
                    total_aux_loss += importance_weight * loss
                elif 'entropy_reg_loss' in key:
                    # NUCLEAR FIX: Direct entropy regularization
                    entropy_weight = getattr(self.config, 'entropy_reg_weight', 1.0)
                    total_aux_loss += entropy_weight * loss
            
            aux_loss = total_aux_loss
        
        return output, aux_loss


def create_moe_layers(
    num_layers: int,
    config: ExpertConfig,
    hidden_dim: int,
    sparse_layers: Optional[List[int]] = None
) -> nn.ModuleList:
    """Create MoE layers for transformer"""
    
    if sparse_layers is None:
        # Default: every other layer is MoE
        sparse_layers = list(range(1, num_layers, 2))
    
    layers = nn.ModuleList()
    
    for layer_idx in range(num_layers):
        if layer_idx in sparse_layers:
            # MoE layer
            layer = MoELayer(config, hidden_dim)
        else:
            # Regular FFN layer (placeholder - would use actual FFN)
            layer = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(config.expert_dropout)
            )
        
        layers.append(layer)
    
    return layers
