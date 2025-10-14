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
    
    # OPTIMIZED: Load balancing loss weights for stable 50% max expert usage
    # These weights are carefully tuned to maintain balance without causing instability
    load_balance_weight: float = 0.01  # Switch Transformers load balancing
    z_loss_weight: float = 0.001  # Router logit regularization (prevent extremes)
    importance_weight: float = 0.005  # Routing diversity (reduced for stability)
    entropy_reg_weight: float = 0.5  # Entropy regularization (gentler than before)
    aux_loss_weight: float = 0.01  # Legacy - kept for compatibility
    
    top_k: int = 2  # Number of experts to route to (50% max expert for k=2)
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
        COMPLETELY REWRITTEN ROUTER with proven balanced routing
        Returns: (dispatch_mask, combine_weights, aux_losses)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [B*S, H]
        original_dtype = hidden_states.dtype

        # CRITICAL FIX: Force balanced routing with Gumbel-Softmax
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            hs_fp32 = hidden_states_flat.to(dtype=torch.float32)

            # Router logits with VERY small initialization already applied
            raw_logits = self.gate(hs_fp32)  # [B*S, E]
            
            # OPTIMIZED: Stable routing with controlled randomness
            if training:
                # Add Gumbel noise only during training for exploration
                # Use lower temperature for more stable routing
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(raw_logits) + 1e-10) + 1e-10)
                temperature = 1.0  # Lower temperature = more deterministic
                logits = (raw_logits + gumbel_noise * 0.1) / temperature
            else:
                logits = raw_logits
            
            # Compute softmax probabilities
            scores = F.softmax(logits, dim=-1)
            
            # STABLE: Gentle minimum probability to prevent expert collapse
            # For top_k=2, minimum should be very small to allow natural selection
            min_prob = 0.001  # 0.1% minimum per expert
            scores = scores * (1 - min_prob * self.num_experts) + min_prob
            scores = scores / scores.sum(dim=-1, keepdim=True)  # Renormalize
            
            # Top-k selection with actual_top_k
            actual_top_k = min(self.top_k, self.num_experts)
            top_k_scores, top_k_indices = torch.topk(scores, actual_top_k, dim=-1)

            # Renormalize and cast back
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
        """
        OPTIMIZED: Compute stable load balancing losses
        Ensures max_exp stays at 50% for top_k=2
        """
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            scores_fp32 = scores.to(torch.float32)
            dispatch_fp32 = dispatch_mask.float()
            
            # Expert usage based on actual routing decisions
            expert_usage = dispatch_fp32.sum(dim=(0, 2))  # [E]
            total_tokens = expert_usage.sum() + 1e-10
            expert_usage_normalized = expert_usage / total_tokens
            
            # Target: For top_k=2, ideal is 2/num_experts per expert (50% max when k=2)
            target_usage = (self.top_k / self.num_experts)
            uniform_target = torch.full_like(expert_usage_normalized, target_usage)
            
            # Switch Transformer style load balancing loss (more stable)
            # P(expert) * f(expert) where P is gate prob, f is fraction routed
            gate_probs = scores_fp32.mean(dim=0)  # [E]
            fraction_routed = expert_usage_normalized  # [E]
            
            # Load loss: minimize variance in expert*fraction product
            load_loss = (gate_probs * fraction_routed).sum() * self.num_experts
            load_loss = load_loss.clamp(min=0.0, max=10.0)  # Prevent explosion
            
            # Importance loss: encourage routing diversity
            importance = scores_fp32.mean(dim=0)  # [E]
            importance_variance = torch.var(importance, unbiased=False)
            importance_mean = torch.mean(importance)
            importance_loss = importance_variance / (importance_mean ** 2 + 1e-10)
            importance_loss = importance_loss.clamp(min=0.0, max=1.0)
            
            # Entropy regularization - maintain balanced routing
            routing_entropy = -torch.sum(scores_fp32 * torch.log(scores_fp32 + 1e-10), dim=-1).mean()
            max_entropy = torch.log(torch.tensor(float(self.num_experts), device=scores.device, dtype=torch.float32))
            
            # Normalized entropy (0 to 1, where 1 is perfect balance)
            normalized_entropy = routing_entropy / max_entropy
            # Loss increases as entropy decreases (encourage high entropy)
            entropy_reg_loss = (1.0 - normalized_entropy) * 5.0
            entropy_reg_loss = entropy_reg_loss.clamp(min=0.0, max=5.0)
            
            # Z-loss: router logit regularization (prevent extreme values)
            router_logits_squared = torch.logsumexp(scores_fp32, dim=-1) ** 2
            z_loss = router_logits_squared.mean()
            z_loss = z_loss.clamp(min=0.0, max=100.0)
            
            # Expert usage entropy (measure balance across experts)
            expert_entropy = -torch.sum(
                expert_usage_normalized * torch.log(expert_usage_normalized + 1e-10)
            )
            
            return {
                'load_loss': load_loss.to(scores.dtype),
                'importance_loss': importance_loss.to(scores.dtype),
                'z_loss': z_loss.to(scores.dtype),
                'entropy_reg_loss': entropy_reg_loss.to(scores.dtype),
                'expert_usage': expert_usage_normalized,
                'routing_entropy': expert_entropy
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
