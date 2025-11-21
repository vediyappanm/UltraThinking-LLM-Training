"""
Parameter-Efficient Fine-Tuning (PEFT) - LoRA, Q-LoRA, Adapters
Enable modular fine-tuning on massive base models efficiently
100x cheaper domain adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer
    Adds trainable low-rank matrices to frozen weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA
        
        Args:
            x: Input tensor [batch, seq, in_features]
        
        Returns:
            output: LoRA output [batch, seq, out_features]
        """
        # x @ A^T @ B^T * scaling
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    Combines frozen base weights with trainable LoRA
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA adaptation
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Merged state
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA"""
        if self.merged:
            # Weights already merged
            return self.base_layer(x)
        else:
            # Base + LoRA
            return self.base_layer(x) + self.lora(x)
    
    def merge(self):
        """Merge LoRA weights into base layer"""
        if self.merged:
            return
        
        # Compute LoRA weight: B @ A * scaling
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        
        # Add to base weight
        self.base_layer.weight.data += lora_weight
        
        self.merged = True
        logger.info("LoRA weights merged into base layer")
    
    def unmerge(self):
        """Unmerge LoRA weights from base layer"""
        if not self.merged:
            return
        
        # Subtract LoRA weight
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        self.base_layer.weight.data -= lora_weight
        
        self.merged = False


class QLoRALinear(nn.Module):
    """
    Quantized LoRA (Q-LoRA)
    Uses 4-bit quantized base weights with LoRA adaptation
    Enables fine-tuning of 65B+ models on single GPU
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        quantize_bits: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.quantize_bits = quantize_bits
        
        # Quantized base weight (frozen)
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, dtype=torch.int8))
        
        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to 4-bit"""
        # Per-channel quantization
        min_val = weight.min(dim=1, keepdim=True)[0]
        max_val = weight.max(dim=1, keepdim=True)[0]
        
        # Compute scale and zero point
        scale = (max_val - min_val) / (2 ** self.quantize_bits - 1)
        zero_point = -min_val / scale
        
        # Quantize
        weight_q = ((weight / scale) + zero_point).round().clamp(0, 2 ** self.quantize_bits - 1).to(torch.int8)
        
        # Store
        self.weight_quantized.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
        self.weight_zero_point.copy_(zero_point.squeeze().to(torch.int8))
    
    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight for computation"""
        weight_dq = (self.weight_quantized.float() - self.weight_zero_point.float().unsqueeze(1)) * self.weight_scale.unsqueeze(1)
        return weight_dq
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with Q-LoRA"""
        # Dequantize base weight
        weight = self.dequantize_weight()
        
        # Base computation
        output = F.linear(x, weight)
        
        # Add LoRA
        output = output + self.lora(x)
        
        return output


class AdapterLayer(nn.Module):
    """
    Adapter layer (bottleneck architecture)
    Adds small trainable modules between frozen layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to near-identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through adapter"""
        # Residual connection
        residual = x
        
        # Adapter
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        return residual + x


class PrefixTuning(nn.Module):
    """
    Prefix Tuning
    Adds trainable prefix tokens to input
    """
    
    def __init__(
        self,
        num_prefix_tokens: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Prefix embeddings for each layer
        self.prefix_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_prefix_tokens, hidden_size))
            for _ in range(num_layers)
        ])
        
        # Initialize
        for emb in self.prefix_embeddings:
            nn.init.normal_(emb, std=0.02)
    
    def get_prefix(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """Get prefix for a specific layer"""
        prefix = self.prefix_embeddings[layer_idx]
        return prefix.unsqueeze(0).expand(batch_size, -1, -1)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj"],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to specific modules in model
    
    Args:
        model: Base model
        target_modules: Names of modules to apply LoRA
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
    
    Returns:
        model: Model with LoRA applied
    """
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA linear
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                
                # Set in model
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_linear)
                else:
                    setattr(model, child_name, lora_linear)
                
                logger.info(f"Applied LoRA to {name}")
    
    return model


def apply_adapters_to_model(
    model: nn.Module,
    adapter_size: int = 64,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> nn.Module:
    """
    Apply adapter layers to model
    
    Args:
        model: Base model
        adapter_size: Adapter bottleneck size
        activation: Activation function
        dropout: Dropout probability
    
    Returns:
        model: Model with adapters
    """
    # Find transformer blocks
    for name, module in model.named_modules():
        if 'block' in name.lower() or 'layer' in name.lower():
            if hasattr(module, 'mlp') or hasattr(module, 'feed_forward'):
                # Add adapter after MLP
                hidden_size = getattr(module, 'hidden_size', 768)
                adapter = AdapterLayer(hidden_size, adapter_size, activation, dropout)
                
                # Attach to module
                module.adapter = adapter
                logger.info(f"Added adapter to {name}")
    
    return model


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base model"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    
    logger.info("All LoRA weights merged")


def get_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Get count of trainable parameters
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_pct': 100.0 * trainable_params / total_params if total_params > 0 else 0,
    }
