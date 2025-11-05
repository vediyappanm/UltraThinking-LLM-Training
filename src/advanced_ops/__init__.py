"""
Advanced Operations for Production LLM Training
Implements 6 cutting-edge optimizations for efficiency and scale
"""

# FlashAttention v3 + Variants
from .attention_ops import (
    FlashAttentionV3,
    MultiQueryAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
    MambaBlock,
    HybridAttentionMamba,
    create_attention_layer,
)

# Unified Communication Layer
from .comm_utils import (
    UnifiedCommLayer,
    AsyncCommHandle,
    FusedGradientReducer,
    OverlappedAllReduce,
    AsyncGradientAccumulator,
    create_comm_layer,
)

# Position Encodings
from .position_encoding import (
    RotaryEmbedding,
    NTKScaledRotaryEmbedding,
    DynamicNTKScalingRoPE,
    ContinuousRoPE,
    apply_rotary_pos_emb,
    rotate_half,
    create_position_encoding,
)

# Fused Optimizers + Gradient Accumulation
from .gradient_accumulator import (
    FusedAdamW,
    AsyncGradientAccumulator as AsyncGradAccum,
    FusedGradientReduceOptimizer,
    GradientCompressionOptimizer,
    create_fused_optimizer,
)

# PEFT (LoRA, Adapters)
from .adapter_layers import (
    LoRALayer,
    LoRALinear,
    QLoRALinear,
    AdapterLayer,
    PrefixTuning,
    apply_lora_to_model,
    apply_adapters_to_model,
    merge_lora_weights,
    get_trainable_parameters,
)

# Compression
from .compression_utils import (
    GradientCompressor,
    ActivationCompressor,
    CompressedLinear,
    apply_gradient_compression,
    apply_activation_compression,
)

__all__ = [
    # Attention
    'FlashAttentionV3',
    'MultiQueryAttention',
    'GroupedQueryAttention',
    'SlidingWindowAttention',
    'MambaBlock',
    'HybridAttentionMamba',
    'create_attention_layer',
    
    # Communication
    'UnifiedCommLayer',
    'AsyncCommHandle',
    'FusedGradientReducer',
    'OverlappedAllReduce',
    'AsyncGradientAccumulator',
    'create_comm_layer',
    
    # Position Encoding
    'RotaryEmbedding',
    'NTKScaledRotaryEmbedding',
    'DynamicNTKScalingRoPE',
    'ContinuousRoPE',
    'apply_rotary_pos_emb',
    'rotate_half',
    'create_position_encoding',
    
    # Optimizers
    'FusedAdamW',
    'AsyncGradAccum',
    'FusedGradientReduceOptimizer',
    'GradientCompressionOptimizer',
    'create_fused_optimizer',
    
    # PEFT
    'LoRALayer',
    'LoRALinear',
    'QLoRALinear',
    'AdapterLayer',
    'PrefixTuning',
    'apply_lora_to_model',
    'apply_adapters_to_model',
    'merge_lora_weights',
    'get_trainable_parameters',
    
    # Compression
    'GradientCompressor',
    'ActivationCompressor',
    'CompressedLinear',
    'apply_gradient_compression',
    'apply_activation_compression',
]
