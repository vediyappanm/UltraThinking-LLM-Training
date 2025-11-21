"""
Advanced Parallel Features for Ultra-Scale Training and Inference
Implements 6 cutting-edge techniques for production LLM systems
"""

# Context Parallelism (1M+ token contexts)
from .context_parallel import (
    ContextParallelAttention,
    RingAttention,
    all_gather_along_seq_dim,
    reduce_scatter_along_seq_dim,
    split_sequence_for_context_parallel,
)

# Hybrid 3D Parallel Scheduler
from .hybrid_parallel_engine import (
    HybridParallelEngine,
    HybridParallelConfig,
    ProcessGroupManager,
    ParallelMode,
)

# Distributed KV Cache
from .kv_cache_manager import (
    DistributedKVCache,
    KVCacheConfig,
    FlashDecodingAttention,
    PagedKVCache,
)

# Selective Activation Recomputation
from .activation_manager import (
    ActivationMemoryManager,
    ActivationConfig,
    RecomputePolicy,
    selective_checkpoint,
    SelectiveCheckpointWrapper,
    apply_selective_checkpointing,
)

# Adaptive MoE
from .adaptive_moe import (
    AdaptiveMoELayer,
    AdaptiveMoEConfig,
    AdaptiveRouter,
    DynamicExpertPool,
)

# Speculative Decoding
from .speculative_decoder import (
    SpeculativeDecoder,
    SpeculativeConfig,
    DraftModel,
    ParallelSampler,
)

__all__ = [
    # Context Parallelism
    'ContextParallelAttention',
    'RingAttention',
    'all_gather_along_seq_dim',
    'reduce_scatter_along_seq_dim',
    'split_sequence_for_context_parallel',
    
    # Hybrid Parallel
    'HybridParallelEngine',
    'HybridParallelConfig',
    'ProcessGroupManager',
    'ParallelMode',
    
    # KV Cache
    'DistributedKVCache',
    'KVCacheConfig',
    'FlashDecodingAttention',
    'PagedKVCache',
    
    # Activation Management
    'ActivationMemoryManager',
    'ActivationConfig',
    'RecomputePolicy',
    'selective_checkpoint',
    'SelectiveCheckpointWrapper',
    'apply_selective_checkpointing',
    
    # Adaptive MoE
    'AdaptiveMoELayer',
    'AdaptiveMoEConfig',
    'AdaptiveRouter',
    'DynamicExpertPool',
    
    # Speculative Decoding
    'SpeculativeDecoder',
    'SpeculativeConfig',
    'DraftModel',
    'ParallelSampler',
]
