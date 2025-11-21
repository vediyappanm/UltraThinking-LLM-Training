"""
Megatron-LM Compatibility Layer
Provides tensor/pipeline/sequence/expert parallelism, fused ops, and distributed checkpointing
"""

from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .pipeline import PipelineEngine, PipelineConfig, PipelineSchedule
from .sequence_parallel import (
    reduce_scatter_to_sequence_parallel_region,
    all_gather_from_sequence_parallel_region,
    SequenceParallelLinear,
    SequenceParallelLayerNorm,
)
from .fused_ops import (
    get_fused_layer_norm,
    get_fused_rms_norm,
    BiasDropoutAddFusion,
    bias_dropout_add_fused,
    TransformerEngineLinear,
    FusedScaleMaskSoftmax,
    get_fp8_recipe,
    FusedAdamW,
)
from .expert_parallel import (
    ExpertParallelMoELayer,
    TopKRouter,
    compute_moe_aux_loss,
)
from .distributed_checkpoint import (
    DistributedCheckpoint,
    save_distributed_checkpoint,
    load_distributed_checkpoint,
    get_parallel_state,
)

__all__ = [
    # Tensor parallel layers
    'ColumnParallelLinear',
    'RowParallelLinear',
    'VocabParallelEmbedding',
    # Pipeline parallel
    'PipelineEngine',
    'PipelineConfig',
    'PipelineSchedule',
    # Sequence parallel
    'reduce_scatter_to_sequence_parallel_region',
    'all_gather_from_sequence_parallel_region',
    'SequenceParallelLinear',
    'SequenceParallelLayerNorm',
    # Fused ops
    'get_fused_layer_norm',
    'get_fused_rms_norm',
    'BiasDropoutAddFusion',
    'bias_dropout_add_fused',
    'TransformerEngineLinear',
    'FusedScaleMaskSoftmax',
    'get_fp8_recipe',
    'FusedAdamW',
    # Expert parallel
    'ExpertParallelMoELayer',
    'TopKRouter',
    'compute_moe_aux_loss',
    # Distributed checkpoint
    'DistributedCheckpoint',
    'save_distributed_checkpoint',
    'load_distributed_checkpoint',
    'get_parallel_state',
]
