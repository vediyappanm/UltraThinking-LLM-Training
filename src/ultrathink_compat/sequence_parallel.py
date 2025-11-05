"""
Sequence Parallelism for Megatron-LM
Partition activations along sequence dimension to reduce memory
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_tensor_parallel_group():
    """Get tensor parallel process group"""
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD


def get_tensor_parallel_world_size():
    """Get tensor parallel world size"""
    group = get_tensor_parallel_group()
    if group is None:
        return 1
    return dist.get_world_size(group)


def get_tensor_parallel_rank():
    """Get tensor parallel rank"""
    group = get_tensor_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """Reduce-scatter along sequence dimension in forward, all-gather in backward"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        """Reduce-scatter: sum across TP group and scatter along sequence"""
        if group is None:
            group = get_tensor_parallel_group()
        
        if group is None or dist.get_world_size(group) == 1:
            return input_tensor
        
        ctx.group = group
        world_size = dist.get_world_size(group)
        
        # Split input along sequence dimension
        input_list = list(torch.chunk(input_tensor, world_size, dim=1))
        
        # Reduce-scatter
        output = torch.empty_like(input_list[0])
        dist.reduce_scatter(output, input_list, group=group)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """All-gather in backward"""
        group = ctx.group
        
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        
        world_size = dist.get_world_size(group)
        
        # All-gather
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output, group=group)
        
        # Concatenate along sequence dimension
        output = torch.cat(tensor_list, dim=1)
        
        return output, None


class _AllGatherFromSequenceParallel(torch.autograd.Function):
    """All-gather along sequence dimension in forward, reduce-scatter in backward"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        """All-gather along sequence dimension"""
        if group is None:
            group = get_tensor_parallel_group()
        
        if group is None or dist.get_world_size(group) == 1:
            return input_tensor
        
        ctx.group = group
        world_size = dist.get_world_size(group)
        
        # All-gather
        tensor_list = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_tensor, group=group)
        
        # Concatenate along sequence dimension
        output = torch.cat(tensor_list, dim=1)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Reduce-scatter in backward"""
        group = ctx.group
        
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        
        world_size = dist.get_world_size(group)
        
        # Split gradient along sequence dimension
        grad_list = list(torch.chunk(grad_output, world_size, dim=1))
        
        # Reduce-scatter
        output = torch.empty_like(grad_list[0])
        dist.reduce_scatter(output, grad_list, group=group)
        
        return output, None


def reduce_scatter_to_sequence_parallel_region(input_tensor, group=None):
    """Reduce-scatter to sequence parallel region"""
    return _ReduceScatterToSequenceParallel.apply(input_tensor, group)


def all_gather_from_sequence_parallel_region(input_tensor, group=None):
    """All-gather from sequence parallel region"""
    return _AllGatherFromSequenceParallel.apply(input_tensor, group)


class SequenceParallelLinear(nn.Module):
    """Linear layer with sequence parallelism support"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sequence_parallel: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sequence_parallel = sequence_parallel
        self.gather_output = gather_output
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_tensor):
        """Forward with optional sequence parallelism"""
        # If input is sequence parallel, gather before matmul
        if self.sequence_parallel:
            input_tensor = all_gather_from_sequence_parallel_region(input_tensor)
        
        # Linear transformation
        output = torch.matmul(input_tensor, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        
        # If output should be sequence parallel, scatter
        if self.sequence_parallel and not self.gather_output:
            output = reduce_scatter_to_sequence_parallel_region(output)
        
        return output


class SequenceParallelLayerNorm(nn.Module):
    """LayerNorm with sequence parallelism support"""
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.sequence_parallel = sequence_parallel
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, input_tensor):
        """LayerNorm with sequence parallel input"""
        if self.sequence_parallel:
            # Input is partitioned along sequence dimension
            # LayerNorm operates independently on each partition
            pass
        
        # Standard LayerNorm
        return nn.functional.layer_norm(
            input_tensor,
            (self.normalized_shape,),
            self.weight,
            self.bias,
            self.eps
        )
