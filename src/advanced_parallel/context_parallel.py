"""
Context Parallelism (CP) for Ultra-Long Sequences
Scales to 128K-1M tokens by splitting attention computation across GPUs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_context_parallel_group():
    """Get context parallel process group"""
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD


def get_context_parallel_world_size():
    """Get context parallel world size"""
    group = get_context_parallel_group()
    if group is None:
        return 1
    return dist.get_world_size(group)


def get_context_parallel_rank():
    """Get context parallel rank"""
    group = get_context_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


class _AllGatherAlongSeqDim(torch.autograd.Function):
    """All-gather along sequence dimension for context parallelism"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        """All-gather tokens across CP ranks"""
        if group is None:
            group = get_context_parallel_group()
        
        if group is None or dist.get_world_size(group) == 1:
            return input_tensor
        
        ctx.group = group
        world_size = dist.get_world_size(group)
        
        # Gather along sequence dimension (dim=1 for [batch, seq, hidden])
        tensor_list = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_tensor, group=group)
        
        # Concatenate along sequence
        output = torch.cat(tensor_list, dim=1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Split gradient back to local shard"""
        group = ctx.group
        
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        
        # Split gradient along sequence dimension
        grad_chunks = torch.chunk(grad_output, world_size, dim=1)
        local_grad = grad_chunks[rank].contiguous()
        
        return local_grad, None


class _ReduceScatterAlongSeqDim(torch.autograd.Function):
    """Reduce-scatter along sequence dimension"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        """Reduce-scatter: sum and split along sequence"""
        if group is None:
            group = get_context_parallel_group()
        
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
        """All-gather gradient in backward"""
        group = ctx.group
        
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        
        world_size = dist.get_world_size(group)
        
        # All-gather
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output, group=group)
        
        # Concatenate
        output = torch.cat(tensor_list, dim=1)
        return output, None


def all_gather_along_seq_dim(input_tensor, group=None):
    """All-gather tokens along sequence dimension"""
    return _AllGatherAlongSeqDim.apply(input_tensor, group)


def reduce_scatter_along_seq_dim(input_tensor, group=None):
    """Reduce-scatter along sequence dimension"""
    return _ReduceScatterAlongSeqDim.apply(input_tensor, group)


class ContextParallelAttention(nn.Module):
    """
    Context-parallel attention for ultra-long sequences
    Splits Q/K/V by sequence dimension across GPUs
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        context_parallel_size: int = 1,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_parallel_size = context_parallel_size
        self.use_flash_attn = use_flash_attn
        
        # Check for FlashAttention
        self.flash_available = False
        if use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.flash_available = True
            except ImportError:
                logger.warning("FlashAttention not available, using standard attention")
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with context parallelism
        
        Args:
            hidden_states: [batch, local_seq_len, hidden] - already partitioned by CP
            attention_mask: Optional mask
        
        Returns:
            output: [batch, local_seq_len, hidden]
        """
        batch_size, local_seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V (local sequences)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, local_seq, num_heads, head_dim]
        q = q.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        
        if self.context_parallel_size > 1:
            # All-gather K and V across context parallel ranks
            # This gives each rank the full K/V for attention
            k_full = all_gather_along_seq_dim(k)  # [batch, full_seq, num_heads, head_dim]
            v_full = all_gather_along_seq_dim(v)
            
            # Q remains local, K/V are full sequence
            # Each rank computes attention for its local Q tokens against full K/V
            if self.flash_available:
                # FlashAttention path
                attn_output = self._flash_attention_cp(q, k_full, v_full)
            else:
                # Standard attention path
                attn_output = self._standard_attention_cp(q, k_full, v_full, attention_mask)
        else:
            # No context parallelism
            if self.flash_available:
                attn_output = self._flash_attention_cp(q, k, v)
            else:
                attn_output = self._standard_attention_cp(q, k, v, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, local_seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output
    
    def _flash_attention_cp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """FlashAttention with context parallelism"""
        # FlashAttention expects [batch, seq, heads, head_dim]
        attn_output = self.flash_attn_func(
            q, k, v,
            causal=True,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
        )
        return attn_output
    
    def _standard_attention_cp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention with context parallelism"""
        # Transpose to [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back to [batch, seq, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output


class RingAttention(nn.Module):
    """
    Ring Attention for extreme context lengths (1M+ tokens)
    Overlaps communication and computation using ring-reduce pattern
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        context_parallel_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_parallel_size = context_parallel_size
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Ring attention forward pass
        Overlaps K/V communication with attention computation
        """
        batch_size, local_seq_len, _ = hidden_states.shape
        
        # Project Q, K, V locally
        q = self.q_proj(hidden_states).view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        
        if self.context_parallel_size == 1:
            # No ring, standard attention
            return self._compute_attention(q, k, v)
        
        # Ring attention: rotate K/V through ranks
        cp_rank = get_context_parallel_rank()
        cp_size = get_context_parallel_world_size()
        cp_group = get_context_parallel_group()
        
        # Initialize output accumulator
        attn_output = torch.zeros(
            batch_size, local_seq_len, self.num_heads, self.head_dim,
            dtype=q.dtype, device=q.device
        )
        
        # Current K/V buffers
        k_curr = k.clone()
        v_curr = v.clone()
        
        # Ring iterations
        for step in range(cp_size):
            # Compute attention with current K/V shard
            partial_output = self._compute_attention_partial(q, k_curr, v_curr, step, cp_rank, cp_size)
            attn_output += partial_output
            
            # Rotate K/V to next rank (except last iteration)
            if step < cp_size - 1:
                k_next = torch.empty_like(k_curr)
                v_next = torch.empty_like(v_curr)
                
                # Send to next rank, receive from previous rank
                send_rank = (cp_rank + 1) % cp_size
                recv_rank = (cp_rank - 1 + cp_size) % cp_size
                
                # Non-blocking send/recv
                send_k = dist.isend(k_curr, dst=send_rank, group=cp_group)
                send_v = dist.isend(v_curr, dst=send_rank, group=cp_group)
                recv_k = dist.irecv(k_next, src=recv_rank, group=cp_group)
                recv_v = dist.irecv(v_next, src=recv_rank, group=cp_group)
                
                # Wait for communication
                send_k.wait()
                send_v.wait()
                recv_k.wait()
                recv_v.wait()
                
                k_curr = k_next
                v_curr = v_next
        
        # Project output
        attn_output = attn_output.reshape(batch_size, local_seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output
    
    def _compute_attention(self, q, k, v):
        """Standard attention computation"""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output.transpose(1, 2)
    
    def _compute_attention_partial(self, q, k, v, step, rank, size):
        """Compute attention for one K/V shard with causal masking"""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute which sequence positions this shard covers
        local_seq_len = k.size(-2)
        shard_start = ((rank + step) % size) * local_seq_len
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask: only attend to positions <= current position
        q_positions = torch.arange(rank * local_seq_len, (rank + 1) * local_seq_len, device=q.device)
        k_positions = torch.arange(shard_start, shard_start + local_seq_len, device=k.device)
        
        causal_mask = q_positions[:, None] >= k_positions[None, :]
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output.transpose(1, 2)


def split_sequence_for_context_parallel(
    input_tensor: torch.Tensor,
    context_parallel_size: int,
) -> torch.Tensor:
    """
    Split sequence across context parallel ranks
    
    Args:
        input_tensor: [batch, seq_len, hidden]
        context_parallel_size: Number of CP ranks
    
    Returns:
        local_tensor: [batch, local_seq_len, hidden]
    """
    if context_parallel_size == 1:
        return input_tensor
    
    rank = get_context_parallel_rank()
    chunks = torch.chunk(input_tensor, context_parallel_size, dim=1)
    return chunks[rank].contiguous()
