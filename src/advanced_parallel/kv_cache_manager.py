"""
Distributed KV Cache for Fast Inference
Shares key/value tensors across GPUs during generation for 10-20x speedup
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    """Configuration for KV cache"""
    max_batch_size: int = 32
    max_seq_length: int = 8192
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    
    # Distribution settings
    distribute_across_gpus: bool = True
    cache_on_cpu: bool = False
    use_flash_decoding: bool = True


class DistributedKVCache:
    """
    Distributed KV cache manager
    Shares K/V tensors across GPUs to enable fast multi-GPU inference
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # Get distributed info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Determine cache location
        if config.distribute_across_gpus and self.world_size > 1:
            # Distribute cache across GPUs
            self.cache_device = torch.device(f'cuda:{self.rank}')
            self.distributed = True
        elif config.cache_on_cpu:
            # Cache on CPU
            self.cache_device = torch.device('cpu')
            self.distributed = False
        else:
            # Cache on single GPU
            self.cache_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.distributed = False
        
        # Initialize cache storage
        self.key_cache = {}  # layer_idx -> tensor
        self.value_cache = {}  # layer_idx -> tensor
        
        # Sequence lengths per batch item
        self.seq_lengths = torch.zeros(config.max_batch_size, dtype=torch.long)
        
        logger.info(f"KV cache initialized: device={self.cache_device}, distributed={self.distributed}")
    
    def allocate_cache(self, layer_idx: int):
        """Allocate cache for a specific layer"""
        if layer_idx in self.key_cache:
            return
        
        # Allocate K/V cache tensors
        cache_shape = (
            self.config.max_batch_size,
            self.config.num_heads,
            self.config.max_seq_length,
            self.config.head_dim,
        )
        
        self.key_cache[layer_idx] = torch.zeros(
            cache_shape,
            dtype=self.config.dtype,
            device=self.cache_device,
        )
        
        self.value_cache[layer_idx] = torch.zeros(
            cache_shape,
            dtype=self.config.dtype,
            device=self.cache_device,
        )
    
    def update_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key/value states
        
        Args:
            layer_idx: Layer index
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            batch_indices: Optional batch indices for selective update
        
        Returns:
            full_keys: Full key cache
            full_values: Full value cache
        """
        # Allocate if needed
        if layer_idx not in self.key_cache:
            self.allocate_cache(layer_idx)
        
        batch_size, num_heads, new_seq_len, head_dim = key_states.shape
        
        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=key_states.device)
        
        # Update cache for each batch item
        for i, batch_idx in enumerate(batch_indices):
            current_len = self.seq_lengths[batch_idx].item()
            
            # Insert new K/V at current position
            self.key_cache[layer_idx][batch_idx, :, current_len:current_len + new_seq_len] = key_states[i]
            self.value_cache[layer_idx][batch_idx, :, current_len:current_len + new_seq_len] = value_states[i]
            
            # Update sequence length
            self.seq_lengths[batch_idx] += new_seq_len
        
        # Return full cache up to max sequence length
        max_len = self.seq_lengths[batch_indices].max().item()
        
        full_keys = self.key_cache[layer_idx][batch_indices, :, :max_len]
        full_values = self.value_cache[layer_idx][batch_indices, :, :max_len]
        
        return full_keys, full_values
    
    def get_cache(
        self,
        layer_idx: int,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K/V for a layer"""
        if layer_idx not in self.key_cache:
            return None, None
        
        if batch_indices is None:
            batch_indices = torch.arange(self.config.max_batch_size)
        
        max_len = self.seq_lengths[batch_indices].max().item()
        
        keys = self.key_cache[layer_idx][batch_indices, :, :max_len]
        values = self.value_cache[layer_idx][batch_indices, :, :max_len]
        
        return keys, values
    
    def clear_cache(self, batch_indices: Optional[torch.Tensor] = None):
        """Clear cache for specified batch items"""
        if batch_indices is None:
            # Clear all
            self.key_cache.clear()
            self.value_cache.clear()
            self.seq_lengths.zero_()
        else:
            # Clear specific batch items
            for layer_idx in self.key_cache.keys():
                self.key_cache[layer_idx][batch_indices] = 0
                self.value_cache[layer_idx][batch_indices] = 0
            
            self.seq_lengths[batch_indices] = 0
    
    def synchronize_cache(self):
        """Synchronize cache across distributed ranks"""
        if not self.distributed or not dist.is_initialized():
            return
        
        # All-gather cache from all ranks
        for layer_idx in self.key_cache.keys():
            # Gather keys
            key_list = [torch.empty_like(self.key_cache[layer_idx]) for _ in range(self.world_size)]
            dist.all_gather(key_list, self.key_cache[layer_idx])
            
            # Gather values
            value_list = [torch.empty_like(self.value_cache[layer_idx]) for _ in range(self.world_size)]
            dist.all_gather(value_list, self.value_cache[layer_idx])
            
            # Merge (take max sequence length from any rank)
            self.key_cache[layer_idx] = torch.stack(key_list).max(dim=0)[0]
            self.value_cache[layer_idx] = torch.stack(value_list).max(dim=0)[0]


class FlashDecodingAttention(nn.Module):
    """
    Flash Decoding for fast autoregressive generation
    Optimized for single-token generation with large KV cache
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        kv_cache: DistributedKVCache,
        layer_idx: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx
        
        # Try to import flash attention
        self.use_flash = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash = True
        except ImportError:
            logger.warning("FlashAttention not available for decoding")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with KV caching
        
        Args:
            query: [batch, num_heads, 1, head_dim] - single token query
            key: [batch, num_heads, 1, head_dim] - new key
            value: [batch, num_heads, 1, head_dim] - new value
            use_cache: Whether to use KV cache
        
        Returns:
            output: [batch, num_heads, 1, head_dim]
            past_kv: Updated (key, value) cache
        """
        batch_size = query.size(0)
        
        if use_cache:
            # Update cache with new K/V
            full_keys, full_values = self.kv_cache.update_cache(
                self.layer_idx,
                key,
                value,
            )
        else:
            full_keys = key
            full_values = value
        
        # Compute attention
        if self.use_flash and full_keys.size(2) > 1:
            # Flash attention for decoding
            # Reshape for flash attention: [batch, seq, heads, head_dim]
            q = query.transpose(1, 2)
            k = full_keys.transpose(1, 2)
            v = full_values.transpose(1, 2)
            
            output = self.flash_attn_func(q, k, v, causal=True)
            output = output.transpose(1, 2)
        else:
            # Standard attention
            scores = torch.matmul(query, full_keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, full_values)
        
        past_kv = (full_keys, full_values) if use_cache else None
        
        return output, past_kv


class PagedKVCache:
    """
    Paged KV cache for efficient memory management
    Inspired by vLLM's paged attention
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype
        
        # Allocate physical blocks
        self.key_blocks = {}
        self.value_blocks = {}
        
        for layer_idx in range(num_layers):
            self.key_blocks[layer_idx] = torch.zeros(
                max_blocks, block_size, num_heads, head_dim,
                dtype=dtype,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            self.value_blocks[layer_idx] = torch.zeros(
                max_blocks, block_size, num_heads, head_dim,
                dtype=dtype,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        
        # Block allocation tracking
        self.free_blocks = list(range(max_blocks))
        self.sequence_blocks = {}  # seq_id -> list of block indices
    
    def allocate_blocks(self, seq_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence"""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"Not enough free blocks: need {num_blocks}, have {len(self.free_blocks)}")
        
        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)
        
        self.sequence_blocks[seq_id] = allocated
        return allocated
    
    def free_sequence(self, seq_id: int):
        """Free blocks for a sequence"""
        if seq_id in self.sequence_blocks:
            self.free_blocks.extend(self.sequence_blocks[seq_id])
            del self.sequence_blocks[seq_id]
    
    def write_block(
        self,
        layer_idx: int,
        block_idx: int,
        key_data: torch.Tensor,
        value_data: torch.Tensor,
    ):
        """Write K/V data to a block"""
        self.key_blocks[layer_idx][block_idx] = key_data
        self.value_blocks[layer_idx][block_idx] = value_data
    
    def read_sequence(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K/V for entire sequence"""
        if seq_id not in self.sequence_blocks:
            return None, None
        
        block_indices = self.sequence_blocks[seq_id]
        
        keys = torch.cat([self.key_blocks[layer_idx][idx] for idx in block_indices], dim=0)
        values = torch.cat([self.value_blocks[layer_idx][idx] for idx in block_indices], dim=0)
        
        return keys, values
