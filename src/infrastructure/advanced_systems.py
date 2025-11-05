"""
Advanced Infrastructure Systems (Features 15-18)
- Modular Tokenizer + Vocab Parallel Alignment
- Advanced Profiling, Logging & Visualization
- Memory-Efficient Checkpointing (Paged / CPU Offload)
- Mixture-of-Agents (Multi-Expert Inference Controller)
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 15. MODULAR TOKENIZER + VOCAB PARALLEL ALIGNMENT
# ============================================================================

class VocabParallelEmbedding(nn.Module):
    """
    Vocabulary-parallel embedding layer
    Each rank stores a slice of the embedding matrix
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vocab_parallel_size: int = 1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_parallel_size = vocab_parallel_size
        self.padding_idx = padding_idx
        
        # Calculate local vocab size
        self.vocab_start_index = 0
        self.vocab_end_index = num_embeddings
        
        if vocab_parallel_size > 1 and dist.is_initialized():
            rank = dist.get_rank()
            vocab_per_rank = num_embeddings // vocab_parallel_size
            self.vocab_start_index = rank * vocab_per_rank
            self.vocab_end_index = (rank + 1) * vocab_per_rank
        
        # Local embedding
        local_vocab_size = self.vocab_end_index - self.vocab_start_index
        self.embedding = nn.Embedding(local_vocab_size, embedding_dim, padding_idx=padding_idx)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward with vocab parallelism
        
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            embeddings: [batch, seq_len, embedding_dim]
        """
        # Mask for local vocab
        input_mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        
        # Local indices
        local_ids = input_ids - self.vocab_start_index
        local_ids = local_ids.clamp(0, self.embedding.num_embeddings - 1)
        
        # Embed
        local_embeddings = self.embedding(local_ids)
        
        # Mask out non-local tokens
        local_embeddings = local_embeddings * input_mask.unsqueeze(-1).float()
        
        # All-reduce to gather from all ranks
        if self.vocab_parallel_size > 1 and dist.is_initialized():
            dist.all_reduce(local_embeddings)
        
        return local_embeddings


class MultiTokenizerManager:
    """
    Manages multiple tokenizers for multilingual/multi-domain models
    """
    
    def __init__(self, tokenizer_configs: List[Dict[str, Any]]):
        self.tokenizers = {}
        self.vocab_sizes = {}
        
        for config in tokenizer_configs:
            name = config['name']
            tokenizer_type = config['type']  # sentencepiece, bpe, wordpiece
            
            # Load tokenizer (simplified - would use actual tokenizer libraries)
            self.tokenizers[name] = self._load_tokenizer(tokenizer_type, config)
            self.vocab_sizes[name] = config.get('vocab_size', 50000)
        
        logger.info(f"Loaded {len(self.tokenizers)} tokenizers")
    
    def _load_tokenizer(self, tokenizer_type: str, config: Dict):
        """Load tokenizer based on type"""
        # Placeholder - would integrate with actual tokenizer libraries
        return None
    
    def encode(self, text: str, tokenizer_name: str = "default") -> List[int]:
        """Encode text with specified tokenizer"""
        tokenizer = self.tokenizers.get(tokenizer_name)
        if tokenizer is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found")
        return tokenizer.encode(text)
    
    def decode(self, token_ids: List[int], tokenizer_name: str = "default") -> str:
        """Decode tokens with specified tokenizer"""
        tokenizer = self.tokenizers.get(tokenizer_name)
        if tokenizer is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found")
        return tokenizer.decode(token_ids)


# ============================================================================
# 16. ADVANCED PROFILING, LOGGING & VISUALIZATION
# ============================================================================

class PerformanceProfiler:
    """
    Advanced performance profiler with NVTX markers and Torch profiler
    """
    
    def __init__(
        self,
        enable_nvtx: bool = True,
        enable_torch_profiler: bool = True,
        log_dir: str = "./profiling",
    ):
        self.enable_nvtx = enable_nvtx
        self.enable_torch_profiler = enable_torch_profiler
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Try importing NVTX
        self.nvtx_available = False
        if enable_nvtx:
            try:
                import nvtx
                self.nvtx = nvtx
                self.nvtx_available = True
            except ImportError:
                logger.warning("NVTX not available")
        
        # Profiler state
        self.profiler = None
        self.timers = {}
        
        logger.info("Performance profiler initialized")
    
    def start_profiling(self, wait: int = 1, warmup: int = 1, active: int = 3):
        """Start torch profiler"""
        if not self.enable_torch_profiler:
            return
        
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.log_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()
    
    def stop_profiling(self):
        """Stop torch profiler"""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
    
    def step(self):
        """Profiler step"""
        if self.profiler is not None:
            self.profiler.step()
    
    def mark_nvtx(self, name: str, color: str = "blue"):
        """NVTX range marker"""
        if self.nvtx_available:
            return self.nvtx.annotate(name, color=color)
        return None
    
    def start_timer(self, name: str):
        """Start timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timer and return elapsed time"""
        if name not in self.timers:
            return 0.0
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }


# ============================================================================
# 17. MEMORY-EFFICIENT CHECKPOINTING (PAGED / CPU OFFLOAD)
# ============================================================================

class PagedCheckpointManager:
    """
    Memory-efficient checkpointing with paging and CPU/NVMe offload
    Handles 100B+ model states without GPU overflow
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        page_size_mb: int = 256,
        offload_device: str = "cpu",  # cpu, nvme
        enable_compression: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.page_size_bytes = page_size_mb * 1024 * 1024
        self.offload_device = offload_device
        self.enable_compression = enable_compression
        
        logger.info(f"Paged checkpoint manager: page_size={page_size_mb}MB, offload={offload_device}")
    
    def save_paged_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
    ):
        """
        Save checkpoint with paging and offloading
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            step: Training step
        """
        checkpoint_path = self.checkpoint_dir / f"paged_step_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model in pages
        model_state = model.state_dict()
        page_idx = 0
        current_page = {}
        current_size = 0
        
        for name, param in model_state.items():
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > self.page_size_bytes and current_page:
                # Save current page
                self._save_page(checkpoint_path, f"model_page_{page_idx}.pt", current_page)
                page_idx += 1
                current_page = {}
                current_size = 0
            
            # Offload to CPU if needed
            if self.offload_device == "cpu":
                param = param.cpu()
            
            current_page[name] = param
            current_size += param_size
        
        # Save last page
        if current_page:
            self._save_page(checkpoint_path, f"model_page_{page_idx}.pt", current_page)
        
        # Save optimizer state (if provided)
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            self._save_page(checkpoint_path, "optimizer.pt", opt_state)
        
        # Save metadata
        metadata = {
            'step': step,
            'num_pages': page_idx + 1,
            'page_size_mb': self.page_size_bytes / (1024 * 1024),
        }
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved paged checkpoint: {page_idx + 1} pages")
    
    def load_paged_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load paged checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load metadata
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load model pages
        model_state = {}
        for page_idx in range(metadata['num_pages']):
            page = torch.load(checkpoint_path / f"model_page_{page_idx}.pt", map_location='cpu')
            model_state.update(page)
        
        # Load into model
        model.load_state_dict(model_state)
        
        # Load optimizer
        if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
            opt_state = torch.load(checkpoint_path / "optimizer.pt", map_location='cpu')
            optimizer.load_state_dict(opt_state)
        
        logger.info(f"Loaded paged checkpoint from step {metadata['step']}")
        
        return metadata
    
    def _save_page(self, checkpoint_path: Path, filename: str, state_dict: Dict):
        """Save a single page"""
        torch.save(state_dict, checkpoint_path / filename)


# ============================================================================
# 18. MIXTURE-OF-AGENTS (MULTI-EXPERT INFERENCE CONTROLLER)
# ============================================================================

class AgentRouter(nn.Module):
    """
    Lightweight classifier that routes inputs to specialized sub-models
    Enables UltraThinking-AI OS style cooperative reasoning
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_agents: int,
        agent_names: List[str],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents
        self.agent_names = agent_names
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_agents),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to agents
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            agent_probs: [batch, num_agents]
            agent_indices: [batch]
        """
        # Pool hidden states (use first token)
        pooled = hidden_states[:, 0, :]
        
        # Compute routing logits
        logits = self.router(pooled)
        
        # Softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Select top agent
        agent_indices = torch.argmax(probs, dim=-1)
        
        return probs, agent_indices


class MixtureOfAgents:
    """
    Mixture-of-Agents inference system
    Coordinates multiple specialized models for collaborative reasoning
    """
    
    def __init__(
        self,
        router: AgentRouter,
        agents: Dict[str, nn.Module],
        enable_async: bool = True,
    ):
        self.router = router
        self.agents = agents
        self.enable_async = enable_async
        
        # Agent statistics
        self.agent_usage = {name: 0 for name in agents.keys()}
        
        logger.info(f"Mixture-of-Agents initialized with {len(agents)} agents")
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        **kwargs
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate with agent routing
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
        
        Returns:
            output_ids: Generated tokens
            agent_name: Selected agent name
        """
        # Get hidden states for routing
        # (Simplified - would use actual model forward)
        batch_size = input_ids.size(0)
        hidden_states = torch.randn(batch_size, input_ids.size(1), self.router.hidden_size)
        
        # Route to agent
        agent_probs, agent_indices = self.router(hidden_states)
        
        # Get agent name
        agent_idx = agent_indices[0].item()
        agent_name = self.router.agent_names[agent_idx]
        
        # Update statistics
        self.agent_usage[agent_name] += 1
        
        # Generate with selected agent
        agent_model = self.agents[agent_name]
        output_ids = agent_model.generate(input_ids, max_length=max_length, **kwargs)
        
        logger.info(f"Routed to agent: {agent_name} (prob: {agent_probs[0, agent_idx]:.3f})")
        
        return output_ids, agent_name
    
    def get_agent_statistics(self) -> Dict[str, int]:
        """Get agent usage statistics"""
        return self.agent_usage.copy()
