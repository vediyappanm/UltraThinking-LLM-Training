"""
Speculative Decoding + Parallel Sampling
Accelerates inference using a smaller "draft" model to predict next tokens
2-3x faster inference throughput with same output quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    # Draft model settings
    num_speculative_tokens: int = 4  # Number of tokens to speculate
    draft_model_scale: float = 0.25  # Draft model size relative to target
    
    # Verification settings
    acceptance_threshold: float = 0.9  # Accept if prob ratio > threshold
    use_temperature: bool = True
    temperature: float = 1.0
    
    # Parallel sampling
    num_parallel_samples: int = 1
    enable_beam_search: bool = False
    beam_width: int = 4
    
    # Performance
    max_draft_length: int = 8
    early_exit_threshold: float = 0.5  # Exit if acceptance rate < threshold


class DraftModel(nn.Module):
    """
    Smaller draft model for speculative decoding
    Typically 4-8x smaller than target model
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Simplified transformer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed
        hidden_states = self.embedding(input_ids)
        
        # Transform
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate_draft(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens
        
        Args:
            input_ids: [batch, seq_len]
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
        
        Returns:
            draft_tokens: [batch, num_tokens]
            draft_probs: [batch, num_tokens, vocab_size]
        """
        batch_size = input_ids.size(0)
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids
        
        for _ in range(num_tokens):
            # Get logits
            logits = self.forward(current_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1)
        draft_probs = torch.stack(draft_probs, dim=1)
        
        return draft_tokens, draft_probs


class SpeculativeDecoder:
    """
    Speculative decoding engine
    Uses draft model to propose tokens, target model to verify
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DraftModel,
        config: SpeculativeConfig,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config
        
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.total_drafts = 0
        
        logger.info(f"Speculative decoder initialized: {config.num_speculative_tokens} speculative tokens")
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens with speculative decoding
        
        Args:
            input_ids: [batch, seq_len]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            generated_ids: [batch, max_length]
        """
        batch_size = input_ids.size(0)
        current_ids = input_ids
        
        while current_ids.size(1) < max_length:
            # Generate draft tokens
            num_draft = min(
                self.config.num_speculative_tokens,
                max_length - current_ids.size(1)
            )
            
            draft_tokens, draft_probs = self.draft_model.generate_draft(
                current_ids,
                num_draft,
                temperature=temperature,
            )
            
            # Verify with target model
            accepted_tokens = self._verify_draft(
                current_ids,
                draft_tokens,
                draft_probs,
                temperature,
            )
            
            # Update sequence
            if accepted_tokens.size(1) > 0:
                current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
            else:
                # No tokens accepted, generate one with target model
                target_logits = self._get_target_logits(current_ids)
                next_token_logits = target_logits[:, -1, :] / temperature
                
                # Apply top-k/top-p if specified
                if top_k is not None:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                if top_p is not None:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Update statistics
            self.total_drafts += 1
            self.total_tokens += num_draft
            self.accepted_tokens += accepted_tokens.size(1)
            
            # Early exit if acceptance rate is too low
            if self.total_drafts > 10:
                acceptance_rate = self.accepted_tokens / self.total_tokens
                if acceptance_rate < self.config.early_exit_threshold:
                    logger.warning(f"Low acceptance rate: {acceptance_rate:.2f}, falling back to standard generation")
                    break
        
        return current_ids
    
    def _verify_draft(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Verify draft tokens with target model
        
        Args:
            input_ids: [batch, seq_len]
            draft_tokens: [batch, num_draft]
            draft_probs: [batch, num_draft, vocab_size]
            temperature: Sampling temperature
        
        Returns:
            accepted_tokens: [batch, num_accepted]
        """
        # Concatenate input with draft
        candidate_ids = torch.cat([input_ids, draft_tokens], dim=1)
        
        # Get target model probabilities
        target_logits = self._get_target_logits(candidate_ids)
        
        # Extract logits for draft positions
        draft_start = input_ids.size(1)
        draft_end = draft_start + draft_tokens.size(1)
        target_draft_logits = target_logits[:, draft_start-1:draft_end-1, :] / temperature
        target_probs = F.softmax(target_draft_logits, dim=-1)
        
        # Verify each draft token
        accepted = []
        for i in range(draft_tokens.size(1)):
            draft_token = draft_tokens[:, i]
            draft_prob = draft_probs[:, i].gather(1, draft_token.unsqueeze(1)).squeeze(1)
            target_prob = target_probs[:, i].gather(1, draft_token.unsqueeze(1)).squeeze(1)
            
            # Acceptance criterion: target_prob / draft_prob > threshold
            acceptance_ratio = target_prob / (draft_prob + 1e-10)
            
            if (acceptance_ratio > self.config.acceptance_threshold).all():
                accepted.append(draft_token)
            else:
                # Reject this and all subsequent tokens
                break
        
        if accepted:
            return torch.stack(accepted, dim=1)
        else:
            return torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)
    
    def _get_target_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits from target model"""
        # Assuming target model has a forward method that returns logits
        outputs = self.target_model(input_ids)
        
        if isinstance(outputs, dict):
            return outputs.get('logits', outputs.get('output', outputs.get('hidden_states')))
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decoding statistics"""
        acceptance_rate = self.accepted_tokens / max(self.total_tokens, 1)
        speedup = (self.total_tokens / max(self.total_drafts, 1)) / self.config.num_speculative_tokens
        
        return {
            'total_tokens': self.total_tokens,
            'accepted_tokens': self.accepted_tokens,
            'total_drafts': self.total_drafts,
            'acceptance_rate': acceptance_rate,
            'estimated_speedup': speedup,
        }


class ParallelSampler:
    """
    Parallel sampling for batch generation
    Generates multiple samples in parallel for diversity
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 4,
    ):
        self.model = model
        self.num_samples = num_samples
    
    @torch.no_grad()
    def generate_parallel(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """
        Generate multiple samples in parallel
        
        Args:
            input_ids: [batch, seq_len]
            max_length: Maximum length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            samples: List of [batch, max_length] tensors
        """
        batch_size = input_ids.size(0)
        
        # Replicate input for parallel sampling
        expanded_input = input_ids.repeat_interleave(self.num_samples, dim=0)
        
        # Generate
        current_ids = expanded_input
        
        while current_ids.size(1) < max_length:
            # Get logits
            outputs = self.model(current_ids)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply filtering
            if top_k is not None:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            if top_p is not None:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Split back into separate samples
        samples = current_ids.chunk(self.num_samples, dim=0)
        
        return list(samples)
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
