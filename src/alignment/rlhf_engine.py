"""
Reinforcement Learning from Human Feedback (RLHF / PPO Engine)
Train UltraThinking to align with user preferences like ChatGPT or Gemini
Transforms raw model into helpful, safe conversational AI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    # PPO parameters
    ppo_epochs: int = 4
    ppo_steps: int = 128
    ppo_batch_size: int = 32
    mini_batch_size: int = 8
    
    # KL divergence
    kl_coeff: float = 0.1
    target_kl: float = 0.01
    adaptive_kl: bool = True
    
    # Clipping
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    
    # Optimization
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Reward
    reward_scale: float = 1.0
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter


class RewardModel(nn.Module):
    """
    Reward model for RLHF
    Predicts scalar reward for (prompt, response) pairs
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 4096,
    ):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            rewards: [batch, 1]
        """
        # Get base model output
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        # Extract last hidden state
        if isinstance(outputs, dict):
            hidden_states = outputs.get('hidden_states', outputs.get('last_hidden_state'))
        else:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Pool (use last token)
        pooled = hidden_states[:, -1, :]
        
        # Compute reward
        reward = self.reward_head(pooled)
        
        return reward


class ValueModel(nn.Module):
    """
    Value model for PPO
    Estimates state value for advantage computation
    """
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute value
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            values: [batch, seq_len, 1]
        """
        return self.value_head(hidden_states)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHF
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        policy_model: nn.Module,
        reference_model: nn.Module,
        reward_model: RewardModel,
        value_model: ValueModel,
    ):
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.value_model = value_model
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(),
            lr=config.value_learning_rate,
        )
        
        # KL coefficient (adaptive)
        self.kl_coeff = config.kl_coeff
        
        logger.info("PPO trainer initialized")
    
    def train_step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single PPO training step
        
        Args:
            prompts: Prompt token IDs [batch, prompt_len]
            responses: Response token IDs [batch, response_len]
            attention_mask: Attention mask [batch, total_len]
        
        Returns:
            metrics: Training metrics
        """
        # Concatenate prompts and responses
        input_ids = torch.cat([prompts, responses], dim=1)
        
        # Get rewards
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        
        # Get old log probs and values
        with torch.no_grad():
            old_logprobs, old_values = self._compute_logprobs_and_values(
                input_ids, attention_mask
            )
            ref_logprobs = self._compute_reference_logprobs(input_ids, attention_mask)
        
        # Compute advantages
        advantages, returns = self._compute_advantages(rewards, old_values)
        
        # PPO epochs
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch training
            for mb_idx in range(0, len(input_ids), self.config.mini_batch_size):
                mb_end = min(mb_idx + self.config.mini_batch_size, len(input_ids))
                
                # Get mini-batch
                mb_input_ids = input_ids[mb_idx:mb_end]
                mb_attention_mask = attention_mask[mb_idx:mb_end]
                mb_old_logprobs = old_logprobs[mb_idx:mb_end]
                mb_advantages = advantages[mb_idx:mb_end]
                mb_returns = returns[mb_idx:mb_end]
                mb_ref_logprobs = ref_logprobs[mb_idx:mb_end]
                
                # Compute new log probs and values
                new_logprobs, new_values = self._compute_logprobs_and_values(
                    mb_input_ids, mb_attention_mask
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)
                
                # KL divergence penalty
                kl_div = (new_logprobs - mb_ref_logprobs).mean()
                kl_penalty = self.kl_coeff * kl_div
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + kl_penalty
                
                # Backward
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Update
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Metrics
                metrics['policy_loss'] = policy_loss.item()
                metrics['value_loss'] = value_loss.item()
                metrics['kl_div'] = kl_div.item()
                metrics['total_loss'] = loss.item()
        
        # Adaptive KL coefficient
        if self.config.adaptive_kl:
            if kl_div > self.config.target_kl * 1.5:
                self.kl_coeff *= 1.5
            elif kl_div < self.config.target_kl / 1.5:
                self.kl_coeff /= 1.5
            metrics['kl_coeff'] = self.kl_coeff
        
        return metrics
    
    def _compute_logprobs_and_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and values"""
        # Forward through policy model
        outputs = self.policy_model(input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits')
            hidden_states = outputs.get('hidden_states')
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            hidden_states = None
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, 2, input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute values
        if hidden_states is not None:
            values = self.value_model(hidden_states).squeeze(-1)
        else:
            values = torch.zeros_like(selected_log_probs)
        
        return selected_log_probs, values
    
    def _compute_reference_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reference model log probabilities"""
        outputs = self.reference_model(input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits')
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, 2, input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE
        
        Args:
            rewards: [batch, 1]
            values: [batch, seq_len]
        
        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len]
        """
        # Simplified: use last value as baseline
        advantages = rewards.squeeze(-1) - values[:, -1]
        returns = rewards.squeeze(-1).unsqueeze(1).expand_as(values)
        
        return advantages.unsqueeze(1).expand_as(values), returns


def create_rlhf_trainer(
    config: RLHFConfig,
    policy_model: nn.Module,
    reference_model: nn.Module,
    reward_model: RewardModel,
) -> PPOTrainer:
    """
    Factory function to create RLHF trainer
    
    Args:
        config: RLHF configuration
        policy_model: Policy model to train
        reference_model: Frozen reference model
        reward_model: Reward model
    
    Returns:
        PPOTrainer instance
    """
    value_model = ValueModel(hidden_size=4096)
    
    return PPOTrainer(
        config,
        policy_model,
        reference_model,
        reward_model,
        value_model,
    )
