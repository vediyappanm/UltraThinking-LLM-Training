"""
Direct Preference Optimization (DPO) Trainer

Paper: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
Link: https://arxiv.org/abs/2305.18290
Authors: Rafailov et al., 2023

DPO simplifies RLHF by directly optimizing the policy using preference data,
eliminating the need for a separate reward model and the complexity of RL.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training"""
    
    # DPO-specific parameters
    beta: float = 0.1  # KL penalty coefficient (typical: 0.1-0.5)
    label_smoothing: float = 0.0  # Label smoothing (0.0-0.1)
    loss_type: str = "sigmoid"  # Loss type: sigmoid, hinge, ipo
    reference_free: bool = False  # Use reference-free variant
    
    # Training parameters
    learning_rate: float = 5e-7  # Lower than SFT (typical: 1e-7 to 1e-6)
    max_length: int = 512  # Maximum sequence length
    batch_size: int = 4  # Batch size (can be small due to paired data)
    gradient_accumulation_steps: int = 4  # Accumulation steps
    max_grad_norm: float = 1.0  # Gradient clipping
    num_epochs: int = 3  # Number of epochs
    logging_steps: int = 10  # Log every N steps
    eval_steps: int = 100  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    warmup_steps: int = 100  # Warmup steps
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.loss_type in ["sigmoid", "hinge", "ipo"], \
            f"Invalid loss_type: {self.loss_type}"
        assert 0 <= self.label_smoothing < 1, \
            "label_smoothing must be in [0, 1)"
        assert self.beta > 0, "beta must be positive"


class DPOTrainer:
    """
    DPO Trainer for aligning language models using preference data
    
    Key Features:
    - No reward model needed
    - More stable than PPO
    - Works with any preference dataset
    - Supports multiple loss types
    
    Example:
        >>> config = DPOConfig(beta=0.1, learning_rate=5e-7)
        >>> trainer = DPOTrainer(
        ...     model=policy_model,
        ...     ref_model=reference_model,
        ...     config=config,
        ...     train_dataset=preference_dataset
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        config: DPOConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize DPO trainer
        
        Args:
            model: Policy model to train
            ref_model: Reference model (frozen)
            config: DPO configuration
            train_dataset: Training dataset with preference pairs
            eval_dataset: Evaluation dataset (optional)
            tokenizer: Tokenizer for the model
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler (created if None)
        """
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Create scheduler if not provided
        if scheduler is None:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_dataset) * config.num_epochs // (config.batch_size * config.gradient_accumulation_steps),
                eta_min=config.learning_rate * 0.1
            )
        else:
            self.scheduler = scheduler
        
        self.global_step = 0
        self.device = next(model.parameters()).device
        
        logger.info(f"DPO Trainer initialized with beta={config.beta}, loss_type={config.loss_type}")
    
    def compute_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for a sequence
        
        Args:
            model: Language model
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            
        Returns:
            log_probs: Sum of log probabilities per sequence [batch]
        """
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # Shift for causal LM (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens and sum
        target_log_probs = target_log_probs * shift_mask
        sequence_log_probs = target_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss using Bradley-Terry model
        
        Loss = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
        
        Where:
        - π_θ is the policy model being trained
        - π_ref is the frozen reference model
        - y_w is the chosen/preferred response
        - y_l is the rejected/dispreferred response
        - β controls the deviation from reference
        - σ is the sigmoid function
        
        Args:
            policy_chosen_logps: Policy log probs for chosen [batch]
            policy_rejected_logps: Policy log probs for rejected [batch]
            reference_chosen_logps: Reference log probs for chosen [batch]
            reference_rejected_logps: Reference log probs for rejected [batch]
            
        Returns:
            loss: DPO loss
            metrics: Dictionary of logging metrics
        """
        # Compute log ratios (implicit rewards)
        policy_chosen_ratio = policy_chosen_logps - reference_chosen_logps
        policy_rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # Compute preference logits
        logits = policy_chosen_ratio - policy_rejected_ratio
        
        # Compute loss based on type
        if self.config.loss_type == "sigmoid":
            # Standard DPO with Bradley-Terry model
            losses = -F.logsigmoid(self.config.beta * logits)
        
        elif self.config.loss_type == "hinge":
            # Hinge loss variant (more robust)
            losses = torch.relu(1 - self.config.beta * logits)
        
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) - paper: arxiv.org/abs/2310.12036
            # More robust to out-of-distribution preferences
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            # Mix with loss on reversed preferences (chosen/rejected swapped)
            reversed_losses = -F.logsigmoid(-self.config.beta * logits)
            losses = (1 - self.config.label_smoothing) * losses + \
                     self.config.label_smoothing * reversed_losses
        
        loss = losses.mean()
        
        # Compute metrics for logging
        with torch.no_grad():
            # Implicit rewards (scaled by beta)
            chosen_rewards = self.config.beta * policy_chosen_ratio
            rejected_rewards = self.config.beta * policy_rejected_ratio
            
            # Accuracy: how often does model prefer chosen over rejected?
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            # Reward margin: how much better is chosen than rejected?
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "chosen_rewards_mean": chosen_rewards.mean().item(),
                "chosen_rewards_std": chosen_rewards.std().item(),
                "rejected_rewards_mean": rejected_rewards.mean().item(),
                "rejected_rewards_std": rejected_rewards.std().item(),
                "reward_margin": reward_margin.item(),
                "logits_mean": logits.mean().item(),
                "logits_std": logits.std().item(),
            }
        
        return loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary containing:
                - chosen_input_ids: Input IDs for chosen responses [batch, seq_len]
                - chosen_attention_mask: Attention mask for chosen
                - chosen_labels: Labels for chosen (usually same as input_ids)
                - rejected_input_ids: Input IDs for rejected responses
                - rejected_attention_mask: Attention mask for rejected
                - rejected_labels: Labels for rejected
                
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Compute policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"]
        )
        
        policy_rejected_logps = self.compute_log_probs(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"]
        )
        
        # Compute reference log probs (no gradients)
        with torch.no_grad():
            self.ref_model.eval()
            reference_chosen_logps = self.compute_log_probs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"]
            )
            
            reference_rejected_logps = self.compute_log_probs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"]
            )
        
        # Compute DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting DPO training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.train()
        accumulation_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                # Training step
                metrics = self.training_step(batch)
                accumulation_counter += 1
                
                # Optimizer step after accumulation
                if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer and scheduler step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Acc: {metrics['accuracy']:.3f} | "
                            f"Margin: {metrics['reward_margin']:.3f} | "
                            f"LR: {lr:.2e}"
                        )
                    
                    # Evaluation
                    if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        logger.info(f"Eval metrics: {eval_metrics}")
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        logger.info("Training complete!")
        return self.model
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Compute log probs
            policy_chosen_logps = self.compute_log_probs(
                self.model, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            policy_rejected_logps = self.compute_log_probs(
                self.model, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )
            reference_chosen_logps = self.compute_log_probs(
                self.ref_model, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            reference_rejected_logps = self.compute_log_probs(
                self.ref_model, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )
            
            # Compute metrics
            loss, metrics = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps
            )
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            num_batches += 1
        
        self.model.train()
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches
        }
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint_path = f"checkpoints/{checkpoint_name}"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    print("DPO Trainer - Example usage:")
    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.training.dpo_trainer import DPOTrainer, DPOConfig
    
    # Load models
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Configure DPO
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        num_epochs=3
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        train_dataset=preference_dataset,
        tokenizer=tokenizer
    )
    
    # Train!
    trainer.train()
    """)
