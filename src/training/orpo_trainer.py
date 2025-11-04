"""
ORPO (Odds Ratio Preference Optimization) Trainer

Paper: ORPO: Monolithic Preference Optimization without Reference Model
Link: https://arxiv.org/abs/2403.07691
Authors: Hong & Lee, 2024

ORPO combines supervised fine-tuning and preference alignment in a single stage
without requiring a reference model, making it simpler and more efficient than DPO.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ORPOConfig:
    """Configuration for ORPO training"""
    
    # ORPO-specific parameters
    alpha: float = 1.0  # Weight for SFT loss (typical: 0.5-2.0)
    beta: float = 0.1  # Weight for OR loss (typical: 0.1-0.5)
    
    # Training parameters
    learning_rate: float = 8e-6  # Slightly higher than DPO
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    warmup_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.alpha >= 0, "alpha must be non-negative"
        assert self.beta > 0, "beta must be positive"


class ORPOTrainer:
    """
    ORPO Trainer - Single-stage alignment without reference model
    
    Key Advantages:
    - No reference model needed (saves memory!)
    - Combines SFT + alignment in one stage
    - More sample-efficient than DPO
    - Simpler implementation
    
    How it works:
    - SFT loss: Standard language modeling on chosen responses
    - OR loss: Odds ratio comparing chosen vs rejected responses
    - Final loss: weighted combination of both
    
    Example:
        >>> config = ORPOConfig(alpha=1.0, beta=0.1)
        >>> trainer = ORPOTrainer(
        ...     model=model,
        ...     config=config,
        ...     train_dataset=preference_dataset
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ORPOConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize ORPO trainer
        
        Args:
            model: Model to train (no reference model needed!)
            config: ORPO configuration
            train_dataset: Training dataset with preference pairs
            eval_dataset: Evaluation dataset (optional)
            tokenizer: Tokenizer for the model
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler (created if None)
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
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
        
        logger.info(f"ORPO Trainer initialized with alpha={config.alpha}, beta={config.beta}")
        logger.info("No reference model needed - memory efficient!")
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for a sequence
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            
        Returns:
            log_probs: Sum of log probabilities per sequence [batch]
        """
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # Shift for causal LM
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
        
        # Mask padding and sum
        target_log_probs = target_log_probs * shift_mask
        sequence_log_probs = target_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ORPO loss combining SFT and Odds Ratio objectives
        
        Loss = α * L_SFT + L_OR
        
        Where:
        - L_SFT: Standard language modeling loss on chosen responses
        - L_OR: Odds ratio loss favoring chosen over rejected
        - α: Weight balancing the two objectives
        
        The odds ratio loss encourages the model to assign higher odds
        to chosen responses compared to rejected ones.
        
        Args:
            chosen_logps: Log probs for chosen responses [batch]
            rejected_logps: Log probs for rejected responses [batch]
            chosen_input_ids: Input IDs for chosen (for SFT loss)
            chosen_attention_mask: Attention mask for chosen
            
        Returns:
            loss: Combined ORPO loss
            metrics: Dictionary of logging metrics
        """
        # 1. SFT Loss: Standard language modeling on chosen responses
        # This ensures the model learns to generate fluent text
        sft_loss = -chosen_logps.mean()
        
        # 2. Odds Ratio (OR) Loss
        # Compute log odds for chosen responses
        # log odds = log(p / (1-p)) = log(p) - log(1-p)
        # For numerical stability, we use: log(1-p) = log(1 - exp(log p))
        log_odds_chosen = chosen_logps - torch.log(
            1 - torch.exp(chosen_logps) + 1e-10  # Add epsilon for stability
        )
        
        # Compute log odds for rejected responses
        log_odds_rejected = rejected_logps - torch.log(
            1 - torch.exp(rejected_logps) + 1e-10
        )
        
        # OR loss: We want log_odds_chosen > log_odds_rejected
        # Use negative log sigmoid to penalize when chosen odds are lower
        log_odds_ratio = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(self.config.beta * log_odds_ratio).mean()
        
        # 3. Combined loss
        loss = self.config.alpha * sft_loss + or_loss
        
        # Compute metrics for logging
        with torch.no_grad():
            # Implicit rewards (similar to DPO)
            chosen_rewards = self.config.beta * log_odds_chosen
            rejected_rewards = self.config.beta * log_odds_rejected
            
            # Accuracy: how often is chosen preferred?
            accuracy = (log_odds_chosen > log_odds_rejected).float().mean()
            
            # Odds ratio margin
            or_margin = log_odds_ratio.mean()
            
            metrics = {
                "loss": loss.item(),
                "sft_loss": sft_loss.item(),
                "or_loss": or_loss.item(),
                "accuracy": accuracy.item(),
                "chosen_rewards_mean": chosen_rewards.mean().item(),
                "rejected_rewards_mean": rejected_rewards.mean().item(),
                "or_margin": or_margin.item(),
                "chosen_logps_mean": chosen_logps.mean().item(),
                "rejected_logps_mean": rejected_logps.mean().item(),
            }
        
        return loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary containing:
                - chosen_input_ids: Input IDs for chosen responses
                - chosen_attention_mask: Attention mask for chosen
                - chosen_labels: Labels for chosen
                - rejected_input_ids: Input IDs for rejected responses
                - rejected_attention_mask: Attention mask for rejected
                - rejected_labels: Labels for rejected
                
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Compute log probs for chosen responses
        chosen_logps = self.compute_log_probs(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"]
        )
        
        # Compute log probs for rejected responses
        rejected_logps = self.compute_log_probs(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"]
        )
        
        # Compute ORPO loss
        loss, metrics = self.orpo_loss(
            chosen_logps,
            rejected_logps,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting ORPO training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Alpha (SFT weight): {self.config.alpha}")
        logger.info(f"Beta (OR weight): {self.config.beta}")
        
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
                            f"SFT: {metrics['sft_loss']:.4f} | "
                            f"OR: {metrics['or_loss']:.4f} | "
                            f"Acc: {metrics['accuracy']:.3f} | "
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
            chosen_logps = self.compute_log_probs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"]
            )
            rejected_logps = self.compute_log_probs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"]
            )
            
            # Compute metrics
            loss, metrics = self.orpo_loss(
                chosen_logps,
                rejected_logps,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"]
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
    print("ORPO Trainer - Example usage:")
    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.training.orpo_trainer import ORPOTrainer, ORPOConfig
    
    # Load model (no reference model needed!)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Configure ORPO
    config = ORPOConfig(
        alpha=1.0,  # SFT weight
        beta=0.1,   # OR weight
        learning_rate=8e-6,
        num_epochs=3
    )
    
    # Create trainer (simpler than DPO!)
    trainer = ORPOTrainer(
        model=model,
        config=config,
        train_dataset=preference_dataset,
        tokenizer=tokenizer
    )
    
    # Train in one stage!
    trainer.train()
    
    # Benefits:
    # - No reference model needed (50% less memory!)
    # - Single-stage training (faster!)
    # - Combines SFT + alignment
    # - More sample-efficient
    """)
