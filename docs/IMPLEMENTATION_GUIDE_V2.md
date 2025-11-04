# 🛠️ UltraThinking 2.0 - Implementation Guide for Contributors

> **For developers who want to contribute to v2.0 features**

## 🎯 Overview

This guide provides detailed technical specifications, code templates, and implementation patterns for contributors working on UltraThinking 2.0 features.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [DPO/ORPO Implementation](#dpoorpo-implementation)
3. [LoRA/QLoRA Implementation](#loraqloraple Implementation)
4. [Evaluation Suite](#evaluation-suite)
5. [Quantization Pipeline](#quantization-pipeline)
6. [Mamba Architecture](#mamba-architecture)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation Requirements](#documentation-requirements)

---

## 🚀 Getting Started

### Development Environment Setup

```bash
# Clone and setup
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training

# Create dev environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Project Structure for v2.0

```
UltraThinking-LLM-Training/
├── src/
│   ├── models/
│   │   ├── transformer.py           # Existing
│   │   ├── mamba.py                 # NEW - Q2 2025
│   │   └── hybrid.py                # NEW - Q2 2025
│   ├── training/
│   │   ├── trainer.py               # Existing
│   │   ├── dpo_trainer.py           # NEW - Q1 2025
│   │   ├── orpo_trainer.py          # NEW - Q1 2025
│   │   └── lora_trainer.py          # NEW - Q1 2025
│   ├── evaluation/
│   │   ├── benchmark_suite.py       # NEW - Q1 2025
│   │   └── metrics.py               # Existing
│   ├── quantization/
│   │   ├── quantizer.py             # NEW - Q1 2025
│   │   └── calibration.py           # NEW - Q1 2025
│   └── data/
│       ├── preference_dataset.py    # NEW - Q1 2025
│       └── ...
├── configs/
│   ├── dpo_config.yaml              # NEW
│   ├── orpo_config.yaml             # NEW
│   └── lora_config.yaml             # NEW
├── tests/
│   ├── test_dpo.py                  # NEW
│   ├── test_orpo.py                 # NEW
│   └── test_lora.py                 # NEW
└── docs/
    ├── tutorials/
    │   ├── dpo_tutorial.md          # NEW
    │   └── lora_tutorial.md         # NEW
    └── ...
```

---

## 1️⃣ DPO/ORPO Implementation

### Design Principles

1. **Build on existing**: Extend `BaseTrainer` class
2. **Modular**: Separate loss functions, data handling
3. **Configurable**: All hyperparameters in config
4. **Tested**: Unit tests for each component
5. **Documented**: Docstrings + tutorial

### File Structure

```
src/training/
├── dpo_trainer.py         # Main DPO trainer
├── orpo_trainer.py        # Main ORPO trainer
├── preference_loss.py     # Loss functions
└── alignment_utils.py     # Shared utilities

src/data/
├── preference_dataset.py  # Dataset handling
└── preference_sampler.py  # Batch sampling

configs/
├── dpo_config.yaml        # DPO configuration
└── orpo_config.yaml       # ORPO configuration

tests/
├── test_dpo_loss.py       # Loss function tests
├── test_dpo_trainer.py    # Integration tests
└── test_preference_data.py # Data tests
```

### Implementation Template: DPO Trainer

```python
# src/training/dpo_trainer.py

"""
Direct Preference Optimization (DPO) Trainer

Paper: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
Link: https://arxiv.org/abs/2305.18290

DPO trains directly on preference pairs without needing a separate reward model.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base_trainer import BaseTrainer
from ..data.preference_dataset import PreferenceDataset


class DPOConfig:
    """Configuration for DPO training"""
    
    def __init__(
        self,
        beta: float = 0.1,              # KL penalty coefficient
        label_smoothing: float = 0.0,   # Label smoothing
        loss_type: str = "sigmoid",     # sigmoid, hinge, ipo
        reference_free: bool = False,   # Use reference-free DPO
        **kwargs
    ):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.reference_free = reference_free
        
        # Base training config
        self.learning_rate = kwargs.get("learning_rate", 5e-7)
        self.max_length = kwargs.get("max_length", 512)
        self.batch_size = kwargs.get("batch_size", 4)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 4)


class DPOTrainer(BaseTrainer):
    """
    DPO Trainer for alignment using preference data
    
    Usage:
        >>> config = DPOConfig(beta=0.1)
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
        model,
        ref_model,
        config: DPOConfig,
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
        **kwargs
    ):
        super().__init__(model=model, config=config, **kwargs)
        
        # Reference model (frozen)
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for a sequence
        
        Args:
            model: The language model
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            
        Returns:
            log_probs: Log probabilities for each sequence [batch]
        """
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # Shift for autoregressive modeling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        shift_mask = attention_mask[:, 1:].contiguous()
        target_log_probs = target_log_probs * shift_mask
        
        # Sum log probs per sequence
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
        Compute DPO loss
        
        Loss = -log(σ(β * [log(π_θ/π_ref)(y_w) - log(π_θ/π_ref)(y_l)]))
        
        Where:
        - π_θ is the policy model
        - π_ref is the reference model
        - y_w is the preferred/chosen response
        - y_l is the rejected/losing response
        - β is the KL penalty coefficient
        - σ is the sigmoid function
        
        Args:
            policy_chosen_logps: Log probs from policy for chosen responses
            policy_rejected_logps: Log probs from policy for rejected responses
            reference_chosen_logps: Log probs from reference for chosen
            reference_rejected_logps: Log probs from reference for rejected
            
        Returns:
            loss: DPO loss
            metrics: Dictionary of metrics for logging
        """
        # Compute log ratios
        policy_chosen_ratio = policy_chosen_logps - reference_chosen_logps
        policy_rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # Compute logits for the Bradley-Terry model
        logits = policy_chosen_ratio - policy_rejected_ratio
        
        # Compute loss based on loss type
        if self.config.loss_type == "sigmoid":
            # Standard DPO loss
            losses = -F.logsigmoid(self.config.beta * logits)
        elif self.config.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - self.config.beta * logits)
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) variant
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            losses = losses * (1 - self.config.label_smoothing) + \
                     (-F.logsigmoid(-self.config.beta * logits)) * self.config.label_smoothing
        
        loss = losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            # Implicit reward
            chosen_rewards = self.config.beta * policy_chosen_ratio
            rejected_rewards = self.config.beta * policy_rejected_ratio
            
            # Accuracy (how often chosen is preferred)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "chosen_rewards_mean": chosen_rewards.mean().item(),
                "rejected_rewards_mean": rejected_rewards.mean().item(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
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
            metrics: Dictionary of metrics
        """
        # Get log probs from policy model
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
        
        # Get log probs from reference model (no gradients)
        with torch.no_grad():
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
        
        # Backward pass
        loss.backward()
        
        return metrics
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                metrics = self.training_step(batch)
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if step % self.config.logging_steps == 0:
                    self.log_metrics(metrics, step)
                
                # Evaluation
                if step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics, step, prefix="eval")
                
                # Checkpointing
                if step % self.config.save_steps == 0:
                    self.save_checkpoint(step)


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load models
    model_name = "gpt2"
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load preference dataset
    from datasets import load_dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    preference_dataset = PreferenceDataset(dataset, tokenizer)
    
    # Configure and train
    config = DPOConfig(beta=0.1, learning_rate=5e-7)
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        config=config,
        train_dataset=preference_dataset
    )
    
    trainer.train()
```

### ORPO Implementation Template

```python
# src/training/orpo_trainer.py

"""
Odds Ratio Preference Optimization (ORPO) Trainer

Paper: ORPO: Monolithic Preference Optimization without Reference Model
Link: https://arxiv.org/abs/2403.07691

ORPO combines SFT and preference optimization in a single stage without needing a reference model.
"""

class ORPOConfig:
    """Configuration for ORPO training"""
    
    def __init__(
        self,
        alpha: float = 1.0,    # Weight for SFT loss
        beta: float = 0.1,     # Weight for OR loss
        **kwargs
    ):
        self.alpha = alpha
        self.beta = beta
        # ... other configs


class ORPOTrainer(BaseTrainer):
    """
    ORPO Trainer - Single-stage alignment without reference model
    
    Key advantage: No need for a separate reference model!
    """
    
    def __init__(self, model, config: ORPOConfig, train_dataset, **kwargs):
        super().__init__(model=model, config=config, **kwargs)
        # No reference model needed!
    
    def orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_labels: torch.Tensor
    ):
        """
        ORPO loss combines SFT + Odds Ratio preference
        
        Loss = α * L_SFT + L_OR
        
        Where:
        - L_SFT is standard language modeling loss on chosen responses
        - L_OR is odds ratio loss comparing chosen vs rejected
        """
        # SFT loss on chosen responses
        sft_loss = -chosen_logps.mean()
        
        # Odds ratio for chosen
        log_odds_chosen = chosen_logps - torch.log(
            1 - torch.exp(chosen_logps) + 1e-10
        )
        
        # Odds ratio for rejected
        log_odds_rejected = rejected_logps - torch.log(
            1 - torch.exp(rejected_logps) + 1e-10
        )
        
        # OR loss: prefer higher odds for chosen
        log_odds_ratio = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(self.config.beta * log_odds_ratio).mean()
        
        # Combined loss
        loss = self.config.alpha * sft_loss + or_loss
        
        metrics = {
            "loss": loss.item(),
            "sft_loss": sft_loss.item(),
            "or_loss": or_loss.item(),
        }
        
        return loss, metrics
```

### Testing Template

```python
# tests/test_dpo_trainer.py

import pytest
import torch
from src.training.dpo_trainer import DPOTrainer, DPOConfig


class TestDPOTrainer:
    """Test suite for DPO trainer"""
    
    @pytest.fixture
    def setup_models(self):
        """Setup models for testing"""
        # Use small models for testing
        from transformers import GPT2LMHeadModel
        
        policy = GPT2LMHeadModel.from_pretrained("gpt2")
        reference = GPT2LMHeadModel.from_pretrained("gpt2")
        
        return policy, reference
    
    def test_log_probs_computation(self, setup_models):
        """Test log probability computation"""
        policy, reference = setup_models
        config = DPOConfig()
        trainer = DPOTrainer(policy, reference, config, train_dataset=None)
        
        # Create dummy input
        input_ids = torch.randint(0, 50257, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        # Compute log probs
        log_probs = trainer.compute_log_probs(
            policy, input_ids, attention_mask, labels
        )
        
        assert log_probs.shape == (2,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()
    
    def test_dpo_loss(self, setup_models):
        """Test DPO loss computation"""
        policy, reference = setup_models
        config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(policy, reference, config, train_dataset=None)
        
        # Create dummy log probs
        policy_chosen = torch.tensor([-10.0, -12.0])
        policy_rejected = torch.tensor([-15.0, -18.0])
        ref_chosen = torch.tensor([-11.0, -13.0])
        ref_rejected = torch.tensor([-14.0, -17.0])
        
        # Compute loss
        loss, metrics = trainer.dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected
        )
        
        assert loss.item() >= 0
        assert 0 <= metrics["accuracy"] <= 1
        assert "reward_margin" in metrics
    
    def test_reference_model_frozen(self, setup_models):
        """Test that reference model is frozen"""
        policy, reference = setup_models
        config = DPOConfig()
        trainer = DPOTrainer(policy, reference, config, train_dataset=None)
        
        for param in trainer.ref_model.parameters():
            assert not param.requires_grad


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 2️⃣ LoRA/QLoRA Implementation

### Implementation using PEFT Library

```python
# src/training/lora_trainer.py

"""
LoRA (Low-Rank Adaptation) Training

Paper: LoRA: Low-Rank Adaptation of Large Language Models
Link: https://arxiv.org/abs/2106.09685

Trains only ~0.1-1% of parameters while maintaining quality.
"""

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
import torch


class LoRATrainingConfig:
    """Configuration for LoRA training"""
    
    def __init__(
        self,
        # LoRA parameters
        r: int = 16,                              # Rank
        lora_alpha: int = 32,                     # Scaling
        lora_dropout: float = 0.1,
        target_modules: list = None,              # Which layers to adapt
        bias: str = "none",                       # "none", "all", "lora_only"
        
        # QLoRA parameters
        use_qlora: bool = False,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",         # or "fp4"
        bnb_4bit_use_double_quant: bool = True,
        
        # Training parameters
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        **kwargs
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.bias = bias
        
        self.use_qlora = use_qlora
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size


def create_lora_model(base_model, config: LoRATrainingConfig):
    """
    Create a LoRA model from a base model
    
    Args:
        base_model: The base language model
        config: LoRA configuration
        
    Returns:
        peft_model: Model with LoRA adapters
    """
    # QLoRA: Quantize the base model
    if config.use_qlora:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True
        )
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM
    )
    
    # Add LoRA adapters
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    # Example output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0622
    
    return peft_model


def load_base_model_for_lora(
    model_name: str,
    use_qlora: bool = False,
    device_map: str = "auto"
):
    """
    Load base model, optionally with 4-bit quantization for QLoRA
    
    Args:
        model_name: HuggingFace model name
        use_qlora: Whether to use 4-bit quantization
        device_map: Device placement strategy
        
    Returns:
        model: Loaded model
    """
    from transformers import AutoModelForCausalLM
    
    if use_qlora:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    return model


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load base model with QLoRA
    model_name = "meta-llama/Llama-2-7b-hf"
    model = load_base_model_for_lora(model_name, use_qlora=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create LoRA model
    config = LoRATrainingConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_qlora=True
    )
    lora_model = create_lora_model(model, config)
    
    # Train (use your existing trainer)
    from src.training.trainer import Trainer
    trainer = Trainer(
        model=lora_model,
        tokenizer=tokenizer,
        config=config
    )
    trainer.train()
    
    # Merge adapters and save
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained("output/llama-2-7b-finetuned")
```

---

## 📊 Summary for Contributors

### Priority Order for Implementation

1. **Week 1-2**: DPO/ORPO
   - Core trainer classes
   - Loss functions
   - Tests
   
2. **Week 3**: LoRA/QLoRA
   - PEFT integration
   - Configuration
   - Tests
   
3. **Week 4-5**: Evaluation Suite
   - lm-evaluation-harness integration
   - Custom benchmarks
   - Tests
   
4. **Week 6-7**: Quantization
   - GPTQ/AWQ integration
   - GGUF export
   - Tests

### Getting Help

- **Questions**: [GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)
- **Issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)
- **Pull Requests**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Ready to contribute?** Pick a feature and get started! 🚀
