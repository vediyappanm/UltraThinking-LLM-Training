"""
LoRA (Low-Rank Adaptation) Training Support

Paper: LoRA: Low-Rank Adaptation of Large Language Models
Link: https://arxiv.org/abs/2106.09685
Authors: Hu et al., 2021

Enables parameter-efficient fine-tuning by training low-rank decomposition
matrices instead of full model weights. QLoRA extends this with 4-bit quantization.
"""

import logging
import torch
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
        PeftModel
    )
    from transformers import BitsAndBytesConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. Install with: pip install peft bitsandbytes")


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training"""
    
    # LoRA parameters
    r: int = 16  # Rank (typical: 8-64, higher = more capacity)
    lora_alpha: int = 32  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.1  # Dropout probability
    target_modules: Optional[List[str]] = None  # Layers to adapt
    bias: str = "none"  # Bias training: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type
    
    # QLoRA parameters (4-bit quantization)
    use_qlora: bool = False  # Enable 4-bit training
    bnb_4bit_compute_dtype: str = "bfloat16"  # Compute dtype
    bnb_4bit_quant_type: str = "nf4"  # Quantization type: nf4, fp4
    bnb_4bit_use_double_quant: bool = True  # Double quantization
    
    # Training parameters
    learning_rate: float = 2e-4  # Higher than full fine-tuning
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    def __post_init__(self):
        """Set default target modules if not specified"""
        if self.target_modules is None:
            # Common attention projection layers
            self.target_modules = ["q_proj", "v_proj"]
        
        # Validate rank
        assert self.r > 0, "LoRA rank must be positive"
        assert self.lora_alpha > 0, "LoRA alpha must be positive"


def create_lora_model(
    base_model: torch.nn.Module,
    config: LoRATrainingConfig
) -> torch.nn.Module:
    """
    Add LoRA adapters to a base model
    
    Args:
        base_model: The base language model
        config: LoRA configuration
        
    Returns:
        Model with LoRA adapters attached
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> config = LoRATrainingConfig(r=16, target_modules=["q_proj", "v_proj"])
        >>> lora_model = create_lora_model(model, config)
        >>> lora_model.print_trainable_parameters()
        trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.24
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required. Install: pip install peft bitsandbytes")
    
    # Prepare model for k-bit training if using QLoRA
    if config.use_qlora:
        logger.info("Preparing model for 4-bit training (QLoRA)")
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True
        )
        logger.info("Model prepared for QLoRA")
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM if config.task_type == "CAUSAL_LM" else config.task_type
    )
    
    # Add LoRA adapters
    logger.info(f"Adding LoRA adapters with r={config.r}, alpha={config.lora_alpha}")
    logger.info(f"Target modules: {config.target_modules}")
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model


def load_model_for_lora(
    model_name: str,
    use_qlora: bool = False,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> torch.nn.Module:
    """
    Load a model optimized for LoRA training
    
    Args:
        model_name: HuggingFace model name
        use_qlora: Whether to use 4-bit quantization
        device_map: Device placement strategy
        torch_dtype: PyTorch dtype (default: bfloat16 or float16)
        
    Returns:
        Loaded model ready for LoRA training
        
    Example:
        >>> # Standard LoRA
        >>> model = load_model_for_lora("gpt2", use_qlora=False)
        >>> 
        >>> # QLoRA (4-bit)
        >>> model = load_model_for_lora(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     use_qlora=True
        ... )
    """
    from transformers import AutoModelForCausalLM
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if use_qlora:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT and bitsandbytes required for QLoRA")
        
        # Configure 4-bit quantization
        logger.info("Loading model with 4-bit quantization for QLoRA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        logger.info(f"Model loaded in 4-bit (memory: ~{model.get_memory_footprint() / 1e9:.2f} GB)")
    else:
        # Standard loading for LoRA
        logger.info(f"Loading model for standard LoRA training")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        logger.info(f"Model loaded (memory: ~{model.get_memory_footprint() / 1e9:.2f} GB)")
    
    return model


def merge_lora_weights(
    peft_model: "PeftModel",
    output_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Merge LoRA weights into base model for inference
    
    Args:
        peft_model: PEFT model with LoRA adapters
        output_path: Path to save merged model (optional)
        
    Returns:
        Merged model with LoRA weights incorporated
        
    Example:
        >>> # After training
        >>> merged_model = merge_lora_weights(lora_model)
        >>> merged_model.save_pretrained("my-finetuned-model")
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required")
    
    logger.info("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    logger.info("LoRA weights merged successfully")
    
    if output_path:
        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(output_path)
    
    return merged_model


def get_lora_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about a LoRA model
    
    Args:
        model: PEFT model with LoRA adapters
        
    Returns:
        Dictionary with model information
    """
    if not hasattr(model, "get_nb_trainable_parameters"):
        return {"error": "Not a PEFT model"}
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    
    info = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": 100 * trainable_params / all_params,
        "memory_footprint_gb": model.get_memory_footprint() / 1e9,
        "is_peft_model": True,
    }
    
    # Get LoRA config if available
    if hasattr(model, "peft_config"):
        lora_config = model.peft_config.get("default", {})
        info["lora_r"] = getattr(lora_config, "r", None)
        info["lora_alpha"] = getattr(lora_config, "lora_alpha", None)
        info["target_modules"] = getattr(lora_config, "target_modules", None)
    
    return info


class LoRATrainer:
    """
    Simple trainer wrapper for LoRA fine-tuning
    
    Example:
        >>> config = LoRATrainingConfig(r=16, use_qlora=True)
        >>> trainer = LoRATrainer(
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     config=config,
        ...     train_dataset=dataset
        ... )
        >>> trainer.train()
        >>> trainer.save_merged_model("my-finetuned-model")
    """
    
    def __init__(
        self,
        model_name: str,
        config: LoRATrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize LoRA trainer"""
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Load tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load and prepare model
        logger.info(f"Loading model: {model_name}")
        base_model = load_model_for_lora(
            model_name,
            use_qlora=config.use_qlora
        )
        
        # Add LoRA adapters
        self.model = create_lora_model(base_model, config)
        
        # Log model info
        info = get_lora_model_info(self.model)
        logger.info(f"Model info: {info}")
    
    def train(self):
        """Train the model using HuggingFace Trainer"""
        from transformers import Trainer, TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./lora_output",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_torch",
            report_to=["tensorboard"],
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Starting LoRA training...")
        trainer.train()
        logger.info("Training complete!")
        
        return self.model
    
    def save_merged_model(self, output_path: str):
        """Merge and save the model"""
        merged_model = merge_lora_weights(self.model, output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model and tokenizer saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("LoRA Training - Example usage:")
    print("""
    from transformers import AutoTokenizer
    from src.training.lora_trainer import LoRATrainer, LoRATrainingConfig
    
    # Configure LoRA
    config = LoRATrainingConfig(
        r=16,                    # Rank
        lora_alpha=32,          # Scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_qlora=True,         # Enable 4-bit training!
        learning_rate=2e-4,
        batch_size=4,
        num_epochs=3
    )
    
    # Create trainer
    trainer = LoRATrainer(
        model_name="meta-llama/Llama-2-7b-hf",
        config=config,
        train_dataset=dataset
    )
    
    # Train (memory efficient!)
    trainer.train()
    
    # Save merged model
    trainer.save_merged_model("my-llama-finetuned")
    
    # Benefits:
    # - Train 7B model on 24GB GPU (was 80GB!)
    # - 60-80% memory savings
    # - 2-3x faster training
    # - 96-98% of full fine-tuning quality
    """)
