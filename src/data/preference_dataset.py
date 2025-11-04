"""
Preference Dataset Handler for DPO/ORPO Training

Handles loading and preprocessing of preference datasets where each example
contains a prompt with a chosen (preferred) and rejected (dispreferred) response.
"""

import logging
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    Dataset for preference-based training (DPO, ORPO, etc.)
    
    Expected format:
    Each example should have:
    - prompt: The instruction or context
    - chosen: The preferred/better response
    - rejected: The dispreferred/worse response
    
    Example:
        >>> from datasets import load_dataset
        >>> hf_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        >>> dataset = PreferenceDataset(hf_dataset, tokenizer)
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
                   'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'])
    """
    
    def __init__(
        self,
        dataset: Union[str, any],
        tokenizer: any,
        max_length: int = 512,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        split: str = "train",
    ):
        """
        Initialize preference dataset
        
        Args:
            dataset: HuggingFace dataset name or dataset object
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            prompt_key: Key for prompt in dataset
            chosen_key: Key for chosen response
            rejected_key: Key for rejected response
            split: Dataset split to use (train/test/validation)
        """
        # Load dataset if string provided
        if isinstance(dataset, str):
            logger.info(f"Loading dataset: {dataset}")
            self.dataset = load_dataset(dataset, split=split)
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Preference dataset initialized with {len(self.dataset)} examples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example
        
        Returns:
            Dictionary containing tokenized chosen and rejected responses
        """
        example = self.dataset[idx]
        
        # Extract prompt and responses
        prompt = example.get(self.prompt_key, "")
        chosen = example.get(self.chosen_key, "")
        rejected = example.get(self.rejected_key, "")
        
        # Combine prompt with responses
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        # Tokenize chosen response
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        chosen_labels = chosen_encodings["input_ids"].clone()
        rejected_labels = rejected_encodings["input_ids"].clone()
        
        # Mask padding tokens in labels (set to -100)
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            # Chosen response
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            
            # Rejected response
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        }


class AnthropicHHDataset(PreferenceDataset):
    """
    Convenience class for Anthropic Helpful-Harmless (HH) dataset
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = AnthropicHHDataset(tokenizer, split="train")
    """
    
    def __init__(
        self,
        tokenizer: any,
        max_length: int = 512,
        split: str = "train",
    ):
        # Anthropic HH-RLHF dataset format
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_key="prompt",  # Note: HH-RLHF has different keys
            chosen_key="chosen",
            rejected_key="rejected",
            split=split
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Anthropic HH-RLHF format has full conversations in chosen/rejected
        No separate prompt field
        """
        example = self.dataset[idx]
        
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]
        
        # Tokenize
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        chosen_labels = chosen_encodings["input_ids"].clone()
        rejected_labels = rejected_encodings["input_ids"].clone()
        
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        }


def load_preference_dataset(
    dataset_name: str,
    tokenizer: any,
    max_length: int = 512,
    split: str = "train"
) -> PreferenceDataset:
    """
    Load a preference dataset by name
    
    Supported datasets:
    - "Anthropic/hh-rlhf": Helpful and Harmless dataset
    - "HuggingFaceH4/ultrafeedback_binarized": UltraFeedback
    - Any HuggingFace dataset with prompt/chosen/rejected format
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        split: Dataset split
        
    Returns:
        PreferenceDataset instance
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = load_preference_dataset(
        ...     "Anthropic/hh-rlhf",
        ...     tokenizer,
        ...     split="train"
        ... )
    """
    if "anthropic" in dataset_name.lower() or "hh-rlhf" in dataset_name.lower():
        return AnthropicHHDataset(tokenizer, max_length, split)
    else:
        # Generic preference dataset
        return PreferenceDataset(dataset_name, tokenizer, max_length, split=split)


def create_preference_example(
    prompt: str,
    chosen: str,
    rejected: str
) -> Dict[str, str]:
    """
    Create a preference example from prompt and responses
    
    Useful for custom datasets or manual creation
    
    Args:
        prompt: The instruction or context
        chosen: The preferred response
        rejected: The dispreferred response
        
    Returns:
        Dictionary with prompt, chosen, rejected keys
    
    Example:
        >>> example = create_preference_example(
        ...     prompt="Write a poem about AI",
        ...     chosen="In silicon dreams, intelligence grows...",
        ...     rejected="AI is cool."
        ... )
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def validate_preference_dataset(dataset: any) -> bool:
    """
    Validate that a dataset has the required format for preference training
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    example = dataset[0]
    
    # Check for required fields
    required_chosen = ["chosen_input_ids", "chosen_attention_mask", "chosen_labels"]
    required_rejected = ["rejected_input_ids", "rejected_attention_mask", "rejected_labels"]
    
    for field in required_chosen + required_rejected:
        if field not in example:
            raise ValueError(f"Missing required field: {field}")
    
    # Check shapes match
    if example["chosen_input_ids"].shape != example["rejected_input_ids"].shape:
        logger.warning("Chosen and rejected sequences have different shapes")
    
    logger.info("Dataset validation passed")
    return True


# Example usage
if __name__ == "__main__":
    print("Preference Dataset - Example usage:")
    print("""
    from transformers import AutoTokenizer
    from src.data.preference_dataset import load_preference_dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load preference dataset
    dataset = load_preference_dataset(
        "Anthropic/hh-rlhf",
        tokenizer,
        max_length=512,
        split="train"
    )
    
    # Get a sample
    sample = dataset[0]
    print("Keys:", sample.keys())
    print("Chosen shape:", sample["chosen_input_ids"].shape)
    print("Rejected shape:", sample["rejected_input_ids"].shape)
    
    # Validate dataset
    from src.data.preference_dataset import validate_preference_dataset
    validate_preference_dataset(dataset)
    
    # Use with DPO/ORPO
    from src.training.dpo_trainer import DPOTrainer, DPOConfig
    
    config = DPOConfig()
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        train_dataset=dataset
    )
    trainer.train()
    """)
