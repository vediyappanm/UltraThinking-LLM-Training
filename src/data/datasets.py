"""
Dataset configurations and loaders for ULTRATHINK training
Supports multiple popular datasets with easy switching
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    name: str = "wikitext"  # Dataset name
    subset: Optional[str] = "wikitext-2-raw-v1"  # Dataset subset/config
    split_train: str = "train"
    split_val: str = "validation" 
    split_test: str = "test"
    text_column: str = "text"  # Column containing text data
    max_length: int = 512
    tokenizer_name: str = "gpt2"
    streaming: bool = False
    cache_dir: Optional[str] = None
    num_proc: int = 4
    
    # Local dataset options
    local_path: Optional[str] = None
    file_type: str = "json"  # json, txt, csv, parquet
    
    # Custom preprocessing
    min_length: int = 10  # Minimum text length
    max_samples: Optional[int] = None  # Limit number of samples
    
    # Data mixing (for multiple datasets)
    mixing_weights: Optional[Dict[str, float]] = None

# Popular dataset configurations
DATASET_CONFIGS = {
    "wikitext": DatasetConfig(
        name="wikitext",
        subset="wikitext-2-raw-v1",
        text_column="text",
        max_length=512,
        streaming=False
    ),
    "wikitext-103": DatasetConfig(
        name="wikitext",
        subset="wikitext-103-raw-v1", 
        text_column="text",
        max_length=1024,
        streaming=True
    ),
    # Use a maintained mirror of OpenWebText
    "openwebtext": DatasetConfig(
        name="Skylion007/openwebtext",
        subset=None,
        text_column="text",
        max_length=1024,
        streaming=True
    ),
    "pile": DatasetConfig(
        name="EleutherAI/pile",
        subset=None,
        text_column="text",
        max_length=2048,
        streaming=True
    ),
    "c4": DatasetConfig(
        name="c4",
        subset="en",
        text_column="text",
        max_length=512,
        streaming=True
    ),
    # BookCorpus script is deprecated on HF; use the open variant
    "bookcorpus": DatasetConfig(
        name="bookcorpusopen",
        subset=None,
        text_column="text",
        max_length=1024,
        streaming=True
    ),
    "oscar": DatasetConfig(
        name="oscar",
        subset="unshuffled_deduplicated_en",
        text_column="text",
        max_length=512,
        streaming=True
    ),
    # Wikipedia English snapshot
    "wikipedia": DatasetConfig(
        name="wikimedia/wikipedia",
        subset="20220301.en",
        text_column="text",
        max_length=1024,
        streaming=True
    ),
    "dummy": DatasetConfig(
        name="dummy",
        subset=None,
        text_column="text",
        max_length=512,
        streaming=False
    )
}

class TextDataset(Dataset):
    """Generic text dataset for language modeling"""
    
    def __init__(self, config: DatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and preprocess data"""
        if self.config.name == "dummy":
            return self._create_dummy_data()
        elif self.config.local_path:
            return self._load_local_data()
        else:
            return self._load_hf_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        logger.info("Creating dummy dataset for testing...")
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can learn complex patterns from data.",
            "Transformers have revolutionized the field of NLP.",
        ] * 2000  # Repeat to create more samples
        
        return [{"text": text} for text in dummy_texts]
    
    def _load_local_data(self):
        """Load data from local files"""
        logger.info(f"Loading local data from {self.config.local_path}")
        data = []
        
        if self.config.file_type == "json":
            with open(self.config.local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if self.config.text_column in item:
                        data.append({self.config.text_column: item[self.config.text_column]})
        
        elif self.config.file_type == "txt":
            with open(self.config.local_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split into chunks
                chunks = text.split('\n\n')  # Split by paragraphs
                data = [{self.config.text_column: chunk.strip()} for chunk in chunks if len(chunk.strip()) > self.config.min_length]
        
        logger.info(f"Loaded {len(data)} samples from local file")
        return data
    
    def _load_hf_data(self):
        """Load data from Hugging Face datasets"""
        logger.info(f"Loading {self.config.name} dataset from Hugging Face...")
        
        try:
            target_name = self.config.name
            target_subset = self.config.subset
            # Legacy to new mapping if needed
            legacy_map = {
                "openwebtext": "Skylion007/openwebtext",
                "bookcorpus": "bookcorpusopen",
            }
            if target_name in legacy_map:
                target_name = legacy_map[target_name]

            if target_subset:
                dataset = load_dataset(
                    target_name,
                    target_subset,
                    split=self.split,
                    streaming=self.config.streaming,
                    cache_dir=self.config.cache_dir
                )
            else:
                dataset = load_dataset(
                    target_name,
                    split=self.split,
                    streaming=self.config.streaming,
                    cache_dir=self.config.cache_dir
                )
            
            # Convert to list if not streaming
            if not self.config.streaming:
                data = []
                for item in dataset:
                    if self.config.text_column in item and item[self.config.text_column]:
                        text = item[self.config.text_column].strip()
                        if len(text) >= self.config.min_length:
                            data.append({self.config.text_column: text})
                            
                            if self.config.max_samples and len(data) >= self.config.max_samples:
                                break
                
                logger.info(f"Loaded {len(data)} samples from {self.config.name}")
                return data
            else:
                # For streaming datasets, we'll handle differently
                return dataset
                
        except Exception as e:
            logger.error(f"Failed to load {self.config.name}: {e}")
            logger.info("Falling back to dummy dataset...")
            return self._create_dummy_data()
    
    def __len__(self):
        if hasattr(self.data, '__len__'):
            return len(self.data)
        else:
            # For streaming datasets, return a large number
            return self.config.max_samples or 100000
    
    def __getitem__(self, idx):
        if isinstance(self.data, list):
            item = self.data[idx % len(self.data)]
        else:
            # For streaming datasets
            item = next(iter(self.data.skip(idx).take(1)))
        
        text = item[self.config.text_column]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        # IMPORTANT: ignore loss on padding positions
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class MixedDataset(Dataset):
    """Dataset that mixes multiple datasets with specified weights"""
    
    def __init__(self, datasets: Dict[str, Dataset], weights: Dict[str, float]):
        self.datasets = datasets
        self.weights = weights
        self.dataset_names = list(datasets.keys())
        
        # Calculate cumulative weights for sampling
        total_weight = sum(weights.values())
        self.cumulative_weights = []
        cumsum = 0
        for name in self.dataset_names:
            cumsum += weights[name] / total_weight
            self.cumulative_weights.append(cumsum)
        
        # Calculate total length
        self.total_length = sum(len(ds) for ds in datasets.values())
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Sample dataset based on weights
        import random
        rand = random.random()
        
        for i, cum_weight in enumerate(self.cumulative_weights):
            if rand <= cum_weight:
                dataset_name = self.dataset_names[i]
                dataset = self.datasets[dataset_name]
                # Sample from the selected dataset
                dataset_idx = idx % len(dataset)
                return dataset[dataset_idx]
        
        # Fallback to first dataset
        return self.datasets[self.dataset_names[0]][idx % len(self.datasets[self.dataset_names[0]])]

def create_dataset(config: Union[str, DatasetConfig], split: str = "train") -> Dataset:
    """Create a dataset from config"""
    if isinstance(config, str):
        if config in DATASET_CONFIGS:
            config = DATASET_CONFIGS[config]
        else:
            raise ValueError(f"Unknown dataset config: {config}")
    
    return TextDataset(config, split)

def create_mixed_dataset(configs: Dict[str, Union[str, DatasetConfig]], 
                        weights: Dict[str, float], 
                        split: str = "train") -> MixedDataset:
    """Create a mixed dataset from multiple configs"""
    datasets = {}
    for name, config in configs.items():
        datasets[name] = create_dataset(config, split)
    
    return MixedDataset(datasets, weights)

def create_dataloader(dataset: Dataset, 
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    """Create a DataLoader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )

# Example usage and dataset information
DATASET_INFO = {
    "wikitext": {
        "description": "WikiText language modeling dataset",
        "size": "~100MB (wikitext-2), ~500MB (wikitext-103)",
        "language": "English",
        "domain": "Wikipedia articles",
        "license": "Creative Commons"
    },
    "openwebtext": {
        "description": "Open source recreation of WebText",
        "size": "~40GB",
        "language": "English", 
        "domain": "Web pages",
        "license": "Public domain"
    },
    "pile": {
        "description": "Large-scale curated text dataset",
        "size": "~800GB",
        "language": "English",
        "domain": "Books, web, academic papers, code",
        "license": "MIT"
    },
    "c4": {
        "description": "Colossal Clean Crawled Corpus",
        "size": "~750GB",
        "language": "Multiple (English subset available)",
        "domain": "Web crawl data",
        "license": "ODC-BY"
    },
    "bookcorpus": {
        "description": "Collection of over 11,000 books",
        "size": "~5GB",
        "language": "English",
        "domain": "Books and novels",
        "license": "Research use"
    }
}

def print_dataset_info():
    """Print information about available datasets"""
    print("\n📚 Available Datasets for ULTRATHINK Training:\n")
    
    for name, info in DATASET_INFO.items():
        print(f"🔹 {name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Language: {info['language']}")
        print(f"   Domain: {info['domain']}")
        print(f"   License: {info['license']}")
        print()
    
    print("💡 Usage Examples:")
    print("   --dataset wikitext              # Small, fast download")
    print("   --dataset openwebtext           # Medium size, diverse")
    print("   --dataset pile                  # Large, comprehensive")
    print("   --dataset custom --data_path /path/to/data.json")
    print()

if __name__ == "__main__":
    print_dataset_info()
