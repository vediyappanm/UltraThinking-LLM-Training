# 📚 ULTRATHINK Dataset Guide

This guide explains how to use different datasets for training ULTRATHINK models, including built-in datasets and custom data.

## 🚀 Quick Start

### Current Training Status
**Currently using**: `dummy` dataset (random tokens for testing)
**Recommended**: Switch to `wikitext` for real training

```bash
# Switch to WikiText dataset (recommended for development)
python train_ultrathink.py --dataset wikitext --vocab_size 50304 --hidden_size 384 --num_layers 4 --num_heads 6

# Use larger WikiText for better results
python train_ultrathink.py --dataset wikitext --dataset_subset wikitext-103-raw-v1
```

## 📊 Available Datasets

### 🔹 Small Datasets (Good for Development)

#### WikiText-2 (Default)
```bash
python train_ultrathink.py --dataset wikitext
```
- **Size**: ~100MB
- **Content**: Wikipedia articles
- **Best for**: Development, testing, quick experiments
- **Download time**: ~1 minute

#### WikiText-103
```bash
python train_ultrathink.py --dataset wikitext --dataset_subset wikitext-103-raw-v1
```
- **Size**: ~500MB  
- **Content**: Larger Wikipedia dataset
- **Best for**: Medium-scale training

### 🔹 Medium Datasets (Good for Serious Training)

#### OpenWebText
```bash
python train_ultrathink.py --dataset openwebtext --streaming
```
- **Size**: ~40GB
- **Content**: Web pages (WebText recreation)
- **Best for**: Diverse web content training
- **Note**: Use `--streaming` for large datasets

#### BookCorpus
```bash
python train_ultrathink.py --dataset bookcorpus --streaming
```
- **Size**: ~5GB
- **Content**: Over 11,000 books
- **Best for**: Literary and narrative training

### 🔹 Large Datasets (Production Training)

#### The Pile
```bash
python train_ultrathink.py --dataset pile --streaming --max_samples 100000
```
- **Size**: ~800GB
- **Content**: Books, web, academic papers, code
- **Best for**: Large-scale production training
- **Note**: Use `--max_samples` to limit size

#### C4 (Colossal Clean Crawled Corpus)
```bash
python train_ultrathink.py --dataset c4 --streaming
```
- **Size**: ~750GB
- **Content**: Clean web crawl data
- **Best for**: Large-scale web data training

## 🛠️ Custom Datasets

### JSON Format
Create a file with one JSON object per line:
```json
{"text": "Your training text here..."}
{"text": "Another training example..."}
```

```bash
python train_ultrathink.py --dataset custom --data_path /path/to/your/data.json
```

### Text Format
Plain text file with paragraphs separated by double newlines:
```
First paragraph of training text.
This continues the first paragraph.

Second paragraph starts here.
More content for training.
```

```bash
python train_ultrathink.py --dataset custom --data_path /path/to/your/data.txt --text_column text
```

### CSV Format
CSV file with a text column:
```csv
text,label
"Training text example 1",positive
"Training text example 2",negative
```

```bash
python train_ultrathink.py --dataset custom --data_path /path/to/your/data.csv --text_column text
```

## ⚙️ Dataset Configuration Options

### Basic Options
```bash
--dataset wikitext                    # Dataset name
--dataset_subset wikitext-2-raw-v1    # Specific subset
--max_samples 50000                   # Limit number of samples
--streaming                           # Use streaming for large datasets
--tokenizer_name gpt2                 # Tokenizer to use
```

### Custom Dataset Options
```bash
--dataset custom                      # Use custom dataset
--data_path /path/to/data.json       # Path to your data file
--text_column text                   # Column containing text
--max_samples 10000                  # Limit samples from custom data
```

### Advanced Options
```bash
--max_seq_length 1024                # Maximum sequence length
--num_workers 4                      # Data loading workers (set to 0 on Windows)
```

## 📋 Dataset Recommendations

### By Use Case

| Use Case | Recommended Dataset | Command |
|----------|-------------------|---------|
| **Quick Testing** | WikiText-2 | `--dataset wikitext` |
| **Development** | WikiText-103 | `--dataset wikitext --dataset_subset wikitext-103-raw-v1` |
| **Research** | OpenWebText | `--dataset openwebtext --streaming` |
| **Production** | The Pile | `--dataset pile --streaming --max_samples 1000000` |
| **Literary Training** | BookCorpus | `--dataset bookcorpus --streaming` |
| **Web Content** | C4 | `--dataset c4 --streaming` |
| **Custom Domain** | Your Data | `--dataset custom --data_path /path/to/data.json` |

### By Available Resources

| Resources | Dataset | Expected Training Time |
|-----------|---------|----------------------|
| **Low** (CPU, <8GB RAM) | WikiText-2 | 30 minutes |
| **Medium** (GPU, 16GB RAM) | WikiText-103 | 2 hours |
| **High** (Multi-GPU, 32GB+ RAM) | OpenWebText | 1-2 days |
| **Production** (Cluster) | The Pile | 1-2 weeks |

## 🔧 Troubleshooting

### Common Issues

#### Dataset Download Fails
```bash
# Try with smaller dataset first
python train_ultrathink.py --dataset wikitext

# Or use dummy dataset for testing
python train_ultrathink.py --dataset dummy
```

#### Out of Memory
```bash
# Reduce batch size and sequence length
python train_ultrathink.py --dataset wikitext --batch_size 1 --max_seq_length 256

# Use streaming for large datasets
python train_ultrathink.py --dataset openwebtext --streaming --batch_size 2
```

#### Slow Loading
```bash
# Set num_workers to 0 on Windows
python train_ultrathink.py --dataset wikitext --num_workers 0

# Limit samples for faster iteration
python train_ultrathink.py --dataset wikitext --max_samples 1000
```

### Dataset Validation
```bash
# Test dataset loading without training
python -c "
from src.data.datasets import create_dataset, DATASET_CONFIGS
dataset = create_dataset('wikitext', 'train')
print(f'Dataset loaded: {len(dataset)} samples')
print(f'Sample: {dataset[0]}')
"
```

## 📈 Performance Tips

### For Development
- Start with `wikitext` (small, fast)
- Use `--max_samples 1000` for quick iteration
- Set `--num_workers 0` on Windows

### For Training
- Use `--streaming` for datasets >1GB
- Increase `--max_seq_length` for better context
- Use `--batch_size 1` if memory is limited

### For Production
- Use The Pile or C4 for comprehensive training
- Enable streaming: `--streaming`
- Set appropriate `--max_samples` based on compute budget

## 🔍 Dataset Information

### Licenses and Usage
- **WikiText**: Creative Commons
- **OpenWebText**: Public Domain
- **The Pile**: MIT License
- **C4**: ODC-BY License
- **BookCorpus**: Research Use Only

### Quality and Content
- **WikiText**: High quality, encyclopedic
- **OpenWebText**: Diverse web content, filtered
- **The Pile**: Mixed domains, curated
- **C4**: Web crawl, heavily filtered
- **BookCorpus**: Literary, narrative style

## 🚀 Next Steps

1. **Start Simple**: Begin with WikiText for development
2. **Scale Up**: Move to OpenWebText for serious training  
3. **Customize**: Add your domain-specific data
4. **Optimize**: Tune batch size and sequence length
5. **Monitor**: Track training metrics and adjust

For more advanced dataset mixing and preprocessing, see the `src/data/datasets.py` module.
