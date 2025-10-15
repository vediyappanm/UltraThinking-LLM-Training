# 📁 Project Structure

Complete guide to understanding the ULTRATHINK codebase organization.

## Directory Overview

```
UltraThinking-LLM-Training/
├── train_ultrathink.py          # Main training script (CLI)
├── train_advanced.py             # YAML config-based training
├── app_gradio.py                 # Web UI for inference
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Docker orchestration
├── setup.py                      # Package installation
├── pytest.ini                    # Test configuration
├── .gitignore                    # Git ignore rules
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── ultrathink.py        # Main UltraThink model integration
│   │   ├── architecture.py      # Base transformer (GQA, RoPE, SwiGLU)
│   │   ├── moe_advanced.py      # Mixture-of-Experts implementation
│   │   ├── dynamic_reasoning.py # Dynamic Reasoning Engine (DRE)
│   │   ├── constitutional_ai.py # Safety and alignment
│   │   └── multimodal.py        # Multi-modal encoders
│   │
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── datasets.py          # Dataset loaders (C4, WikiText, etc.)
│   │   ├── tokenization.py      # Tokenizer utilities
│   │   ├── validation.py        # Data validation
│   │   └── preprocessing.py     # Data preprocessing
│   │
│   ├── training/                 # Training infrastructure
│   │   ├── __init__.py
│   │   ├── loop.py              # Main training loop with diagnostics
│   │   ├── optim.py             # Optimizer configuration
│   │   ├── scheduler.py         # Learning rate scheduling
│   │   ├── rlhf_advanced.py     # RLHF 2.0 implementation
│   │   ├── distributed_4d.py    # 4D parallelism (DP, FSDP, TP, PP)
│   │   └── callbacks.py         # Training callbacks
│   │
│   ├── monitoring/               # Metrics and monitoring
│   │   ├── __init__.py
│   │   ├── metrics.py           # Custom metrics (MoE, DRE stats)
│   │   └── system_monitor.py    # GPU/CPU/memory monitoring
│   │
│   ├── security/                 # Security and validation
│   │   ├── __init__.py
│   │   └── input_validation.py  # Input sanitization
│   │
│   └── evaluation/               # Model evaluation
│       ├── __init__.py
│       ├── benchmarks.py        # Benchmark suite
│       └── generation.py        # Text generation utilities
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_models.py           # Model architecture tests
│   ├── test_training.py         # Training loop tests
│   ├── test_data.py             # Data loading tests
│   ├── test_moe.py              # MoE-specific tests
│   ├── test_dre.py              # DRE-specific tests
│   ├── smoke_test.py            # Quick smoke test
│   └── integration/             # Integration tests
│       └── test_end_to_end.py
│
├── configs/                      # Configuration files
│   ├── train_tiny.yaml          # Tiny model config (CPU-friendly)
│   ├── train_small.yaml         # Small model config (6-16GB GPU)
│   ├── train_medium.yaml        # Medium model config (24-40GB GPU)
│   ├── train_large.yaml         # Large model config (40GB+ GPU)
│   ├── moe_config.yaml          # MoE-specific configuration
│   ├── dre_config.yaml          # DRE-specific configuration
│   └── benchmark_config.yaml    # Benchmark configuration
│
├── scripts/                      # Utility scripts
│   ├── inference.py             # Run inference on trained models
│   ├── evaluate.py              # Evaluate model performance
│   ├── profile_model.py         # Performance profiling
│   ├── export_model.py          # Export to ONNX/TorchScript
│   ├── cleanup.py               # Clean cache and temp files
│   └── download_datasets.py     # Pre-download datasets
│
├── docs/                         # Documentation
│   ├── README.md                # Documentation index
│   ├── TRAINING_QUICKSTART.md   # 5-minute quickstart
│   ├── BENCHMARKS.md            # Performance benchmarks
│   ├── COMPARISON.md            # Framework comparisons
│   ├── TROUBLESHOOTING.md       # Common issues & solutions
│   ├── ROADMAP.md               # Future plans
│   ├── PROJECT_STRUCTURE.md     # This file
│   ├── datasets.md              # Dataset guide
│   ├── training_small.md        # Small dataset training
│   ├── training_deepspeed.md    # DeepSpeed integration
│   ├── colab.md                 # Google Colab guide
│   ├── colab.ipynb              # Colab notebook
│   ├── faq.md                   # FAQ
│   └── images/                  # Documentation images
│
├── outputs/                      # Training outputs (gitignored)
│   └── .gitkeep
│
├── checkpoints/                  # Model checkpoints (gitignored)
│   └── .gitkeep
│
└── mlruns/                       # MLflow tracking (gitignored)
    └── .gitkeep
```

## Core Components

### 1. Training Scripts

#### `train_ultrathink.py`
Main training script with CLI arguments.

**Usage**:
```bash
python train_ultrathink.py \
  --dataset c4 \
  --hidden_size 768 \
  --num_layers 12 \
  --enable_moe
```

**Key Features**:
- Simple CLI interface
- All hyperparameters as arguments
- Good for quick experiments

#### `train_advanced.py`
YAML configuration-based training for reproducibility.

**Usage**:
```bash
python train_advanced.py --config configs/train_small.yaml
```

**Key Features**:
- YAML configuration files
- Better for production
- Easy to version control

### 2. Model Architecture (`src/models/`)

#### `ultrathink.py`
Main model class integrating all components.

```python
from src.models.ultrathink import UltraThinkModel

model = UltraThinkModel(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    enable_moe=True,
    enable_dre=True,
    enable_constitutional=True
)
```

#### `architecture.py`
Base transformer with modern optimizations:
- **GQA** (Grouped Query Attention) - 4x KV cache reduction
- **RoPE** (Rotary Position Embeddings) - Better sequence modeling
- **SwiGLU** - Improved activation function
- **RMSNorm** - Faster normalization
- **Flash Attention** - Memory-efficient attention

#### `moe_advanced.py`
Hierarchical Mixture-of-Experts:
- 4-level expert hierarchy (Knowledge/Skill/Meta/Safety)
- Balanced router with entropy regularization
- Load balancing losses
- Expert utilization metrics

#### `dynamic_reasoning.py`
Dynamic Reasoning Engine (DRE):
- 5 adaptive compute paths (FAST/STANDARD/EXPERT/DEEP/ULTRA_DEEP)
- Complexity scoring (9 features)
- Adaptive routing
- 40-60% compute savings

#### `constitutional_ai.py`
Safety and alignment:
- 10-category harm detection
- Self-critique module
- Revision loop
- Constitutional principles

### 3. Data Pipeline (`src/data/`)

#### `datasets.py`
Dataset loaders for:
- **C4** - Colossal Clean Crawled Corpus
- **WikiText** - Wikipedia articles
- **The Pile** - Diverse text corpus
- **OpenWebText** - Web text
- **Custom datasets** - Your own data

**Example**:
```python
from src.data.datasets import get_dataset

dataset = get_dataset(
    name='c4',
    subset='en',
    streaming=True,
    split='train'
)
```

#### `tokenization.py`
Tokenizer utilities:
- GPT-2 tokenizer (default)
- SentencePiece support
- Custom tokenizer integration

### 4. Training Infrastructure (`src/training/`)

#### `loop.py`
Main training loop with:
- Automatic mixed precision (AMP)
- Gradient accumulation
- Gradient clipping
- Checkpointing
- Metrics logging
- Diagnostic tools

#### `distributed_4d.py`
4D parallelism support:
- **Data Parallelism (DP)** - Replicate model across GPUs
- **Fully Sharded Data Parallelism (FSDP)** - Shard model parameters
- **Tensor Parallelism (TP)** - Split tensors across GPUs
- **Pipeline Parallelism (PP)** - Split layers across GPUs

### 5. Monitoring (`src/monitoring/`)

#### `metrics.py`
Custom metrics:
- MoE expert utilization
- DRE path distribution
- Routing entropy
- Load balance factors
- Gradient norms

#### `system_monitor.py`
System resource monitoring:
- GPU utilization
- Memory usage
- Disk I/O
- Network bandwidth

### 6. Configuration (`configs/`)

All configs follow this structure:

```yaml
model:
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  
training:
  batch_size: 4
  learning_rate: 3e-4
  num_epochs: 3
  
data:
  dataset: c4
  streaming: true
  max_seq_length: 2048
```

## File Naming Conventions

- **Scripts**: `snake_case.py` (e.g., `train_ultrathink.py`)
- **Modules**: `snake_case.py` (e.g., `dynamic_reasoning.py`)
- **Classes**: `PascalCase` (e.g., `UltraThinkModel`)
- **Functions**: `snake_case` (e.g., `get_dataset()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQ_LENGTH`)
- **Configs**: `snake_case.yaml` (e.g., `train_small.yaml`)
- **Docs**: `UPPER_SNAKE_CASE.md` (e.g., `README.md`)

## Import Patterns

### Absolute Imports (Preferred)
```python
from src.models.ultrathink import UltraThinkModel
from src.data.datasets import get_dataset
from src.training.loop import train_model
```

### Relative Imports (Within Package)
```python
# In src/models/ultrathink.py
from .architecture import AdvancedGPTModel
from .moe_advanced import MoELayer
```

## Adding New Components

### Adding a New Model Component

1. Create file in `src/models/`:
```python
# src/models/my_component.py
import torch.nn as nn

class MyComponent(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your implementation
        
    def forward(self, x):
        # Your forward pass
        return x
```

2. Add to `src/models/__init__.py`:
```python
from .my_component import MyComponent
```

3. Integrate in `ultrathink.py`:
```python
from .my_component import MyComponent

class UltraThinkModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.my_component = MyComponent(config)
```

### Adding a New Dataset

1. Add loader in `src/data/datasets.py`:
```python
def load_my_dataset(split='train', streaming=False):
    # Your dataset loading logic
    return dataset
```

2. Register in `get_dataset()`:
```python
def get_dataset(name, **kwargs):
    if name == 'my_dataset':
        return load_my_dataset(**kwargs)
```

### Adding a New Configuration

1. Create YAML file in `configs/`:
```yaml
# configs/my_config.yaml
model:
  hidden_size: 512
  num_layers: 6

training:
  batch_size: 8
  learning_rate: 1e-4
```

2. Use with training script:
```bash
python train_advanced.py --config configs/my_config.yaml
```

## Testing Structure

### Unit Tests
Test individual components:
```python
# tests/test_models.py
def test_ultrathink_forward():
    model = UltraThinkModel(config)
    output = model(input_ids)
    assert output.shape == expected_shape
```

### Integration Tests
Test component interactions:
```python
# tests/integration/test_end_to_end.py
def test_full_training_pipeline():
    # Test complete training workflow
    pass
```

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_models.py

# With coverage
pytest --cov=src --cov-report=html

# Smoke test (quick validation)
python tests/smoke_test.py
```

## Output Structure

### Training Outputs (`outputs/`)
```
outputs/
└── experiment_name/
    ├── config.yaml              # Saved configuration
    ├── checkpoints/             # Model checkpoints
    │   ├── checkpoint-1000/
    │   ├── checkpoint-2000/
    │   └── best_model/
    ├── logs/                    # Training logs
    │   ├── train.log
    │   └── metrics.json
    └── tensorboard/             # TensorBoard logs
```

### MLflow Tracking (`mlruns/`)
```
mlruns/
└── 0/                           # Experiment ID
    └── run_id/                  # Run ID
        ├── metrics/             # Logged metrics
        ├── params/              # Hyperparameters
        ├── artifacts/           # Model artifacts
        └── meta.yaml            # Metadata
```

## Environment Variables

Set these for customization:

```bash
# Cache directory
export HF_HOME=/path/to/cache

# MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Weights & Biases
export WANDB_PROJECT=ultrathink
export WANDB_ENTITY=your_username

# CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Best Practices

### Code Organization
- ✅ Keep related functionality together
- ✅ Use clear, descriptive names
- ✅ Add docstrings to all functions/classes
- ✅ Follow PEP 8 style guide

### Configuration
- ✅ Use YAML configs for reproducibility
- ✅ Version control your configs
- ✅ Document all hyperparameters

### Testing
- ✅ Write tests for new features
- ✅ Run tests before committing
- ✅ Maintain >80% code coverage

### Documentation
- ✅ Update docs when adding features
- ✅ Include usage examples
- ✅ Keep README.md current

## Quick Navigation

| Need to... | Go to... |
|-----------|----------|
| **Train a model** | `train_ultrathink.py` or `train_advanced.py` |
| **Add new architecture** | `src/models/` |
| **Add new dataset** | `src/data/datasets.py` |
| **Modify training loop** | `src/training/loop.py` |
| **Add metrics** | `src/monitoring/metrics.py` |
| **Write tests** | `tests/` |
| **Create config** | `configs/` |
| **Run inference** | `scripts/inference.py` |
| **Profile performance** | `scripts/profile_model.py` |

## Related Documentation

- **[Training Quickstart](TRAINING_QUICKSTART.md)** - Get started in 5 minutes
- **[Advanced Training Guide](../ADVANCED_TRAINING_GUIDE.md)** - Deep dive into features
- **[Contributing Guide](../CONTRIBUTING.md)** - Contribution guidelines
- **[Architecture Overview](../ARCHITECTURE_OVERVIEW.md)** - System design

---

**Questions?** Open an issue or discussion on GitHub!
