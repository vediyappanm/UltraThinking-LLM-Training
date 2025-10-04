# UltraThink Colab: Training + Monitoring Quickstart

This guide provides copy-paste cells for Colab to train UltraThink and monitor DRE/MoE metrics. It uses MLflow (file store) and the training log written to each run's `output_dir`.

---

## 1) Environment setup

```python
# Check GPU and install dependencies
!nvidia-smi || echo "No NVIDIA GPU"

!pip -q install --upgrade pip
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers datasets accelerate mlflow tqdm sentencepiece protobuf==3.20.* safetensors
```

## 2) Get latest code

```python
import os
repo = 'UltraThinking-LLM-Training'
if not os.path.isdir(f'/content/{repo}'):
    !git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git /content/UltraThinking-LLM-Training
else:
    %cd /content/UltraThinking-LLM-Training
    !git pull origin main
%cd /content/UltraThinking-LLM-Training
```

## 3) Configure MLflow local store

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:/content/mlruns"
os.makedirs("/content/mlruns", exist_ok=True)
```

## 4) Smoke test run (forced expert path)

```python
RUN_NAME = 'expert_forced_colab_check'
OUT_DIR = f'./outputs/{RUN_NAME}'

!python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --train_samples 200 --val_samples 50 \
  --tokenizer_name gpt2 --vocab_size 50257 \
  --hidden_size 512 --num_layers 6 --num_heads 8 --num_kv_heads 4 \
  --intermediate_size 2048 --max_seq_length 256 \
  --enable_moe --enable_dre --dre_force_path expert \
  --num_knowledge_experts 4 --num_skill_experts 2 --num_meta_experts 1 --num_safety_experts 1 \
  --moe_top_k 1 --expert_capacity 1.25 \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 3e-4 --weight_decay 0.01 \
  --warmup_steps 500 --num_epochs 1 \
  --use_amp --gradient_checkpointing \
  --eval_frequency 5 --perf_log_interval 200 --num_workers 2 \
  --use_mlflow --run_name $RUN_NAME \
  --output_dir $OUT_DIR
```

## 5) Live log tail (console)

```python
# Live tail the training log for a short period
import time
from IPython.display import clear_output

LOG_PATH = f"{OUT_DIR}/training.log"
for _ in range(60):  # ~2 minutes
    clear_output(wait=True)
    try:
        with open(LOG_PATH, 'r') as f:
            data = f.read()
        # Show last ~4000 chars
        print(data[-4000:])
    except Exception as e:
        print("Waiting for log...", e)
    time.sleep(2)
```

## 6) MLflow metrics: DRE/MoE visualization

```python
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

EXPERIMENT_NAME = 'UltraThinking-LLM-Training'
RUN_NAME = 'expert_forced_colab_check'  # set to your run

client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

# Find the latest run by run name
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if len(runs) == 0:
    raise RuntimeError(f"No runs found for name: {RUN_NAME}")
run_id = runs.iloc[0]['run_id']
print("Using run:", run_id)

# Helper to fetch a full metric history into a DataFrame
def metric_df(metric_name: str) -> pd.DataFrame:
    hist = client.get_metric_history(run_id, metric_name)
    if not hist:
        return pd.DataFrame(columns=['step','value','metric'])
    return pd.DataFrame({
        'step': [m.step for m in hist],
        'value': [m.value for m in hist],
        'metric': metric_name,
    })

# Fetch key metrics
metrics_to_plot = [
    'train/step_loss',
    'train/tokens_per_sec',
    'moe/avg_routing_entropy',
]
frames = [metric_df(m) for m in metrics_to_plot]
plot_df = pd.concat(frames, ignore_index=True)

# Plot timeseries
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for i, m in enumerate(metrics_to_plot):
    df = plot_df[plot_df.metric == m]
    axes[i].plot(df['step'], df['value'], label=m)
    axes[i].set_ylabel(m)
    axes[i].legend(loc='best')
axes[-1].set_xlabel('global_step')
plt.tight_layout()
plt.show()

# DRE path distribution (latest values)
PATHS = ['fast','standard','expert','deep','ultra_deep']
path_vals = {}
for p in PATHS:
    mname = f'dre/path_{p}'
    hist = client.get_metric_history(run_id, mname)
    path_vals[p] = (hist[-1].value if hist else 0.0)

plt.figure(figsize=(8,4))
plt.bar(list(path_vals.keys()), list(path_vals.values()), color='#4C78A8')
plt.title('DRE Path Distribution (latest)')
plt.ylabel('Percent')
plt.ylim(0, 100)
plt.show()

# Optional: top-expert concentration by group (latest)
EXPERT_GROUPS = ['knowledge','skill','meta','safety']
concentration = {}
for g in EXPERT_GROUPS:
    m = f'moe/{g}_top_expert_pct'
    hist = client.get_metric_history(run_id, m)
    concentration[g] = (hist[-1].value if hist else 0.0)

plt.figure(figsize=(8,4))
plt.bar(list(concentration.keys()), list(concentration.values()), color='#F58518')
plt.title('MoE max expert concentration (latest)')
plt.ylabel('Percent')
plt.ylim(0, 100)
plt.show()
```

## 7) Natural routing run (no forced path)

```python
RUN_NAME = 'ultrathink_train_colab'
OUT_DIR = f'./outputs/{RUN_NAME}'

!python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --train_samples 2000 --val_samples 1000 \
  --tokenizer_name gpt2 --vocab_size 50257 \
  --hidden_size 512 --num_layers 6 --num_heads 8 --num_kv_heads 4 \
  --intermediate_size 2048 --max_seq_length 256 \
  --enable_moe --enable_dre \
  --num_knowledge_experts 4 --num_skill_experts 2 --num_meta_experts 1 --num_safety_experts 1 \
  --moe_top_k 1 --expert_capacity 1.25 \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 500 --num_epochs 1 \
  --use_amp --gradient_checkpointing \
  --eval_frequency 100 --perf_log_interval 200 --num_workers 2 \
  --use_mlflow --run_name $RUN_NAME \
  --output_dir $OUT_DIR
```

## 8) Small-dataset quick check (dummy data)

```python
RUN_NAME = 'dummy_quick_check'
OUT_DIR = f'./outputs/{RUN_NAME}'

!python train_ultrathink.py \
  --dataset dummy --train_samples 200 --val_samples 50 \
  --tokenizer_name gpt2 --vocab_size 50257 \
  --hidden_size 512 --num_layers 6 --num_heads 8 --num_kv_heads 4 \
  --intermediate_size 2048 --max_seq_length 256 \
  --enable_moe --enable_dre \
  --moe_top_k 1 --expert_capacity 1.25 \
  --batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 --num_epochs 1 \
  --use_amp --gradient_checkpointing \
  --use_mlflow --run_name $RUN_NAME \
  --output_dir $OUT_DIR
```

---

### Notes
- Training log file path: `./outputs/<run_name>/training.log`.
- MLflow experiment: `UltraThinking-LLM-Training` (can be changed in `train_ultrathink.py`).
- Metrics logged include:
  - `train/step_loss`, `train/tokens_per_sec`.
  - `moe/avg_routing_entropy`, `moe/*_top_expert_pct`, `moe/aux_*` (aggregated as available).
  - `dre/path_<fast|standard|expert|deep|ultra_deep>`, `dre/avg_complexity`, `dre/avg_confidence` when provided.
- For longer runs, consider `--moe_top_k 2` and keep `--expert_capacity 1.25` as per roadmap.
