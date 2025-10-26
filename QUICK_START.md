# ⚡ Quick Start - UltraThinking Training

## 🎯 Choose Your Training Mode

### 1️⃣ Easy Training (Beginners)
**Perfect for**: Testing, learning, modest hardware (4-8GB GPU or CPU)

```bash
# Windows
train_easy.bat

# Linux/Mac
python train_complete.py --dataset custom --data_path ./easy_dataset.json
```

**Time**: 10-30 minutes | **Features**: Basic training

---

### 2️⃣ Advanced Training (All Features)
**Perfect for**: Full feature demonstration (8GB+ GPU required)

```bash
# Windows
train_advanced.bat

# Linux/Mac - see TRAINING_GUIDE.md
```

**Time**: 30-60 minutes | **Features**: MoE, DRE, Constitutional AI, Flash Attention

---

### 3️⃣ Checkpoint Training (Long Runs)
**Perfect for**: Resumable training, production runs

```bash
# Windows
train_with_checkpoints.bat
```

**Features**: Auto-save every epoch, easy resumption

---

## 📊 Monitor Training

### View MLflow Dashboard
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open: http://localhost:5000

### View Logs
```bash
# Windows
type outputs\easy_training\training.log

# Linux/Mac
tail -f outputs/easy_training/training.log
```

---

## 🔄 Resume Training

```bash
python train_complete.py --resume_checkpoint ./outputs/checkpoint_training/checkpoint_epoch_5.pt
```

---

## 🛠️ Common Issues

### Out of Memory?
```bash
python train_complete.py --batch_size 2 --gradient_checkpointing --hidden_size 512
```

### Training Too Slow?
```bash
python train_complete.py --use_amp --use_flash_attention --batch_size 16
```

### NaN Loss?
```bash
python train_complete.py --learning_rate 1e-4 --warmup_steps 5000 --gradient_clipping 1.0
```

---

## 📁 Output Structure

```
outputs/
└── easy_training/
    ├── checkpoint_epoch_0.pt      # Checkpoint files
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    ├── final_model/               # Final model (HuggingFace format)
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── ...
    ├── training.log               # Training logs
    └── evaluation_results.json    # Evaluation metrics
```

---

## 🎓 Key Flags Reference

| Purpose | Flag | Example |
|---------|------|---------|
| **Model Size** | `--hidden_size` | `--hidden_size 768` |
| **Training Length** | `--num_epochs` | `--num_epochs 5` |
| **Batch Size** | `--batch_size` | `--batch_size 4` |
| **Learning Rate** | `--learning_rate` | `--learning_rate 3e-4` |
| **Enable MoE** | `--enable_moe` | `--enable_moe` |
| **Enable DRE** | `--enable_dre` | `--enable_dre` |
| **Save Memory** | `--gradient_checkpointing` | `--gradient_checkpointing` |
| **Speed Up** | `--use_amp` | `--use_amp` |
| **Resume** | `--resume_checkpoint` | `--resume_checkpoint ./path/to/checkpoint.pt` |
| **Output Dir** | `--output_dir` | `--output_dir ./my_training` |

---

## ✅ Pre-flight Checklist

- [ ] Python 3.9+ installed: `python --version`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset exists: Check `easy_dataset.json` is present
- [ ] GPU available (optional): `nvidia-smi` or use CPU
- [ ] Disk space: ~5GB free for checkpoints

---

## 🚀 Ready to Train!

**Start with Easy Training**:
```bash
train_easy.bat
```

**See full guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

**Need help?**: Check [Troubleshooting](TRAINING_GUIDE.md#troubleshooting)

---

**Training Time Estimates**:
- Easy Training: 10-30 min
- Advanced Training: 30-60 min
- Production Training: Hours to days (depending on data size)
