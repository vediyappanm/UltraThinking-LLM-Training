# 🎯 START HERE - UltraThinking Training

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Training Mode

#### 🟢 **Easy Mode** (Recommended for beginners)
```bash
train_easy.bat
```
- **Time**: 10-30 minutes
- **Hardware**: 4-8GB GPU or CPU
- **Features**: Basic training

#### 🟡 **Advanced Mode** (All features)
```bash
train_advanced.bat
```
- **Time**: 30-60 minutes
- **Hardware**: 8GB+ GPU
- **Features**: MoE, DRE, Constitutional AI

#### 🔵 **Checkpoint Mode** (Long runs)
```bash
train_with_checkpoints.bat
```
- **Features**: Auto-save, resume support

### Step 3: Monitor Training
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open: http://localhost:5000

---

## 📚 Documentation Guide

| Read This | When You Need |
|-----------|---------------|
| **QUICK_START.md** | Fast reference and common commands |
| **TRAINING_GUIDE.md** | Complete guide with troubleshooting |
| **ALL_FLAGS_REFERENCE.md** | Every flag explained |
| **TRAINING_SETUP_COMPLETE.md** | Overview of what's been created |

---

## 🎮 Training Modes Explained

### Easy Training
- **Model**: Small (768 hidden, 12 layers)
- **Dataset**: easy_dataset.json (100 examples)
- **Features**: Basic transformer, AMP, gradient checkpointing
- **Best for**: Testing, learning, modest hardware

### Advanced Training
- **Model**: Medium (1024 hidden, 12 layers)
- **Dataset**: easy_dataset.json (100 examples)
- **Features**: MoE, DRE, Constitutional AI, Flash Attention
- **Best for**: Full feature demonstration

### Checkpoint Training
- **Model**: Medium (1024 hidden, 12 layers)
- **Features**: Frequent checkpointing, auto-resume
- **Best for**: Long training runs, production

---

## 🎛️ Common Commands

### Start Training
```bash
# Windows
train_easy.bat

# Linux/Mac
python train_complete.py --dataset custom --data_path ./easy_dataset.json
```

### Resume Training
```bash
python train_complete.py --resume_checkpoint ./outputs/checkpoint_training/checkpoint_epoch_5.pt
```

### View Logs
```bash
# Windows
type outputs\easy_training\training.log

# Linux/Mac
tail -f outputs/easy_training/training.log
```

### View MLflow Dashboard
```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## 🛠️ Quick Fixes

### Out of Memory?
```bash
python train_complete.py --batch_size 2 --gradient_checkpointing
```

### Too Slow?
```bash
python train_complete.py --use_amp --use_flash_attention
```

### NaN Loss?
```bash
python train_complete.py --learning_rate 1e-4 --gradient_clipping 1.0
```

---

## 📁 What's Included

### ✅ Dataset
- `easy_dataset.json` - 100 training examples

### ✅ Training Scripts
- `train_complete.py` - Main training script
- `train_easy.bat` - Easy training
- `train_advanced.bat` - Advanced training
- `train_with_checkpoints.bat` - Checkpoint training

### ✅ Documentation
- `QUICK_START.md` - Quick reference
- `TRAINING_GUIDE.md` - Complete guide
- `ALL_FLAGS_REFERENCE.md` - All flags
- `TRAINING_SETUP_COMPLETE.md` - Setup overview

---

## 🎯 Recommended Path

1. **Read**: `QUICK_START.md` (5 minutes)
2. **Run**: `train_easy.bat` (10-30 minutes)
3. **Monitor**: MLflow dashboard
4. **Experiment**: Try different flags
5. **Scale**: Use `train_advanced.bat`

---

## ✅ Pre-Flight Checklist

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `easy_dataset.json` exists
- [ ] GPU available (optional)
- [ ] 5GB+ free disk space

---

## 🚀 Ready to Train!

**Start now**:
```bash
train_easy.bat
```

**Need help?** See `TRAINING_GUIDE.md`

---

**Project**: UltraThinking LLM Training  
**Status**: 🟢 Ready to Train  
**Documentation**: Complete  
**Dataset**: Ready  
**Scripts**: Ready  

**Happy Training! 🎉**
