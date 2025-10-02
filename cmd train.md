For Local/Colab GPU (16GB+ VRAM recommended)
bash
python train_ultrathink.py `
  --dataset c4 --dataset_subset en --streaming `
  --tokenizer_name gpt2 --vocab_size 50257 `
  --hidden_size 768 --num_layers 12 --num_heads 12 --num_kv_heads 4 `
  --intermediate_size 3072 --max_seq_length 1024 `
  --activation swiglu `
  --dropout 0.1 --attention_dropout 0.1 `
  --enable_moe `
  --num_knowledge_experts 16 --num_skill_experts 8 `
  --num_meta_experts 4 --num_safety_experts 2 `
  --moe_top_k 2 --expert_capacity 1.25 `
  --enable_dre --dre_warmup_steps 1000 `
  --enable_constitutional `
  --amp_warmup_steps 500 `
  --batch_size 2 --gradient_accumulation_steps 64 `
  --learning_rate 3e-4 --weight_decay 0.1 `
  --adam_beta1 0.9 --adam_beta2 0.999 `
  --warmup_steps 5000 --num_epochs 3 `
  --gradient_clipping 1.0 `
  --use_amp --gradient_checkpointing --use_flash_attention `
  --eval_frequency 1 `
  --use_mlflow --run_name ultrathink_complete_model `
  --output_dir ./outputs/ultrathink_complete
For High-End GPU (32GB+ VRAM - A100/V100)
bash
python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --tokenizer_name gpt2 --vocab_size 50257 \
  --hidden_size 2048 --num_layers 24 --num_heads 16 --num_kv_heads 8 \
  --intermediate_size 8192 --max_seq_length 2048 \
  --activation swiglu \
  --dropout 0.05 --attention_dropout 0.05 \
  --enable_moe \
  --num_knowledge_experts 32 --num_skill_experts 16 \
  --num_meta_experts 8 --num_safety_experts 4 \
  --moe_top_k 2 --expert_capacity 1.25 \
  --enable_dre --dre_warmup_steps 2000 \
  --enable_constitutional \
  --enable_multimodal \
  --amp_warmup_steps 1000 \
  --batch_size 4 --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.999 \
  --warmup_steps 10000 --num_epochs 3 \
  --gradient_clipping 1.0 \
  --use_amp --gradient_checkpointing --use_flash_attention \
  --eval_frequency 2 \
  --use_mlflow --run_name ultrathink_large_complete \
  --output_dir ./outputs/ultrathink_large_complete