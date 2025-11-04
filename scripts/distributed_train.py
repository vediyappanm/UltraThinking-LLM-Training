#!/usr/bin/env python3
"""
Distributed Training Script for Multi-GPU/Multi-Node Training
Supports FSDP, DeepSpeed, and DDP
"""

import os
import sys
import time
import math
import random
import argparse
from typing import Dict, Any, Optional
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.architecture import ModelConfig, TransformerBlock
from models.ultrathink import UltraThinkModel, UltraThinkConfig
from models.moe_advanced import ExpertConfig
from training.auto_parallel import plan_parallelism, make_deepspeed_config
from training.monitoring import get_resource_stats
from security.dp import make_dp_optimizer
from training.preference_optimization import orpo_loss, cpo_loss
from training.curriculum import CurriculumConfig, max_length_for_step
try:
    from tracking.mlflow_utils import start_run, log_params, log_metrics
except Exception:
    start_run = None  # type: ignore
    log_params = None  # type: ignore
    log_metrics = None  # type: ignore
try:
    from src.peft_utils.adapters import apply_lora, apply_qlora  # when run as module
except Exception:
    try:
        from peft_utils.adapters import apply_lora, apply_qlora
    except Exception:
        apply_lora = None
        apply_qlora = None

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print("Not running in distributed mode")
        return 0, 1, 0

    # Select backend (Windows uses gloo)
    backend = 'nccl'
    if sys.platform.startswith('win') or not torch.cuda.is_available():
        backend = 'gloo'

    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_fsdp_model(model, config):
    """Setup FSDP wrapped model"""
    # Auto wrap policy for transformer blocks
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock},
    )
    
    # Mixed precision policy
    from torch.distributed.fsdp import MixedPrecision
    if config['training']['mixed_precision'] == 'bf16':
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config['training']['mixed_precision'] == 'fp16':
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=None,
    )
    
    return model


def setup_deepspeed_model(model, config, optimizer=None, ds_override: dict = None):
    """Setup DeepSpeed model"""
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed not available")
    
    deepspeed_config = config['training'].get('deepspeed_config')
    if deepspeed_config and os.path.exists(deepspeed_config):
        with open(deepspeed_config, 'r') as f:
            ds_config = yaml.safe_load(f)
    else:
        # Default DeepSpeed config
        ds_config = {
            "train_batch_size": config['training']['batch_size'],
            "train_micro_batch_size_per_gpu": config['training'].get('micro_batch_size', 1),
            "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "fp16": {"enabled": config['training']['mixed_precision'] == 'fp16'},
            "bf16": {"enabled": config['training']['mixed_precision'] == 'bf16'},
        }
    # Apply auto-parallel override if provided
    if ds_override:
        ds_config.update(ds_override)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )
    
    return model_engine, optimizer


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--backend", type=str, choices=['fsdp', 'deepspeed', 'ddp'], default='fsdp')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--architecture", type=str, choices=['transformer','mamba','hybrid'], default=None)
    parser.add_argument("--hybrid-pattern", type=str, default=None, help="Comma-separated e.g. mamba,attn,mamba")
    parser.add_argument("--auto-parallel", action="store_true", help="Auto-plan 4D parallel and apply DS config")
    parser.add_argument("--align", type=str, choices=['none','orpo','cpo'], default='none', help="Preference optimization mode")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Global performance toggles
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Seeding for reproducibility
    seed = int(config.get('seed', 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model via UltraThink
    model_config = ModelConfig(**config['model_config_dict'])
    ut_cfg = UltraThinkConfig(model_config=model_config)
    if args.architecture:
        ut_cfg.architecture = args.architecture
    if args.hybrid_pattern:
        ut_cfg.hybrid_pattern = [x.strip() for x in args.hybrid_pattern.split(',') if x.strip()]
    # Map YAML ultrathink section (if present) to UltraThinkConfig toggles
    ultra_cfg = config.get('ultrathink', {})
    if isinstance(ultra_cfg, dict):
        if 'enable_moe' in ultra_cfg:
            ut_cfg.enable_moe = bool(ultra_cfg.get('enable_moe'))
        if 'enable_dre' in ultra_cfg:
            ut_cfg.enable_dre = bool(ultra_cfg.get('enable_dre'))
        if 'moe_layers' in ultra_cfg and isinstance(ultra_cfg['moe_layers'], (list, tuple)):
            try:
                ut_cfg.moe_layers = [int(x) for x in ultra_cfg['moe_layers']]
            except Exception:
                pass
        if 'moe_config' in ultra_cfg and isinstance(ultra_cfg['moe_config'], dict):
            try:
                # Overlay provided fields on top of existing defaults
                current = ut_cfg.moe_config.__dict__.copy()
                current.update(ultra_cfg['moe_config'])
                ut_cfg.moe_config = ExpertConfig(**current)
            except Exception:
                pass

    model = UltraThinkModel(ut_cfg)
    model = model.cuda(local_rank)
    
    # Setup distributed model
    if args.backend == 'fsdp':
        model = setup_fsdp_model(model, config)
        print("Using FSDP")
    elif args.backend == 'deepspeed':
        ds_override = None
        if args.auto_parallel:
            # Plan and generate DS config override
            cfg4d = plan_parallelism(model, global_batch_size=config['training']['batch_size'], sequence_length=config['model_config_dict'].get('n_positions', 4096))
            ds_override = make_deepspeed_config(cfg4d)
        model, _ = setup_deepspeed_model(model, config, optimizer=None, ds_override=ds_override)
        print("Using DeepSpeed")
    elif args.backend == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        print("Using DDP")
    
    print(f"Model setup complete on rank {rank}")

    # -------- Production Training Pipeline -------- #
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    log_cfg = config.get('logging', {})
    ckpt_cfg = config.get('checkpointing', {})

    tokenizer_name = data_cfg.get('tokenizer', 'gpt2')
    max_length = int(data_cfg.get('max_length', config['model_config_dict'].get('n_positions', 1024)))
    train_name = data_cfg.get('dataset', 'wikitext')
    train_subset = data_cfg.get('subset', 'wikitext-2-raw-v1')
    eval_name = data_cfg.get('eval_dataset', train_name)
    eval_subset = data_cfg.get('eval_subset', train_subset)
    use_streaming = bool(data_cfg.get('streaming', False))
    text_column = data_cfg.get('text_column', 'text')

    global_batch = int(train_cfg.get('batch_size', 8))
    micro_batch = int(train_cfg.get('micro_batch_size', 1))
    grad_accum = int(train_cfg.get('gradient_accumulation_steps', max(1, global_batch // max(1, micro_batch))))
    epochs = int(train_cfg.get('epochs', 1))
    max_steps = int(train_cfg.get('max_steps', 0))
    lr = float(train_cfg.get('learning_rate', 3e-5))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    warmup_steps = int(train_cfg.get('warmup_steps', 100))
    mixed_precision = train_cfg.get('mixed_precision', 'bf16')
    log_interval = int(log_cfg.get('log_interval', 50))
    eval_interval = int(log_cfg.get('eval_interval', 500))
    early_patience = int(log_cfg.get('early_stop_patience', 0))
    use_tensorboard = bool(log_cfg.get('tensorboard', False))
    mlflow_cfg = log_cfg.get('mlflow', {})
    use_mlflow = bool(mlflow_cfg.get('enable', False))
    mlflow_experiment = mlflow_cfg.get('experiment', 'UltraThink')
    mlflow_run_name = mlflow_cfg.get('run_name', None)
    save_interval = int(ckpt_cfg.get('save_interval', 1000))
    out_dir = ckpt_cfg.get('output_dir', './checkpoints')
    resume_from = ckpt_cfg.get('resume_from')
    use_dp = bool(train_cfg.get('use_dp', False))
    lora_method = train_cfg.get('lora', None)  # 'lora' | 'qlora' | None

    os.makedirs(out_dir, exist_ok=True) if (rank == 0) else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Curriculum setup
    cur_cfg = None
    if isinstance(train_cfg.get('curriculum'), dict):
        c = train_cfg['curriculum']
        cur_cfg = CurriculumConfig(
            start_seq=int(c.get('start_seq', 512)),
            end_seq=int(c.get('end_seq', max_length)),
            ramp_steps=int(c.get('ramp_steps', 10000)),
        )
    state = {'step': 0}

    def collate(batch):
        # Detect pairwise preference dataset
        if batch and isinstance(batch[0], dict) and ('chosen' in batch[0] and 'rejected' in batch[0]):
            chosen = [ex['chosen'] for ex in batch]
            rejected = [ex['rejected'] for ex in batch]
            return {'chosen': chosen, 'rejected': rejected}
        texts = [ex.get(text_column, '') for ex in batch if ex.get(text_column)]
        # Dynamic curriculum max length
        dyn_max_len = max_length
        if cur_cfg is not None:
            dyn_max_len = max_length_for_step(state['step'], cur_cfg)
        toks = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=dyn_max_len,
        )
        input_ids = toks['input_ids']
        labels = input_ids.clone()
        return {
            'input_ids': input_ids.cuda(local_rank, non_blocking=True),
            'attention_mask': toks['attention_mask'].cuda(local_rank, non_blocking=True),
            'labels': labels.cuda(local_rank, non_blocking=True),
        }

    # Datasets / Dataloaders
    if use_streaming:
        train_ds = load_dataset(train_name, train_subset, split='train', streaming=True)
        eval_ds = load_dataset(eval_name, eval_subset, split='validation', streaming=True)

        def stream_batches(ds, batch_size):
            buf = []
            for ex in ds:
                buf.append(ex)
                if len(buf) == batch_size:
                    yield collate(buf)
                    buf = []
        train_iter = stream_batches(train_ds, micro_batch)
        eval_iter = stream_batches(eval_ds, micro_batch)
        train_loader = None
        eval_loader = None
    else:
        train_ds = load_dataset(train_name, train_subset, split='train')
        eval_split = 'validation' if 'validation' in load_dataset(eval_name, eval_subset).keys() else 'test'
        eval_ds = load_dataset(eval_name, eval_subset, split=eval_split)
        train_sampler = None
        eval_sampler = None
        if world_size > 1 and args.backend in ['ddp','fsdp']:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=micro_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=micro_batch,
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        train_iter = None
        eval_iter = None

    # LoRA/QLoRA (best-effort, only if PEFT available)
    if lora_method and (apply_lora or apply_qlora):
        try:
            if lora_method == 'qlora' and apply_qlora is not None:
                model = apply_qlora(model)
            elif lora_method == 'lora' and apply_lora is not None:
                model = apply_lora(model)
            if rank == 0:
                print(f"Applied {lora_method} adapters")
        except Exception as e:
            if rank == 0:
                print(f"LoRA setup skipped: {e}")

    # Optimizer / Scheduler (non-DeepSpeed)
    if args.backend != 'deepspeed':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Handle streaming where len(train_loader) is None
        if max_steps > 0:
            total_steps = max_steps
        else:
            len_train = (len(train_loader) if train_loader is not None else 10000)
            total_steps = max(1, (len_train * epochs // max(1, grad_accum)))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        privacy_engine = None
        # Opacus compatibility is best with single-GPU DDP or single process
        if use_dp and (not use_streaming) and (args.backend == 'ddp') and (world_size == 1):
            model, optimizer, train_loader, privacy_engine = make_dp_optimizer(
                model,
                optimizer,
                train_loader,
                target_epsilon=float(train_cfg.get('dp_target_epsilon', 5.0)),
                max_grad_norm=float(train_cfg.get('dp_max_grad_norm', 1.0)),
                epochs=epochs,
            )
    else:
        optimizer = None
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == 'fp16'))

    # Resume checkpoint (best effort for non-DeepSpeed)
    if resume_from:
        try:
            if args.backend == 'deepspeed':
                # Expect resume_from like: /path/to/dir and optional tag in ckpt_cfg['tag']
                tag = ckpt_cfg.get('tag')
                model.load_checkpoint(resume_from, tag=tag)
            elif args.backend == 'fsdp':
                state = torch.load(resume_from, map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    model.load_state_dict(state['model'], strict=False)
                    if 'optimizer' in state and optimizer is not None and state['optimizer'] is not None:
                        optimizer.load_state_dict(state['optimizer'])
                    if 'scheduler' in state and scheduler is not None and state['scheduler'] is not None:
                        scheduler.load_state_dict(state['scheduler'])
                    if 'scaler' in state and scaler is not None and state['scaler'] is not None:
                        scaler.load_state_dict(state['scaler'])
                else:
                    model.load_state_dict(state, strict=False)
            else:
                state = torch.load(resume_from, map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    model.load_state_dict(state['model'], strict=False)
                    if 'optimizer' in state and optimizer is not None and state['optimizer'] is not None:
                        optimizer.load_state_dict(state['optimizer'])
                    if 'scheduler' in state and scheduler is not None and state['scheduler'] is not None:
                        scheduler.load_state_dict(state['scheduler'])
                    if 'scaler' in state and scaler is not None and state['scaler'] is not None:
                        scaler.load_state_dict(state['scaler'])
                else:
                    model.load_state_dict(state, strict=False)
            if rank == 0:
                print(f"Resumed from checkpoint: {resume_from}")
        except Exception as e:
            if rank == 0:
                print(f"Resume failed: {e}")

    # TensorBoard
    writer = None
    if use_tensorboard and SummaryWriter is not None and rank == 0:
        writer = SummaryWriter(log_dir=out_dir)

    # MLflow
    mlflow_run = None
    if use_mlflow and start_run is not None and rank == 0:
        mlflow_run = start_run(mlflow_experiment, mlflow_run_name)
        if log_params is not None:
            log_params({
                'architecture': args.architecture or 'transformer',
                'backend': args.backend,
                'batch_size': global_batch,
                'lr': lr,
                'mixed_precision': mixed_precision,
                'dataset': f"{train_name}:{train_subset}",
            })

    def evaluate_ppl(eval_source) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            if eval_loader is not None:
                iterator = iter(eval_loader)
            else:
                # Re-create fresh streaming iterator for eval
                iterator = stream_batches(eval_ds, micro_batch)
            # Evaluate up to N batches
            for i, batch in enumerate(iterator):
                # Support pairwise eval by using chosen texts
                if isinstance(batch, dict) and ('chosen' in batch):
                    ch = tokenizer(batch['chosen'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                    ch = {k: v.cuda(local_rank, non_blocking=True) for k, v in ch.items()}
                    ch['labels'] = ch['input_ids']
                    out = model(**ch)
                    if isinstance(out, dict) and out.get('loss') is not None:
                        losses.append(out['loss'].detach().float().item())
                else:
                    out = model(**batch)
                    if isinstance(out, dict) and out.get('loss') is not None:
                        losses.append(out['loss'].detach().float().item())
                if i >= 50:  # cap for speed
                    break
        model.train()
        if not losses:
            return float('inf')
        return math.exp(sum(losses) / len(losses))

    def sample_generations(prompts):
        outs = []
        model.eval()
        with torch.no_grad():
            for p in prompts:
                toks = tokenizer(p, return_tensors='pt').to(next(model.parameters()).device)
                try:
                    gen = model.generate(input_ids=toks['input_ids'], max_new_tokens=128)
                except Exception:
                    gen = model.core.generate(input_ids=toks['input_ids'], max_new_tokens=128) if hasattr(model, 'core') else None
                if gen is not None:
                    outs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        model.train()
        return outs

    # Training loop
    step = 0
    start_time = time.time()
    model.train()
    best_ppl = float('inf')
    no_improve = 0
    sample_prompts = log_cfg.get('sample_prompts', [])

    stop_training = False
    for epoch in range(epochs if max_steps == 0 else 10**9):
        if max_steps and step >= max_steps:
            break
        if train_loader is not None:
            # Advance distributed sampler epoch for proper shuffling
            try:
                if 'train_sampler' in locals() and train_sampler is not None:
                    train_sampler.set_epoch(epoch)
            except Exception:
                pass
            iterator = iter(train_loader)
        else:
            iterator = train_iter

        accum = 0
        for batch in iterator:
            if max_steps and step >= max_steps:
                break

            # Update curriculum step for collate
            state['step'] = step

            with torch.cuda.amp.autocast(enabled=(mixed_precision in ['fp16', 'bf16']), dtype=(torch.bfloat16 if mixed_precision == 'bf16' else torch.float16)):
                moe_info = None
                if args.align in ['orpo','cpo'] and (text_column == 'chosen' or data_cfg.get('pairwise', False)):
                    # Expect batch to have chosen/rejected
                    if 'chosen' in batch and 'rejected' in batch:
                        ch = tokenizer(batch['chosen'], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(next(model.parameters()).device)
                        rj = tokenizer(batch['rejected'], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(next(model.parameters()).device)
                        out_ch = model(**ch)
                        out_rj = model(**rj)
                        if isinstance(out_ch, dict) and 'moe_info' in out_ch:
                            moe_info = out_ch['moe_info']
                        def seq_logps(logits, ids, mask):
                            lp = torch.log_softmax(logits, dim=-1)
                            g = torch.gather(lp, -1, ids.unsqueeze(-1)).squeeze(-1)
                            g = g * mask
                            return g.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
                        lp_c = seq_logps(out_ch['logits'], ch['input_ids'], ch['attention_mask']).mean()
                        lp_r = seq_logps(out_rj['logits'], rj['input_ids'], rj['attention_mask']).mean()
                        if args.align == 'orpo':
                            loss = orpo_loss(lp_c, lp_r)
                        else:
                            loss = cpo_loss(lp_c, lp_r)
                    else:
                        out = model(**batch)
                        if isinstance(out, dict) and 'moe_info' in out:
                            moe_info = out['moe_info']
                        loss = out['loss'] if isinstance(out, dict) else out
                else:
                    out = model(**batch)
                    if isinstance(out, dict) and 'moe_info' in out:
                        moe_info = out['moe_info']
                    loss = out['loss'] if isinstance(out, dict) else out

            if args.backend == 'deepspeed':
                model.backward(loss)
                accum += 1
                if accum % grad_accum == 0:
                    model.step()
                    accum = 0
            else:
                scaler.scale(loss).backward()
                accum += 1
                if accum % grad_accum == 0:
                    # Unscale before clipping, then clip and step
                    max_gn = float(train_cfg.get('max_grad_norm', 1.0))
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    if writer is not None and rank == 0:
                        writer.add_scalar('train/grad_norm', float(total_norm.detach().float().item() if torch.is_tensor(total_norm) else total_norm), step)
                    accum = 0

            # Logging
            if step % log_interval == 0 and (rank == 0):
                stats = get_resource_stats()
                # tokens/sec (best-effort)
                tps = None
                try:
                    if isinstance(batch, dict) and 'labels' in batch:
                        tokens = int(batch['labels'].numel())
                        tps = tokens / max(1e-6, (time.time() - start_time) / max(1, step+1))
                except Exception:
                    pass
                log_obj = {'step': step, 'loss': float(loss.detach().float().item()), **stats}
                if tps is not None:
                    log_obj['tokens_per_sec'] = float(tps)
                print(log_obj)
                # Log MoE metrics if available
                try:
                    if moe_info and 'expert_utilization' in moe_info:
                        eu = moe_info['expert_utilization']
                        if writer is not None:
                            writer.add_scalar('moe/total_routing_entropy', eu.get('total_routing_entropy', 0.0), step)
                            writer.add_scalar('moe/avg_routing_entropy', eu.get('avg_routing_entropy', 0.0), step)
                        if log_metrics is not None and mlflow_run is not None:
                            log_metrics({'moe_total_entropy': float(eu.get('total_routing_entropy', 0.0)), 'moe_avg_entropy': float(eu.get('avg_routing_entropy', 0.0))}, step=step)
                except Exception:
                    pass

                # Learning rate
                current_lr = None
                try:
                    current_lr = float(optimizer.param_groups[0]['lr']) if optimizer is not None else None
                except Exception:
                    pass

                if writer is not None:
                    writer.add_scalar('train/loss', float(loss.detach().float().item()), step)
                    if current_lr is not None:
                        writer.add_scalar('train/lr', current_lr, step)
                    for k, v in stats.items():
                        writer.add_scalar(f'resources/{k}', v, step)
                if log_metrics is not None and mlflow_run is not None:
                    lm = {'train_loss': float(loss.detach().float().item())}
                    if current_lr is not None:
                        lm['lr'] = current_lr
                    log_metrics(lm, step=step)

            # Evaluation
            if eval_interval and (step % eval_interval == 0) and (step > 0) and (rank == 0):
                ppl = evaluate_ppl(eval_loader or eval_iter)
                print({'step': step, 'eval_ppl': ppl})
                if writer is not None:
                    writer.add_scalar('eval/ppl', ppl, step)
                if log_metrics is not None and mlflow_run is not None:
                    log_metrics({'eval_ppl': float(ppl)}, step=step)
                # Early stopping
                if ppl < best_ppl:
                    best_ppl = ppl
                    no_improve = 0
                    # Save best
                    if args.backend == 'fsdp':
                        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa
                        cpu_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cpu_cfg):
                            state = model.state_dict()
                        if rank == 0:
                            torch.save(state, os.path.join(out_dir, "model_best.pt"))
                    elif args.backend != 'deepspeed':
                        torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
                    else:
                        try:
                            model.save_checkpoint(out_dir, tag='best')
                        except Exception:
                            pass
                    # Sample
                    if sample_prompts:
                        gens = sample_generations(sample_prompts)
                        for i, g in enumerate(gens):
                            if writer is not None:
                                writer.add_text(f'samples/prompt_{i}', g, step)
                else:
                    no_improve += 1
                    if early_patience and no_improve >= early_patience:
                        print("Early stopping due to no improvement")
                        stop_training = True
                        break

            # Checkpoint
            if save_interval and (step % save_interval == 0) and (step > 0):
                tag = f"step-{step}"
                if args.backend == 'deepspeed':
                    try:
                        model.save_checkpoint(out_dir, tag=tag)
                    except Exception:
                        pass
                elif args.backend == 'fsdp':
                    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa
                    cpu_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cpu_cfg):
                        model_state = model.state_dict()
                    if rank == 0:
                        pkg = {
                            'model': model_state,
                            'optimizer': (optimizer.state_dict() if optimizer is not None else None),
                            'scheduler': (scheduler.state_dict() if scheduler is not None else None),
                            'scaler': (scaler.state_dict() if (mixed_precision=='fp16' and scaler is not None) else None),
                            'step': step,
                            'epoch': epoch,
                        }
                        torch.save(pkg, os.path.join(out_dir, f"model_{tag}.pt"))
                else:
                    if rank == 0:
                        pkg = {
                            'model': model.state_dict(),
                            'optimizer': (optimizer.state_dict() if optimizer is not None else None),
                            'scheduler': (scheduler.state_dict() if scheduler is not None else None),
                            'scaler': (scaler.state_dict() if (mixed_precision=='fp16' and scaler is not None) else None),
                            'step': step,
                            'epoch': epoch,
                        }
                        torch.save(pkg, os.path.join(out_dir, f"model_{tag}.pt"))

            step += 1

        if stop_training:
            break

    if rank == 0:
        print({
            'status': 'completed',
            'steps': step,
            'elapsed_min': (time.time() - start_time) / 60.0,
        })

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
