import math
import time
import torch
from typing import Dict, Tuple, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _is_deepspeed_engine(obj) -> bool:
    return hasattr(obj, "backward") and hasattr(obj, "step") and hasattr(obj, "module")


def train_one_epoch(
    *,
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    global_step: int,
    use_wandb: bool,
    is_main_process: bool,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    grad_norms = []

    start_time = time.time()
    last_step_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        step_start = time.time()

        use_dre_now = True
        if getattr(args, "dre_warmup_steps", 0) and global_step < args.dre_warmup_steps:
            use_dre_now = False

        if _is_deepspeed_engine(model):
            outputs = model(**batch, use_dre=use_dre_now)
            loss = outputs["loss"]
            model.backward(loss)
            model.step()
        else:
            # AMP path
            if scaler is not None:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                amp_enabled = True
                if getattr(args, "amp_warmup_steps", 0) and global_step < args.amp_warmup_steps:
                    amp_enabled = False
                with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                    outputs = model(**batch, use_dre=use_dre_now)
                    loss = outputs["loss"]
            else:
                outputs = model(**batch, use_dre=use_dre_now)
                loss = outputs["loss"]

            loss = loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.gradient_clipping > 0 and optimizer is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    params_for_norm = model.parameters() if not _is_deepspeed_engine(model) else model.module.parameters()
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        params_for_norm,
                        args.gradient_clipping,
                    )
                    if getattr(args, "log_grad_norm", False) and is_main_process:
                        try:
                            gnorm = float(total_norm.detach().cpu())
                        except Exception:
                            gnorm = float(total_norm)
                        print(f"[perf] step={global_step} grad_norm={gnorm:.4f}")
                        grad_norms.append(gnorm)
                if optimizer is not None:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                global_step += 1

        total_loss += float(loss.detach())
        num_batches += 1

        # Optional CUDA memory logging per interval
        if getattr(args, "log_cuda_memory", False) and torch.cuda.is_available():
            if (batch_idx + 1) % max(1, int(getattr(args, "perf_log_interval", 200))) == 0 and is_main_process:
                mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                mem_res = torch.cuda.memory_reserved() / (1024**2)
                print(f"[mem] step={global_step} alloc_mb={mem_alloc:.1f} reserved_mb={mem_res:.1f}")

        # Loss and performance logging per interval
        if (batch_idx + 1) % max(1, int(getattr(args, "perf_log_interval", 200))) == 0 and is_main_process:
            # Calculate current step loss (unscaled for gradient accumulation)
            step_loss = float(loss.detach()) * args.gradient_accumulation_steps
            try:
                # Calculate perplexity from loss
                perplexity = math.exp(min(step_loss, 20))  # Cap at 20 to avoid overflow
            except (OverflowError, ValueError):
                perplexity = float('inf')
            
            # Token throughput
            try:
                toks = int(batch['input_ids'].numel())
            except Exception:
                toks = 0
            step_time = time.time() - step_start
            toks_per_sec = toks / step_time if step_time > 0 and toks > 0 else 0.0
            
            # Extract MoE utilization metrics, auxiliary losses, and DRE metrics if available
            moe_metrics = {}
            dre_metrics = {}
            aux_loss_value = 0.0
            
            # DRE metrics extraction
            if hasattr(outputs, 'get') and outputs.get('routing_info'):
                routing_info = outputs['routing_info']
                if 'dre_metrics' in routing_info:
                    dre_info = routing_info['dre_metrics']
                    
                    # Key DRE metrics for console display
                    if 'avg_complexity' in dre_info:
                        dre_metrics['complexity'] = dre_info['avg_complexity']
                    if 'avg_confidence' in dre_info:
                        dre_metrics['confidence'] = dre_info['avg_confidence']
                    if 'path_distribution' in dre_info:
                        # Find most used path
                        path_dist = dre_info['path_distribution']
                        if path_dist:
                            most_used_path = max(path_dist.items(), key=lambda x: x[1])
                            dre_metrics['main_path'] = f"{most_used_path[0]}({most_used_path[1]:.0f}%)"
                    if 'cache_hit_rate' in dre_info:
                        dre_metrics['cache_hit'] = dre_info['cache_hit_rate']
            
            if hasattr(outputs, 'get') and outputs.get('moe_info'):
                moe_info = outputs['moe_info']
                
                # Expert utilization metrics
                if 'expert_utilization' in moe_info:
                    util = moe_info['expert_utilization']
                    
                    # Key metrics for console display
                    if 'avg_routing_entropy' in util:
                        moe_metrics['entropy'] = util['avg_routing_entropy']
                    
                    # Check for expert collapse (top expert getting >80% of traffic)
                    max_concentration = 0
                    for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
                        key = f"{expert_type}_top_expert_pct"
                        if key in util:
                            max_concentration = max(max_concentration, util[key])
                    
                    if max_concentration > 0:
                        moe_metrics['max_expert_pct'] = max_concentration
                
                # Auxiliary loss metrics
                if 'aux_losses' in moe_info:
                    aux_losses = moe_info['aux_losses']
                    total_aux = 0.0
                    for key, loss_val in aux_losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            # Handle both scalar and multi-element tensors
                            if loss_val.numel() == 1:
                                total_aux += float(loss_val.detach())
                            else:
                                total_aux += float(loss_val.detach().mean())
                    aux_loss_value = total_aux
                    moe_metrics['aux_loss'] = aux_loss_value
            
            # Log loss, perplexity, throughput, MoE and DRE metrics to console
            moe_str = ""
            if moe_metrics:
                moe_parts = []
                if 'entropy' in moe_metrics:
                    moe_parts.append(f"entropy={moe_metrics['entropy']:.2f}")
                if 'max_expert_pct' in moe_metrics:
                    moe_parts.append(f"max_exp={moe_metrics['max_expert_pct']:.1f}%")
                if 'aux_loss' in moe_metrics and moe_metrics['aux_loss'] > 0:
                    moe_parts.append(f"aux={moe_metrics['aux_loss']:.4f}")
                if moe_parts:
                    moe_str = f" moe=[{','.join(moe_parts)}]"
            
            dre_str = ""
            if dre_metrics:
                dre_parts = []
                if 'complexity' in dre_metrics:
                    dre_parts.append(f"comp={dre_metrics['complexity']:.2f}")
                if 'confidence' in dre_metrics:
                    dre_parts.append(f"conf={dre_metrics['confidence']:.2f}")
                if 'main_path' in dre_metrics:
                    dre_parts.append(f"path={dre_metrics['main_path']}")
                if dre_parts:
                    dre_str = f" dre=[{','.join(dre_parts)}]"
            
            print(f"[step] step={global_step} loss={step_loss:.4f} ppl={perplexity:.2f} toks/s={toks_per_sec:.1f}{moe_str}{dre_str}")
            
            # Log to MLflow if available and enabled
            if MLFLOW_AVAILABLE and getattr(args, "use_mlflow", False):
                try:
                    metrics = {
                        'train/step_loss': step_loss,
                        'train/step_perplexity': perplexity if perplexity != float('inf') else 1e10,
                        'train/tokens_per_sec': toks_per_sec,
                        'train/learning_rate': float(scheduler.get_last_lr()[0]) if scheduler is not None else 0.0,
                    }
                    
                    # Add detailed MoE metrics to MLflow
                    if hasattr(outputs, 'get') and outputs.get('moe_info'):
                        moe_info = outputs['moe_info']
                        if 'expert_utilization' in moe_info:
                            util = moe_info['expert_utilization']
                            
                            # Log all expert utilization metrics
                            for key, value in util.items():
                                if isinstance(value, (int, float)):
                                    metrics[f'moe/{key}'] = value
                                elif isinstance(value, list) and len(value) <= 10:  # Avoid logging huge lists
                                    for i, v in enumerate(value):
                                        metrics[f'moe/{key}_expert_{i}'] = v
                    
                    # Add detailed DRE metrics to MLflow
                    if hasattr(outputs, 'get') and outputs.get('routing_info'):
                        routing_info = outputs['routing_info']
                        if 'dre_metrics' in routing_info:
                            dre_info = routing_info['dre_metrics']
                            
                            # Log all DRE metrics
                            for key, value in dre_info.items():
                                if isinstance(value, (int, float)):
                                    metrics[f'dre/{key}'] = value
                                elif isinstance(value, dict):
                                    # Log path distribution
                                    for path_name, path_pct in value.items():
                                        metrics[f'dre/path_{path_name}'] = path_pct
                    
                    mlflow.log_metrics(metrics, step=global_step)
                except Exception as e:
                    pass  # Silent fail for MLflow logging
        last_step_time = time.time()

        # Profiler step if provided
        profiler = getattr(args, "_profiler", None)
        if profiler is not None:
            try:
                profiler.step()
            except Exception:
                pass

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)
    if is_main_process:
        # Calculate epoch-level perplexity
        try:
            avg_perplexity = math.exp(min(avg_loss, 20))
        except (OverflowError, ValueError):
            avg_perplexity = float('inf')
        
        toks = (len(train_loader.dataset) * getattr(args, "max_seq_length", 1)) if hasattr(train_loader, "dataset") else 0
        if toks:
            toks_per_sec = toks / max(1e-6, epoch_time)
        else:
            toks_per_sec = 0.0
        print(f"[train] avg_loss={avg_loss:.4f} avg_ppl={avg_perplexity:.2f} epoch_time={epoch_time:.1f}s toks/s={toks_per_sec:.1f}")
        
        # Log epoch summary to MLflow
        if MLFLOW_AVAILABLE and getattr(args, "use_mlflow", False):
            try:
                mlflow.log_metrics({
                    'train/epoch_avg_loss': avg_loss,
                    'train/epoch_avg_perplexity': avg_perplexity if avg_perplexity != float('inf') else 1e10,
                    'train/epoch_time_sec': epoch_time,
                }, step=global_step)
            except Exception:
                pass
        
        # Grad norm histogram summary
        if grad_norms:
            import numpy as np
            arr = np.array(grad_norms, dtype=float)
            q50, q90, q99 = np.percentile(arr, [50, 90, 99]).tolist()
            print(f"[grad_norm] p50={q50:.3f} p90={q90:.3f} p99={q99:.3f}")

    return avg_loss, global_step


def validate_epoch(*, model, val_loader, device, is_main_process: bool) -> float:
    model.eval()
    total_loss = 0.0
    num = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            step_loss = float(outputs["loss"].detach())
            total_loss += step_loss
            num += 1
            
            # Log validation progress every 50 steps
            if num % 50 == 0 and is_main_process:
                try:
                    step_ppl = math.exp(min(step_loss, 20))
                except (OverflowError, ValueError):
                    step_ppl = float('inf')
                print(f"[val_progress] batch={num} loss={step_loss:.4f} ppl={step_ppl:.2f}")
    
    avg = total_loss / max(1, num)
    if is_main_process:
        try:
            avg_ppl = math.exp(min(avg, 20))
        except (OverflowError, ValueError):
            avg_ppl = float('inf')
        print(f"[val] avg_loss={avg:.4f} avg_ppl={avg_ppl:.2f}")
        
        # Log validation results to MLflow
        # Note: We don't have global_step here, so we log without step parameter
        # MLflow will use the current run's step counter
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'val/avg_loss': avg,
                    'val/avg_perplexity': avg_ppl if avg_ppl != float('inf') else 1e10,
                })
            except Exception:
                pass
    return avg
