import math
import time
import torch
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

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
    total_raw_loss = 0.0
    num_batches = 0
    grad_norms = []
    
    # Track gradient norms at loop level for logging
    current_total_grad_norm = 0.0
    current_router_grad_norm = 0.0

    start_time = time.time()
    last_step_time = time.time()
    
    print(f"[DEBUG] Starting training loop, train_loader length estimate: {len(train_loader) if hasattr(train_loader, '__len__') else 'unknown'}")
    print(f"[DEBUG] gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    
    for batch_idx, batch in enumerate(train_loader):
        # DEBUG: Print EVERY batch to see loop execution
        print(f"[DEBUG] Batch {batch_idx}: global_step={global_step}")
        
        batch = {k: v.to(device) for k, v in batch.items()}
        step_start = time.time()

        use_dre_now = True
        if getattr(args, "dre_warmup_steps", 0) and global_step < args.dre_warmup_steps:
            use_dre_now = False
        
        # Debug: log the reasoning_path being passed
        force_path = getattr(args, "dre_force_path", None)
        if force_path:
            logger.info(f"[DEBUG] Training loop passing reasoning_path={force_path} to model")

        if _is_deepspeed_engine(model):
            outputs = model(
                **batch,
                use_dre=use_dre_now,
                reasoning_path=getattr(args, "dre_force_path", None)
            )
            loss = outputs["loss"]
            model.backward(loss)
            model.step()
        else:
            # AMP path
            raw_loss_item = None
            if scaler is not None:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                amp_enabled = True
                if getattr(args, "amp_warmup_steps", 0) and global_step < args.amp_warmup_steps:
                    amp_enabled = False
                with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                    outputs = model(
                        **batch,
                        use_dre=use_dre_now,
                        reasoning_path=getattr(args, "dre_force_path", None)
                    )
                    loss = outputs["loss"]
                    # Capture unscaled loss for accurate metrics
                    raw_loss_item = float(loss.detach())
            else:
                outputs = model(
                    **batch,
                    use_dre=use_dre_now,
                    reasoning_path=getattr(args, "dre_force_path", None)
                )
                loss = outputs["loss"]
                raw_loss_item = float(loss.detach())
            loss = loss / args.gradient_accumulation_steps
            
            # Skip if loss is not finite
            if not torch.isfinite(loss):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                continue
                
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Defer accurate gradient norm computation to accumulation boundary after unscale
            
            # Gradient clipping and optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if optimizer is not None:
                    # Gradient clipping
                    if args.gradient_clipping > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        # Compute accurate grad norms AFTER unscale and BEFORE clipping
                        current_total_grad_norm = 0.0
                        current_router_grad_norm = 0.0
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                current_total_grad_norm += param_norm.item() ** 2
                                if 'gate' in name or 'router' in name:
                                    current_router_grad_norm += param_norm.item() ** 2
                        current_total_grad_norm = current_total_grad_norm ** 0.5
                        current_router_grad_norm = current_router_grad_norm ** 0.5

                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                    
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # Accumulate unscaled (true) loss for epoch metrics
        if raw_loss_item is not None:
            total_raw_loss += raw_loss_item
        num_batches += 1

        # Optional CUDA memory logging per interval
        if getattr(args, "log_cuda_memory", False) and torch.cuda.is_available():
            if (batch_idx + 1) % max(1, int(getattr(args, "perf_log_interval", 200))) == 0 and is_main_process:
                mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                mem_res = torch.cuda.memory_reserved() / (1024**2)
                print(f"[mem] step={global_step} alloc_mb={mem_alloc:.1f} reserved_mb={mem_res:.1f}")

        # ALWAYS LOG after each gradient accumulation step; also log first micro-batch for verification
        should_log = ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (batch_idx == 0)
        print(f"[DEBUG] After batch {batch_idx}: (batch_idx+1)={batch_idx+1}, grad_accum={args.gradient_accumulation_steps}, should_log={should_log}")
        
        if should_log:
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
            
            # CRITICAL DEBUG: Always print to confirm logging is working
            print(f"[DEBUG] batch_idx={batch_idx}, global_step={global_step}, should_log={should_log}")
            if hasattr(outputs, 'keys'):
                print(f"[DEBUG] outputs keys: {list(outputs.keys())}")
            else:
                print(f"[DEBUG] outputs type: {type(outputs)}")

            # FIRST STEP VERIFICATION: dump MoE info structure
            if batch_idx == 0:
                print("\n=== FIRST STEP MoE VERIFICATION ===")
                try:
                    if hasattr(outputs, 'get') and 'moe_info' in outputs:
                        _mi = outputs['moe_info']
                        print(f"MoE info keys: {list(_mi.keys())}")
                        _aux = _mi.get('aux_losses', {})
                        if isinstance(_aux, dict):
                            print(f"Aux loss keys: {list(_aux.keys())[:10]}")
                        util = _mi.get('expert_utilization', {})
                        if isinstance(util, dict):
                            print(f"Expert util keys: {list(util.keys())[:10]}")
                    else:
                        print("ERROR: moe_info not in model outputs!")
                except Exception as _e:
                    print(f"[WARN] FIRST STEP VERIFICATION failed: {_e}")
            
            # DRE metrics extraction
            if hasattr(outputs, 'get') and outputs.get('routing_info'):
                routing_info = outputs['routing_info']
                # Current-step routing info
                if isinstance(routing_info, dict):
                    if 'path' in routing_info:
                        dre_metrics['path_now'] = routing_info['path']
                    if 'complexity_score' in routing_info:
                        try:
                            dre_metrics['comp_now'] = float(routing_info['complexity_score'])
                        except Exception:
                            pass
                    if 'confidence' in routing_info:
                        try:
                            dre_metrics['conf_now'] = float(routing_info['confidence'])
                        except Exception:
                            pass
                # Aggregated DRE metrics
                if 'dre_metrics' in routing_info:
                    dre_info = routing_info['dre_metrics']
                    
                    # Key DRE metrics for console display (averages)
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
                    
                    # Prefer entropy averaged over groups with >1 expert
                    num_used = moe_info.get('num_experts_used', {})
                    entropies = []
                    for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
                        if num_used.get(expert_type, 1) > 1:
                            ent_key = f"{expert_type}_entropy"
                            if ent_key in util:
                                try:
                                    entropies.append(float(util[ent_key]))
                                except Exception:
                                    pass
                    if entropies:
                        moe_metrics['entropy'] = float(sum(entropies) / max(1, len(entropies)))
                    elif 'avg_routing_entropy' in util:
                        # Fallback to overall average if no multi-expert groups exist
                        moe_metrics['entropy'] = util['avg_routing_entropy']
                    
                    # Max expert %: ignore single-expert groups to avoid false 100%
                    max_concentration = None
                    for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
                        if num_used.get(expert_type, 1) > 1:
                            key = f"{expert_type}_top_expert_pct"
                            if key in util:
                                val = float(util[key])
                                max_concentration = val if (max_concentration is None) else max(max_concentration, val)
                    # Fallback: if no multi-expert groups, use the existing aggregate (may be 100% by design)
                    if max_concentration is None:
                        tmp = 0.0
                        for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
                            key = f"{expert_type}_top_expert_pct"
                            if key in util:
                                tmp = max(tmp, float(util[key]))
                        max_concentration = tmp
                    if max_concentration > 0:
                        moe_metrics['max_expert_pct'] = max_concentration
                
                # Auxiliary loss metrics (detailed)
                if 'aux_losses' in moe_info:
                    aux_losses = moe_info['aux_losses']
                    total_aux = 0.0
                    lb, zl, imp, ent = 0.0, 0.0, 0.0, 0.0
                    for key, loss_val in aux_losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            val = float(loss_val.detach().mean()) if loss_val.numel() > 1 else float(loss_val.detach())
                        else:
                            try:
                                val = float(loss_val)
                            except Exception:
                                continue
                        total_aux += val
                        if 'load_loss' in key:
                            lb += val
                        elif 'z_loss' in key:
                            zl += val
                        elif 'importance_loss' in key:
                            imp += val
                        elif 'entropy_reg_loss' in key:
                            ent += val
                    aux_loss_value = total_aux
                    moe_metrics['aux_loss'] = aux_loss_value
                    # Expose detailed components
                    moe_metrics['load_balance'] = lb
                    moe_metrics['z_loss'] = zl
                    moe_metrics['importance'] = imp
                    moe_metrics['entropy_reg'] = ent
            
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
                if 'load_balance' in moe_metrics:
                    moe_parts.append(f"lb={moe_metrics['load_balance']:.4f}")
                if 'z_loss' in moe_metrics:
                    moe_parts.append(f"z={moe_metrics['z_loss']:.4f}")
                if 'importance' in moe_metrics:
                    moe_parts.append(f"imp={moe_metrics['importance']:.4f}")
                if 'entropy_reg' in moe_metrics:
                    moe_parts.append(f"ent_reg={moe_metrics['entropy_reg']:.4f}")
                # Whether MoE was actually used this step
                try:
                    used_moe_flag = False
                    if hasattr(outputs, 'get') and outputs.get('routing_info'):
                        ri = outputs['routing_info']
                        used_moe_flag = bool(ri.get('used_moe', False)) if isinstance(ri, dict) else False
                    moe_parts.append(f"used_moe={str(used_moe_flag)}")
                except Exception:
                    pass
                if moe_parts:
                    moe_str = f" moe=[{','.join(moe_parts)}]"
            
            dre_str = ""
            if dre_metrics:
                dre_parts = []
                # Prefer current-step metrics when available
                if 'comp_now' in dre_metrics:
                    dre_parts.append(f"comp={dre_metrics['comp_now']:.2f}")
                elif 'complexity' in dre_metrics:
                    dre_parts.append(f"comp_avg={dre_metrics['complexity']:.2f}")
                if 'conf_now' in dre_metrics:
                    dre_parts.append(f"conf={dre_metrics['conf_now']:.2f}")
                elif 'confidence' in dre_metrics:
                    dre_parts.append(f"conf_avg={dre_metrics['confidence']:.2f}")
                if 'path_now' in dre_metrics:
                    dre_parts.append(f"path={dre_metrics['path_now']}")
                elif 'main_path' in dre_metrics:
                    dre_parts.append(f"path_avg={dre_metrics['main_path']}")
                if dre_parts:
                    dre_str = f" dre=[{','.join(dre_parts)}]"
            
            # Add gradient norm to output
            grad_str = ""
            if current_total_grad_norm > 0:
                grad_str = f" grad=[total={current_total_grad_norm:.3f}"
                if current_router_grad_norm > 0:
                    grad_str += f",router={current_router_grad_norm:.3f}"
                grad_str += "]"
            
            print(f"[step] step={global_step} loss={step_loss:.4f} ppl={perplexity:.2f} toks/s={toks_per_sec:.1f}{moe_str}{dre_str}{grad_str}")
            
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
                        # Also log aux loss components if we computed them into moe_metrics
                        if 'aux_loss' in moe_metrics:
                            metrics['moe/aux_loss_total'] = float(moe_metrics['aux_loss'])
                        if 'load_balance' in moe_metrics:
                            metrics['moe/load_balance_loss'] = float(moe_metrics['load_balance'])
                        if 'z_loss' in moe_metrics:
                            metrics['moe/z_loss'] = float(moe_metrics['z_loss'])
                        if 'importance' in moe_metrics:
                            metrics['moe/importance_loss'] = float(moe_metrics['importance'])
                        if 'entropy_reg' in moe_metrics:
                            metrics['moe/entropy_reg_loss'] = float(moe_metrics['entropy_reg'])
                        # used_moe flag
                        try:
                            used_flag = 0
                            if hasattr(outputs, 'get') and outputs.get('routing_info'):
                                ri = outputs['routing_info']
                                used_flag = 1 if bool(ri.get('used_moe', False)) else 0
                            metrics['moe/used_moe'] = used_flag
                        except Exception:
                            pass
                    
                    # Add detailed DRE metrics to MLflow
                    if hasattr(outputs, 'get') and outputs.get('routing_info'):
                        routing_info = outputs['routing_info']
                        # Current step values
                        if isinstance(routing_info, dict):
                            if 'path' in routing_info:
                                metrics['dre/path_now_flag'] = {
                                    'fast': 0, 'standard': 0, 'expert': 0, 'deep': 0, 'ultra_deep': 0
                                }.get(str(routing_info['path']).lower(), 0)
                            if 'complexity_score' in routing_info:
                                metrics['dre/complexity_now'] = float(routing_info['complexity_score'])
                            if 'confidence' in routing_info:
                                metrics['dre/confidence_now'] = float(routing_info['confidence'])
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
    avg_loss = total_raw_loss / max(1, num_batches)
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
