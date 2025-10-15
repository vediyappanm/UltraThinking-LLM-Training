import logging
import math
import time
import torch
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
    total_loss = 0.0
    num_batches = 0
    grad_norms = []
    
    # Track gradient norms at loop level for logging
    current_total_grad_norm = 0.0
    current_router_grad_norm = 0.0

    start_time = time.time()
    last_step_time = time.time()
    
    logger.debug("Starting training loop, train_loader length estimate: %s", len(train_loader) if hasattr(train_loader, '__len__') else 'unknown')
    logger.debug("gradient_accumulation_steps: %s", args.gradient_accumulation_steps)
    
    for batch_idx, batch in enumerate(train_loader):
        measured_grad_this_step = False
        # DEBUG: per-batch visibility (hidden when logger level is INFO)
        logger.debug("Batch %d: global_step=%d", batch_idx, global_step)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        step_start = time.time()

        use_dre_now = True
        if getattr(args, "dre_warmup_steps", 0) and global_step < args.dre_warmup_steps:
            use_dre_now = False

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
            else:
                outputs = model(
                    **batch,
                    use_dre=use_dre_now,
                    reasoning_path=getattr(args, "dre_force_path", None)
                )
                loss = outputs["loss"]
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
            
            # Defer grad norm measurement until after clipping at step boundary
            current_router_grad_norm = 0.0
            
            # Gradient clipping and optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if optimizer is not None:
                    # Gradient clipping (scale threshold by accumulation factor)
                    if args.gradient_clipping > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        effective_max_norm = float(args.gradient_clipping) * float(args.gradient_accumulation_steps)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), effective_max_norm)
                        # Measure AFTER clipping for accurate logging
                        total_sq = 0.0
                        router_sq = 0.0
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                pn = float(param.grad.data.norm(2).item())
                                total_sq += pn * pn
                                if 'gate' in name or 'router' in name:
                                    router_sq += pn * pn
                        current_total_grad_norm = total_sq ** 0.5
                        current_router_grad_norm = router_sq ** 0.5
                        if current_total_grad_norm > effective_max_norm * 1.01:
                            print(f"[ERROR] Clipping failed! norm={current_total_grad_norm:.2f} max={effective_max_norm:.2f}")
                        measured_grad_this_step = True

                    # Diagnostics: Gradient norms by component (first 10 steps)
                    if global_step < 10:
                        try:
                            comp_sq = {
                                'embedding': 0.0,
                                'attention': 0.0,
                                'experts': 0.0,
                                'router': 0.0,
                                'output': 0.0,
                            }
                            expert_param_grads = []
                            router_param_norms = []
                            for name, param in model.named_parameters():
                                if param.grad is None:
                                    continue
                                g = float(param.grad.data.norm(2).item())
                                lname = name.lower()
                                if ('embedding' in lname) or ('wte' in lname) or ('wpe' in lname):
                                    comp_sq['embedding'] += g * g
                                elif ('router' in lname) or ('gate' in lname):
                                    comp_sq['router'] += g * g
                                elif ('expert' in lname) or ('experts' in lname) or ('moe' in lname):
                                    comp_sq['experts'] += g * g
                                elif ('attn' in lname) or ('attention' in lname):
                                    comp_sq['attention'] += g * g
                                elif ('lm_head' in lname) or ('output' in lname):
                                    comp_sq['output'] += g * g
                                # Collect expert grads for top listing
                                if (('expert' in lname) or ('experts' in lname)) and g > 0:
                                    expert_param_grads.append((name, g))
                                # Collect router param norms
                                if (('router' in lname) or ('gate' in lname)):
                                    try:
                                        router_param_norms.append((name, float(param.data.norm(2).item())))
                                    except Exception:
                                        pass
                            comp_norms = {k: (v ** 0.5) for k, v in comp_sq.items()}
                            total_comp_norm = (sum(v for v in comp_sq.values())) ** 0.5
                            print("[diag] grad_norms:",
                                  f"embedding={comp_norms['embedding']:.4f}",
                                  f"attention={comp_norms['attention']:.4f}",
                                  f"experts={comp_norms['experts']:.4f}",
                                  f"router={comp_norms['router']:.4f}",
                                  f"output={comp_norms['output']:.4f}",
                                  f"total={total_comp_norm:.4f}")
                            if expert_param_grads:
                                expert_param_grads.sort(key=lambda x: x[1], reverse=True)
                                topk = expert_param_grads[:10]
                                print("[diag] top_expert_grads:")
                                for n, g in topk:
                                    print(f"        {n}: {g:.6f}")
                            if router_param_norms:
                                router_param_norms.sort(key=lambda x: x[1], reverse=True)
                                topk_r = router_param_norms[:5]
                                print("[diag] router_param_norms:")
                                for n, pn in topk_r:
                                    print(f"        {n}: {pn:.6f}")
                            # Logit statistics if available
                            try:
                                if hasattr(outputs, 'get') and outputs.get('logits') is not None:
                                    logits = outputs['logits']
                                    m = float(logits.mean().detach().cpu().item())
                                    s = float(logits.std().detach().cpu().item())
                                    mn = float(logits.min().detach().cpu().item())
                                    mx = float(logits.max().detach().cpu().item())
                                    print(f"[diag] logits: mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}")
                                    if s < 0.1:
                                        print("[diag] WARN: logits std < 0.1 (low variance)")
                                    if abs(m) > 10.0:
                                        print("[diag] WARN: logits mean magnitude > 10 (extreme)")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1

        total_loss += float(loss.detach())
        num_batches += 1

        # Optional CUDA memory logging per interval
        if getattr(args, "log_cuda_memory", False) and torch.cuda.is_available():
            if (batch_idx + 1) % max(1, int(getattr(args, "perf_log_interval", 200))) == 0 and is_main_process:
                mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                mem_res = torch.cuda.memory_reserved() / (1024**2)
                print(f"[mem] step={global_step} alloc_mb={mem_alloc:.1f} reserved_mb={mem_res:.1f}")

        # ALWAYS LOG after each gradient accumulation step OR on first batch for visibility
        should_log = ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (batch_idx == 0)
        logger.debug("After batch %d: (batch_idx+1)=%d, grad_accum=%d, should_log=%s", batch_idx, batch_idx+1, args.gradient_accumulation_steps, should_log)
        
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
            
            # CRITICAL DEBUG: confirm logging is working (hidden at INFO level)
            logger.debug("batch_idx=%d, global_step=%d, should_log=%s", batch_idx, global_step, should_log)
            if hasattr(outputs, 'keys'):
                logger.debug("outputs keys: %s", list(outputs.keys()))
            else:
                logger.debug("outputs type: %s", type(outputs))
            
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
            # Capture MoE usage flag even if no moe_info metrics are present
            if hasattr(outputs, 'get') and outputs.get('routing_info'):
                try:
                    used_flag = bool(outputs['routing_info'].get('used_moe', False))
                    moe_metrics['used_moe'] = used_flag
                except Exception:
                    pass
            
            if hasattr(outputs, 'get') and outputs.get('moe_info'):
                moe_info = outputs['moe_info']
                
                # Expert utilization metrics
                if 'expert_utilization' in moe_info:
                    util = moe_info['expert_utilization']
                    
                    if 'avg_routing_entropy' in util:
                        moe_metrics['entropy'] = util['avg_routing_entropy']
                    
                    max_concentration = 0
                    max_concentration_multi = 0
                    max_ratio = 0.0
                    k_ratio = None
                    s_ratio = None
                    num_used = moe_info.get('num_experts_used', {})
                    for expert_type in ['knowledge', 'skill', 'meta', 'safety']:
                        key = f"{expert_type}_top_expert_pct"
                        if key in util:
                            val = util[key]
                            if val > max_concentration:
                                max_concentration = val
                            n = num_used.get(expert_type, 0)
                            if n and n > 1 and val > max_concentration_multi:
                                max_concentration_multi = val
                            # Compute ideal top expert percentage for this group
                            try:
                                n_int = int(n)
                            except Exception:
                                n_int = 0
                            if n_int > 0:
                                ideal_pct = 100.0 * float(min(getattr(args, 'moe_top_k', 2), n_int)) / float(n_int)
                                if ideal_pct > 0:
                                    ratio = float(val) / float(ideal_pct)
                                    if ratio > max_ratio:
                                        max_ratio = ratio
                                    if expert_type == 'knowledge':
                                        k_ratio = ratio
                                    if expert_type == 'skill':
                                        s_ratio = ratio
                    if max_concentration > 0:
                        moe_metrics['max_expert_pct'] = max_concentration
                    if max_concentration_multi > 0:
                        moe_metrics['max_expert_pct_multi'] = max_concentration_multi
                    if 'knowledge_top_expert_pct' in util:
                        moe_metrics['k_max'] = util['knowledge_top_expert_pct']
                    if 'skill_top_expert_pct' in util:
                        moe_metrics['s_max'] = util['skill_top_expert_pct']
                    if max_ratio > 0:
                        moe_metrics['max_exp_rel'] = max_ratio
                    if k_ratio is not None:
                        moe_metrics['k_rel'] = k_ratio
                    if s_ratio is not None:
                        moe_metrics['s_rel'] = s_ratio
                
                # Auxiliary loss metrics
                if 'aux_losses' in moe_info:
                    aux_losses = moe_info['aux_losses']
                    total_aux = 0.0
                    lb_total = 0.0
                    z_total = 0.0
                    imp_total = 0.0
                    ent_reg_total = 0.0
                    for key, loss_val in aux_losses.items():
                        # Normalize tensor to float
                        if isinstance(loss_val, torch.Tensor):
                            val = float(loss_val.detach().mean()) if loss_val.numel() > 1 else float(loss_val.detach())
                        else:
                            try:
                                val = float(loss_val)
                            except Exception:
                                val = 0.0
                        # Only include recognized loss terms in totals
                        if 'load_loss' in key:
                            lb_total += val
                            total_aux += val
                        elif 'z_loss' in key:
                            z_total += val
                            total_aux += val
                        elif 'importance_loss' in key:
                            imp_total += val
                            total_aux += val
                        elif 'entropy_reg_loss' in key:
                            ent_reg_total += val
                            total_aux += val
                    aux_loss_value = total_aux
                    moe_metrics['aux_loss'] = aux_loss_value
                    # Record aggregated components for detailed logging
                    if lb_total > 0:
                        moe_metrics['lb'] = lb_total
                    if z_total > 0:
                        moe_metrics['z'] = z_total
                    if imp_total > 0:
                        moe_metrics['imp'] = imp_total
                    if ent_reg_total > 0:
                        moe_metrics['ent_reg'] = ent_reg_total
            
            # Log loss, perplexity, throughput, MoE and DRE metrics to console
            moe_str = ""
            if moe_metrics:
                moe_parts = []
                if 'entropy' in moe_metrics:
                    moe_parts.append(f"entropy={moe_metrics['entropy']:.2f}")
                if 'max_expert_pct' in moe_metrics:
                    moe_parts.append(f"max_exp={moe_metrics['max_expert_pct']:.1f}%")
                if 'max_expert_pct_multi' in moe_metrics:
                    moe_parts.append(f"max_exp_multi={moe_metrics['max_expert_pct_multi']:.1f}%")
                if 'k_max' in moe_metrics:
                    moe_parts.append(f"k_max={moe_metrics['k_max']:.1f}%")
                if 's_max' in moe_metrics:
                    moe_parts.append(f"s_max={moe_metrics['s_max']:.1f}%")
                if 'max_exp_rel' in moe_metrics:
                    moe_parts.append(f"max_rel={moe_metrics['max_exp_rel']:.2f}x")
                if 'k_rel' in moe_metrics:
                    moe_parts.append(f"k_rel={moe_metrics['k_rel']:.2f}x")
                if 's_rel' in moe_metrics:
                    moe_parts.append(f"s_rel={moe_metrics['s_rel']:.2f}x")
                if 'aux_loss' in moe_metrics and moe_metrics['aux_loss'] > 0:
                    moe_parts.append(f"aux={moe_metrics['aux_loss']:.4f}")
                # Detailed aux components if available
                if 'lb' in moe_metrics:
                    moe_parts.append(f"lb={moe_metrics['lb']:.4f}")
                if 'z' in moe_metrics:
                    moe_parts.append(f"z={moe_metrics['z']:.4f}")
                if 'imp' in moe_metrics:
                    moe_parts.append(f"imp={moe_metrics['imp']:.4f}")
                if 'ent_reg' in moe_metrics:
                    moe_parts.append(f"ent_reg={moe_metrics['ent_reg']:.4f}")
                if 'used_moe' in moe_metrics:
                    moe_parts.append(f"used_moe={moe_metrics['used_moe']}")
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
            
            # Learning rate display
            try:
                curr_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else 0.0
            except Exception:
                curr_lr = 0.0
            print(f"[step] step={global_step} loss={step_loss:.4f} ppl={perplexity:.2f} toks/s={toks_per_sec:.1f} lr={curr_lr:.6g}{moe_str}{dre_str}{grad_str}")
            
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

    # If the epoch ended with leftover grads (not an exact multiple of grad_accum),
    # perform a final optimizer step so progress still increments.
    try:
        if (num_batches % max(1, int(getattr(args, "gradient_accumulation_steps", 1)))) != 0 and optimizer is not None and not _is_deepspeed_engine(model):
            # Gradient clipping
            if getattr(args, "gradient_clipping", 0) > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Emit a final step line using the last computed metric strings if available
            try:
                step_loss = float(loss.detach()) * args.gradient_accumulation_steps
                try:
                    perplexity = math.exp(min(step_loss, 20))
                except (OverflowError, ValueError):
                    perplexity = float('inf')
                # Reuse previously built strings if present; otherwise compute minimal ones
                if 'moe_str' not in locals():
                    moe_str = ""
                if 'dre_str' not in locals():
                    dre_str = ""
                if 'grad_str' not in locals():
                    grad_str = ""
                print(f"[step] step={global_step} loss={step_loss:.4f} ppl={perplexity:.2f} toks/s=0.0{moe_str}{dre_str}{grad_str}")
            except Exception:
                pass
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
