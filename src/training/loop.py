import math
import time
import torch
from typing import Dict, Tuple, Optional


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

        # Tokens/sec logging per iteration (approx)
        if (batch_idx + 1) % max(1, int(getattr(args, "perf_log_interval", 200))) == 0 and is_main_process:
            try:
                toks = int(batch['input_ids'].numel())
            except Exception:
                toks = 0
            step_time = time.time() - step_start
            if step_time > 0 and toks > 0:
                print(f"[speed] step={global_step} toks={toks} toks/s={toks/step_time:.1f}")
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
        toks = (len(train_loader.dataset) * getattr(args, "max_seq_length", 1)) if hasattr(train_loader, "dataset") else 0
        if toks:
            toks_per_sec = toks / max(1e-6, epoch_time)
        else:
            toks_per_sec = 0.0
        print(f"[train] avg_loss={avg_loss:.4f} epoch_time={epoch_time:.1f}s toks/s={toks_per_sec:.1f}")
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
            total_loss += float(outputs["loss"].detach())
            num += 1
    avg = total_loss / max(1, num)
    if is_main_process:
        print(f"[val] avg_loss={avg:.4f}")
    return avg
