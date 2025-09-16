"""
Advanced Optimizers for Large Scale Training
Including Lion, AdamW variants, and SAM
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Any, Dict, Optional, Tuple, Union


class Lion(Optimizer):
    """
    Lion optimizer from "Symbolic Discovery of Optimization Algorithms"
    More memory efficient than AdamW for large models
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        maximize: bool = False,
        foreach: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []

            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.dtype in {torch.float16, torch.bfloat16}:
                    grads.append(p.grad.float())
                else:
                    grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])

            lion(
                params_with_grad,
                grads,
                exp_avgs,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
            )

        return loss


def lion(
    params,
    grads,
    exp_avgs,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
):
    """Functional API that performs Lion algorithm computation."""
    
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        param.add_(torch.sign(update), alpha=-lr)

        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


class AdamWScale(torch.optim.AdamW):
    """
    AdamW with learning rate scaling based on parameter norm
    Useful for very large models
    """
    
    def __init__(self, *args, scale_lr: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_lr = scale_lr
    
    def step(self, closure=None):
        if not self.scale_lr:
            return super().step(closure)
        
        # Scale learning rate based on parameter norm
        for group in self.param_groups:
            total_norm = 0.0
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.data.norm()
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Scale learning rate
            if total_norm > 0:
                scale = min(1.0, 1.0 / total_norm)
                group['lr'] = group['lr'] * scale
        
        return super().step(closure)


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer
    Improves generalization by finding flatter minima
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def state_dict(self):
        return super().state_dict()


class Sophia(Optimizer):
    """
    Sophia optimizer - Second-order clipped stochastic optimization
    More efficient than Adam for large language models
    """
    
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
            maximize=maximize, capturable=capturable
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        """Update Hessian diagonal approximation"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if group['capturable'] else torch.tensor(0.)
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    hessian = state['hessian']
                    beta2 = group['betas'][1]
                    
                    # Compute diagonal Hessian approximation
                    hessian.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            hessians = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.dtype in {torch.float16, torch.bfloat16}:
                    grads.append(p.grad.float())
                else:
                    grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if group['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                hessians.append(state['hessian'])
                state_steps.append(state['step'])

            sophia(
                params_with_grad,
                grads,
                exp_avgs,
                hessians,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                rho=group['rho'],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                capturable=group['capturable'],
            )

        return loss


def sophia(
    params,
    grads,
    exp_avgs,
    hessians,
    state_steps,
    capturable: bool = False,
    *,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
):
    """Functional API that performs Sophia algorithm computation."""
    
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hessian = hessians[i]
        step_t = state_steps[i]

        if capturable:
            bs = torch.ones_like(step_t) * 5120
            assert param.dtype == torch.float32

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Bias correction
        bias_correction1 = 1 - beta1 ** step_t.item()

        # Clipped update
        k = hessian.abs().clamp_(min=1e-8)
        u = (exp_avg / bias_correction1) / k.sqrt()
        u.clamp_(min=-rho, max=rho)
        
        param.add_(u, alpha=-lr)


def get_optimizer(model, config):
    """Get optimizer based on configuration"""
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'bias' in name or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=float(config.learning_rate),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
        )
    elif optimizer_name == "adamw_scale":
        optimizer = AdamWScale(
            optim_groups,
            lr=float(config.learning_rate),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
            scale_lr=True,
        )
    elif optimizer_name == "lion":
        optimizer = Lion(
            optim_groups,
            lr=float(config.learning_rate) * 0.3,  # Lion typically needs lower LR
            betas=(float(config.beta1), float(config.beta2)),
            weight_decay=float(config.weight_decay),
        )
    elif optimizer_name == "sophia":
        optimizer = Sophia(
            optim_groups,
            lr=float(config.learning_rate),
            betas=(float(config.beta1), float(config.beta2)),
            rho=0.04,
            weight_decay=float(config.weight_decay),
        )
    elif optimizer_name == "sam_adamw":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            optim_groups,
            base_optimizer,
            rho=0.05,
            adaptive=False,
            lr=float(config.learning_rate),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    scheduler_name = config.lr_scheduler.lower()
    
    if scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.total_steps - config.warmup_steps,
            eta_min=getattr(config, 'lr_scheduler_kwargs', {}).get('eta_min', 0),
        )
    elif scheduler_name == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.total_steps - config.warmup_steps,
        )
    elif scheduler_name == "polynomial":
        from torch.optim.lr_scheduler import PolynomialLR
        scheduler = PolynomialLR(
            optimizer,
            total_iters=config.total_steps - config.warmup_steps,
            power=getattr(config, 'lr_scheduler_kwargs', {}).get('power', 1.0),
        )
    else:
        return None
    
    # Add warmup if specified
    if config.warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[config.warmup_steps]
        )
    
    return scheduler
