"""
Fused Operations with Apex/Transformer-Engine fallbacks
Megatron-LM compatible fused kernels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try importing Apex fused ops
try:
    from apex.normalization import FusedLayerNorm, FusedRMSNorm
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    FusedLayerNorm = None
    FusedRMSNorm = None

# Try importing Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    te = None


class FusedLayerNormAffineFunction(torch.autograd.Function):
    """Fused LayerNorm with affine transform (fallback implementation)"""
    
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        
        # Compute mean and variance
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        # Affine transform
        output = weight * normalized + bias
        
        # Save for backward
        ctx.save_for_backward(input, weight, mean, var)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, var = ctx.saved_tensors
        eps = ctx.eps
        
        # Compute gradients
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        grad_weight = (grad_output * normalized).sum(dim=tuple(range(grad_output.ndim - 1)))
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        
        # Input gradient
        grad_normalized = grad_output * weight
        grad_var = (grad_normalized * (input - mean) * -0.5 * torch.pow(var + eps, -1.5)).sum(-1, keepdim=True)
        grad_mean = (grad_normalized * -1.0 / torch.sqrt(var + eps)).sum(-1, keepdim=True)
        grad_mean = grad_mean + grad_var * (input - mean).mean(-1, keepdim=True) * -2.0
        
        grad_input = grad_normalized / torch.sqrt(var + eps)
        grad_input = grad_input + grad_var * 2.0 * (input - mean) / input.size(-1)
        grad_input = grad_input + grad_mean / input.size(-1)
        
        return grad_input, grad_weight, grad_bias, None, None


def get_fused_layer_norm(normalized_shape: int, eps: float = 1e-5, use_apex: bool = True):
    """Get fused LayerNorm if available, else fallback"""
    if use_apex and APEX_AVAILABLE and FusedLayerNorm is not None:
        try:
            return FusedLayerNorm(normalized_shape, eps=eps)
        except Exception as e:
            logger.warning(f"Failed to create Apex FusedLayerNorm: {e}, falling back")
    
    # Fallback to PyTorch LayerNorm
    return nn.LayerNorm(normalized_shape, eps=eps)


def get_fused_rms_norm(normalized_shape: int, eps: float = 1e-6, use_apex: bool = True):
    """Get fused RMSNorm if available, else fallback"""
    if use_apex and APEX_AVAILABLE and FusedRMSNorm is not None:
        try:
            return FusedRMSNorm(normalized_shape, eps=eps)
        except Exception as e:
            logger.warning(f"Failed to create Apex FusedRMSNorm: {e}, falling back")
    
    # Fallback to custom RMSNorm
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
        
        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    
    return RMSNorm(normalized_shape, eps=eps)


class BiasDropoutAddFusion(nn.Module):
    """Fused bias + dropout + residual add"""
    
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Fused bias + dropout + add"""
        # Add bias
        if bias is not None:
            x = x + bias
        
        # Dropout
        if training and self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=training)
        
        # Residual add
        return x + residual


def bias_dropout_add_fused(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    dropout_prob: float,
    training: bool = True,
) -> torch.Tensor:
    """Fused bias + dropout + residual add operation"""
    if bias is not None:
        x = x + bias
    
    if training and dropout_prob > 0:
        x = F.dropout(x, p=dropout_prob, training=training)
    
    return x + residual


class TransformerEngineLinear(nn.Module):
    """Transformer Engine Linear layer with FP8 support"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8 and TE_AVAILABLE
        
        if self.use_fp8 and te is not None:
            try:
                self.linear = te.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                )
                self._is_te = True
            except Exception as e:
                logger.warning(f"Failed to create TE Linear: {e}, falling back")
                self.linear = nn.Linear(in_features, out_features, bias=bias)
                self._is_te = False
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self._is_te = False
    
    def forward(self, x):
        return self.linear(x)


class FusedScaleMaskSoftmax(nn.Module):
    """Fused scale + mask + softmax for attention"""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused scale, mask, and softmax"""
        # Scale
        input = input * self.scale
        
        # Mask
        if mask is not None:
            input = input + mask
        
        # Softmax
        return F.softmax(input, dim=-1)


def get_fp8_recipe(margin: int = 0, fp8_format: str = "HYBRID"):
    """Get FP8 recipe for Transformer Engine"""
    if not TE_AVAILABLE or recipe is None:
        return None
    
    try:
        return recipe.DelayedScaling(
            margin=margin,
            fp8_format=getattr(recipe.Format, fp8_format, recipe.Format.HYBRID),
        )
    except Exception as e:
        logger.warning(f"Failed to create FP8 recipe: {e}")
        return None


class FusedAdamW(torch.optim.Optimizer):
    """Fused AdamW optimizer (fallback to standard if Apex unavailable)"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Try to use Apex fused optimizer
        self.use_fused = False
        if APEX_AVAILABLE:
            try:
                from apex.optimizers import FusedAdam
                self._fused_impl = FusedAdam
                self.use_fused = True
            except ImportError:
                pass
    
    @torch.no_grad()
    def step(self, closure=None):
        """Optimizer step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
