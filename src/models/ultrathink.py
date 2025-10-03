"""
ULTRATHINK: The Complete GPT-5/Claude 4.1 Architecture
Combines all advanced components into a unified system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import sys
import os

# Import all components
from .architecture import AdvancedGPTModel, ModelConfig
from .dynamic_reasoning import DynamicReasoningEngine, ReasoningPath, ComplexityFeatures
from .constitutional_ai import ConstitutionalReasoningCore, HarmCategory
from .moe_advanced import MoELayer, ExpertConfig, HierarchicalMoE
from .multimodal import UnifiedMultiModalModel, MultiModalConfig, Modality
import numpy as np

logger = logging.getLogger(__name__)


@dataclass 
class UltraThinkConfig:
    """Complete configuration for ULTRATHINK model"""
    
    # Base model config
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Dynamic reasoning
    enable_dre: bool = True
    dre_paths: List[str] = field(default_factory=lambda: ["fast", "standard", "deep", "ultra_deep"])
    adaptive_routing: bool = True
    
    # Constitutional AI
    enable_constitutional: bool = True
    safety_threshold: float = 0.8
    constitutional_weight: float = 0.15
    
    # MoE configuration
    enable_moe: bool = True
    moe_config: ExpertConfig = field(default_factory=ExpertConfig)
    moe_layers: List[int] = field(default_factory=lambda: list(range(8, 64, 4)))
    
    # Multi-modal
    enable_multimodal: bool = True
    multimodal_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    supported_modalities: List[Modality] = field(default_factory=lambda: [
        Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.CODE, Modality.MATH
    ])
    
    # RLHF settings
    enable_rlhf: bool = True
    rlhf_objectives: List[str] = field(default_factory=lambda: [
        "helpfulness", "harmlessness", "honesty", "accuracy"
    ])
    
    # Training configuration
    batch_size: int = 32
    gradient_accumulation: int = 4
    learning_rate: float = 3e-5
    warmup_steps: int = 10000
    max_steps: int = 1000000
    
    # Memory and optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    # Make compilation opt-in to avoid AMP/MoE dtype issues on some platforms
    compile_model: bool = False
    
    # Inference settings
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1


class UltraThinkCore(nn.Module):
    """Core ULTRATHINK model architecture"""
    
    def __init__(self, config: UltraThinkConfig):
        super().__init__()
        
        self.config = config
        
        # Base transformer model
        self.base_model = AdvancedGPTModel(config.model_config)
        
        # Dynamic Reasoning Engine
        if config.enable_dre:
            self.dre = DynamicReasoningEngine(
                base_model=self.base_model,
                config={'hidden_dim': config.model_config.n_embd}
            )
        else:
            self.dre = None
        
        # Constitutional Reasoning Core
        if config.enable_constitutional:
            self.crc = ConstitutionalReasoningCore(
                base_model=self.base_model,
                config={
                    'hidden_dim': config.model_config.n_embd,
                    'safety_threshold': config.safety_threshold,
                    'constitutional_weight': config.constitutional_weight
                }
            )
        else:
            self.crc = None
        
        # MoE layers
        if config.enable_moe:
            self.moe_layers = nn.ModuleDict()
            for layer_idx in config.moe_layers:
                self.moe_layers[str(layer_idx)] = MoELayer(
                    config.moe_config,
                    config.model_config.n_embd,
                    config.model_config.intermediate_size
                )
        else:
            self.moe_layers = None
        
        # Multi-modal components
        if config.enable_multimodal:
            self.multimodal = UnifiedMultiModalModel(
                config.multimodal_config,
                self.base_model
            )
        else:
            self.multimodal = None
        
        # Output heads
        self.lm_head = nn.Linear(
            config.model_config.n_embd,
            config.model_config.vocab_size,
            bias=False
        )
        
        # Value head for RLHF
        if config.enable_rlhf:
            self.value_head = nn.Sequential(
                nn.Linear(config.model_config.n_embd, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.value_head = None
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[Dict[Modality, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_dre: Optional[bool] = None,
        enforce_safety: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through ULTRATHINK
        
        Args:
            input_ids: Text input token IDs
            inputs: Multi-modal inputs dict
            attention_mask: Attention mask
            labels: Target labels for training
            use_dre: Whether to use Dynamic Reasoning Engine
            enforce_safety: Whether to enforce constitutional safety
            return_dict: Return dictionary of outputs
        """
        
        # Determine if we should use DRE
        if use_dre is None:
            use_dre = self.config.enable_dre and self.dre is not None
        # Hard guard: only use DRE if the module exists
        use_dre = bool(use_dre) and (self.dre is not None)
        
        # Determine if we should enforce safety
        if enforce_safety is None:
            enforce_safety = self.config.enable_constitutional and self.crc is not None
        
        # Handle multi-modal inputs
        if self.multimodal and inputs:
            # Multi-modal forward pass
            mm_outputs = self.multimodal(
                inputs=inputs,
                labels=labels,
                primary_modality=Modality.TEXT,
                return_dict=True
            )
            hidden_states = mm_outputs['hidden_states']
            # Provide a minimal routing_info placeholder for UI
            routing_info = {
                'chosen_path': 'multimodal',
                'used_dre': False,
                'note': 'multimodal_path_no_dre'
            }
            
        elif input_ids is not None:
            # Text-only path
            
            # Dynamic reasoning routing
            if use_dre:
                # Extract text for complexity analysis (simplified)
                text = kwargs.get('text', '')
                
                # DRE forward pass
                dre_outputs = self.dre(
                    input_ids=input_ids,
                    text=text,
                    override_path=kwargs.get('reasoning_path'),
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                base_outputs = dre_outputs
                
            else:
                # Standard forward pass
                base_outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            hidden_states = base_outputs.get('hidden_states')
            # Create/normalize routing_info for UI visibility
            routing_info = None
            if isinstance(base_outputs, dict) and 'routing_info' in base_outputs:
                routing_info = base_outputs['routing_info']
            else:
                routing_info = {
                    'chosen_path': 'base',
                    'used_dre': False,
                    'note': 'dre_unavailable_or_no_routing_info'
                }
            total_aux_loss = 0
            moe_info = {}
            
            # Apply MoE layers only if available
            if self.moe_layers is not None:
                for layer_idx_str, moe_layer in self.moe_layers.items():
                    layer_idx = int(layer_idx_str)
                    
                    # Apply MoE at specific layers
                    if layer_idx < hidden_states.shape[1]:  # Check if layer exists
                        moe_output, aux_loss = moe_layer(hidden_states)
                        hidden_states = moe_output
                        
                        if aux_loss is not None:
                            total_aux_loss += aux_loss
                        
                        # Collect MoE info if available
                        if hasattr(moe_layer, 'moe') and hasattr(moe_layer.moe, 'forward'):
                            try:
                                # Get MoE info from the layer
                                _, layer_moe_info = moe_layer.moe(hidden_states, return_all_levels=True)
                                if layer_moe_info and 'expert_utilization' in layer_moe_info:
                                    moe_info = layer_moe_info  # Use the latest layer's info
                            except Exception:
                                pass  # Skip if MoE info extraction fails
        
        # Constitutional safety check (only if CRC is available)
        if enforce_safety and (self.crc is not None) and hidden_states is not None:
            crc_outputs = self.crc(
                input_ids=input_ids if input_ids is not None else inputs[Modality.TEXT],
                labels=labels,
                generate_critique=True,
                enforce_safety=True,
                hidden_states=hidden_states
            )
            
            # Update outputs with safety info
            if 'constitutional_info' in crc_outputs:
                constitutional_info = crc_outputs['constitutional_info']
            else:
                constitutional_info = None
                
            # Use revised hidden states if available
            if 'revised_hidden_states' in crc_outputs:
                hidden_states = crc_outputs['revised_hidden_states']
        else:
            constitutional_info = None
        
        # Generate logits
        if hidden_states is not None:
            if len(hidden_states.shape) == 2:
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states)
        else:
            logits = None
        
        # Calculate value for RLHF
        value = None
        if self.value_head and hidden_states is not None:
            if len(hidden_states.shape) == 3:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = hidden_states
            value = self.value_head(pooled).squeeze(-1)
        
        # Calculate loss
        loss = None
        if labels is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary losses
            loss = lm_loss
            
            if self.moe_layers and 'total_aux_loss' in locals():
                loss = loss + self.config.moe_config.aux_loss_weight * total_aux_loss
            
            if constitutional_info and 'constitutional_loss' in constitutional_info:
                loss = loss + self.config.constitutional_weight * constitutional_info['constitutional_loss']
        
        if not return_dict:
            return logits
        
        # Compile comprehensive output
        outputs = {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'value': value,
        }
        
        # Add component-specific outputs
        if 'routing_info' in locals() and routing_info is not None:
            outputs['routing_info'] = routing_info
        
        if constitutional_info:
            outputs['constitutional_info'] = constitutional_info
        
        if self.moe_layers:
            outputs['moe_aux_loss'] = total_aux_loss if 'total_aux_loss' in locals() else None
            if 'moe_info' in locals() and moe_info:
                outputs['moe_info'] = moe_info
        
        return outputs


class UltraThinkModel(nn.Module):
    """Complete ULTRATHINK model with all systems integrated"""
    
    def __init__(self, config: UltraThinkConfig):
        super().__init__()
        
        self.config = config
        
        # Core model
        self.core = UltraThinkCore(config)
        
        # Compile model for better performance if requested (disabled on Windows and when TORCHDYNAMO_DISABLE=1)
        can_compile = (
            config.compile_model
            and not sys.platform.startswith('win')
            and os.environ.get('TORCHDYNAMO_DISABLE', '') != '1'
        )
        if can_compile:
            self.core = torch.compile(self.core)
        
        # Generation config
        self.generation_config = {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'repetition_penalty': config.repetition_penalty,
            'do_sample': True,
            'pad_token_id': 0,
            'eos_token_id': 2,
        }
        # Storage for last generation's reasoning trace (for UI debugging)
        self.last_reasoning = None
        
    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.core(*args, **kwargs)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[Dict[Modality, torch.Tensor]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_dre: Optional[bool] = None,
        reasoning_path: Optional[ReasoningPath] = None,
        enforce_safety: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with ULTRATHINK
        
        Args:
            input_ids: Input token IDs for text
            inputs: Multi-modal inputs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_path: Force specific reasoning path
            enforce_safety: Enforce constitutional safety
        """
        
        self.eval()
        
        # Update generation config
        gen_config = self.generation_config.copy()
        if max_new_tokens is not None:
            gen_config['max_new_tokens'] = max_new_tokens
        if temperature is not None:
            gen_config['temperature'] = temperature
        
        # Prepare inputs
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            generated = input_ids
        elif inputs and Modality.TEXT in inputs:
            generated = inputs[Modality.TEXT]
            batch_size = generated.shape[0]
            device = generated.device
        else:
            raise ValueError("No input provided for generation")
        
        # Generation loop
        reasoning_trace = []
        for _ in range(gen_config['max_new_tokens']):
            # Get model outputs
            outputs = self.core(
                input_ids=generated if input_ids is not None else None,
                inputs=inputs if inputs else None,
                use_dre=use_dre,
                reasoning_path=reasoning_path,
                enforce_safety=enforce_safety,
                use_cache=True,
                return_dict=True
            )
            
            logits = outputs['logits']
            # Collect routing info if available
            if outputs is not None and isinstance(outputs, dict) and 'routing_info' in outputs:
                reasoning_trace.append(outputs['routing_info'])
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if gen_config['temperature'] > 0:
                next_token_logits = next_token_logits / gen_config['temperature']
            
            # Apply top-k filtering
            if gen_config['top_k'] > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, gen_config['top_k'])[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if gen_config['top_p'] < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > gen_config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply repetition penalty
            if gen_config['repetition_penalty'] != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= gen_config['repetition_penalty']
            
            # Sample next token
            if gen_config['do_sample']:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS token
            if (next_token == gen_config['eos_token_id']).all():
                break
        
        # Expose reasoning to callers (e.g., UI)
        self.last_reasoning = reasoning_trace
        return generated
    
    def save_pretrained(self, save_path: str):
        """Save model and configuration"""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.core.state_dict(), os.path.join(save_path, 'model.pt'))
        
        # Save configuration
        config_dict = {
            'model_config': self.config.model_config.__dict__,
            'moe_config': self.config.moe_config.__dict__ if self.config.enable_moe else None,
            'multimodal_config': self.config.multimodal_config.__dict__ if self.config.enable_multimodal else None,
            'ultrathink_config': {
                k: v for k, v in self.config.__dict__.items()
                if k not in ['model_config', 'moe_config', 'multimodal_config']
            }
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load model from saved checkpoint"""
        import os
        import json
        
        # Load configuration
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct configuration
        model_config = ModelConfig(**config_dict['model_config'])
        
        config = UltraThinkConfig()
        config.model_config = model_config
        
        if config_dict.get('moe_config'):
            config.moe_config = ExpertConfig(**config_dict['moe_config'])
        
        if config_dict.get('multimodal_config'):
            config.multimodal_config = MultiModalConfig(**config_dict['multimodal_config'])
        
        for k, v in config_dict['ultrathink_config'].items():
            if hasattr(config, k):
                setattr(config, k, v)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        raw_state_dict = torch.load(os.path.join(load_path, 'model.pt'), map_location='cpu')

        # Handle torch.compile wrapping which prefixes keys with '_orig_mod.'
        cleaned_state_dict = {}
        remapped = 0
        for k, v in raw_state_dict.items():
            new_k = k
            if k.startswith('_orig_mod.'):
                new_k = k[len('_orig_mod.') :]
                remapped += 1
            cleaned_state_dict[new_k] = v

        # Filter keys to only those present in the target to avoid unexpected keys (e.g., base_model.lm_head)
        target_keys = set(model.core.state_dict().keys())
        filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in target_keys}
        dropped = len(cleaned_state_dict) - len(filtered_state_dict)

        missing_before = [k for k in target_keys if k not in filtered_state_dict]
        if remapped or dropped or missing_before:
            logger.info(
                f"from_pretrained: remapped {remapped} keys, dropped {dropped} extraneous keys; "
                f"will load with strict=False (missing={len(missing_before)})."
            )

        model.core.load_state_dict(filtered_state_dict, strict=False)
        
        logger.info(f"Model loaded from {load_path}")
        
        return model
