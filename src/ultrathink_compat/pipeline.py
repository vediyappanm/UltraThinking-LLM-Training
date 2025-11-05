"""
Pipeline Parallel Engine with 1F1B and Interleaved Schedules
Megatron-LM compatible pipeline parallelism
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineSchedule(Enum):
    """Pipeline schedule types"""
    GPIPE = "gpipe"  # All forward, then all backward
    ONE_F_ONE_B = "1f1b"  # 1 forward 1 backward interleaved
    INTERLEAVED = "interleaved"  # Interleaved 1F1B with virtual stages


@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration"""
    num_stages: int = 1
    num_microbatches: int = 4
    schedule: str = "1f1b"
    virtual_pipeline_size: int = 1  # For interleaved schedule
    enable_recompute: bool = False
    scatter_gather_tensors: bool = True


class PipelineStage(nn.Module):
    """Single pipeline stage wrapper"""
    
    def __init__(self, module: nn.Module, stage_id: int, num_stages: int):
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)
    
    def forward(self, hidden_states, **kwargs):
        return self.module(hidden_states, **kwargs)


class PipelineEngine:
    """Pipeline parallel training engine"""
    
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        loss_fn: Optional[callable] = None,
    ):
        self.config = config
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Split model into stages
        self.stages = self._partition_model(model)
        
        # Get pipeline parallel group
        self.pp_group = self._get_pipeline_parallel_group()
        self.pp_rank = dist.get_rank(self.pp_group) if self.pp_group else 0
        self.pp_size = dist.get_world_size(self.pp_group) if self.pp_group else 1
        
        # Current stage
        self.stage = self.stages[self.pp_rank] if self.pp_rank < len(self.stages) else None
        
        # Communication buffers
        self.input_tensors = []
        self.output_tensors = []
        
    def _partition_model(self, model: nn.Module) -> List[PipelineStage]:
        """Partition model into pipeline stages"""
        if not hasattr(model, 'layers'):
            # Fallback: wrap entire model as single stage
            return [PipelineStage(model, 0, 1)]
        
        layers = list(model.layers)
        num_layers = len(layers)
        layers_per_stage = num_layers // self.config.num_stages
        
        stages = []
        for stage_id in range(self.config.num_stages):
            start_idx = stage_id * layers_per_stage
            end_idx = start_idx + layers_per_stage if stage_id < self.config.num_stages - 1 else num_layers
            
            stage_layers = nn.Sequential(*layers[start_idx:end_idx])
            stages.append(PipelineStage(stage_layers, stage_id, self.config.num_stages))
        
        return stages
    
    def _get_pipeline_parallel_group(self):
        """Get pipeline parallel process group"""
        if not dist.is_available() or not dist.is_initialized():
            return None
        # Assume pipeline group is the default group for now
        return dist.group.WORLD
    
    def _send_forward(self, tensor: torch.Tensor):
        """Send activation to next stage"""
        if self.pp_rank < self.pp_size - 1:
            dist.send(tensor, dst=self.pp_rank + 1, group=self.pp_group)
    
    def _recv_forward(self) -> torch.Tensor:
        """Receive activation from previous stage"""
        if self.pp_rank > 0:
            # Get shape from first microbatch (assume same shape)
            tensor = torch.empty_like(self.input_tensors[0]) if self.input_tensors else None
            if tensor is None:
                raise RuntimeError("Cannot receive forward without knowing tensor shape")
            dist.recv(tensor, src=self.pp_rank - 1, group=self.pp_group)
            return tensor
        return None
    
    def _send_backward(self, tensor: torch.Tensor):
        """Send gradient to previous stage"""
        if self.pp_rank > 0:
            dist.send(tensor, dst=self.pp_rank - 1, group=self.pp_group)
    
    def _recv_backward(self) -> torch.Tensor:
        """Receive gradient from next stage"""
        if self.pp_rank < self.pp_size - 1:
            tensor = torch.empty_like(self.output_tensors[-1]) if self.output_tensors else None
            if tensor is None:
                raise RuntimeError("Cannot receive backward without knowing tensor shape")
            dist.recv(tensor, src=self.pp_rank + 1, group=self.pp_group)
            return tensor
        return None
    
    def forward_backward_1f1b(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """1F1B pipeline schedule"""
        num_microbatches = self.config.num_microbatches
        microbatch_size = input_ids.size(0) // num_microbatches
        
        # Split into microbatches
        input_microbatches = torch.chunk(input_ids, num_microbatches, dim=0)
        label_microbatches = torch.chunk(labels, num_microbatches, dim=0) if labels is not None else [None] * num_microbatches
        
        losses = []
        
        # Warmup phase: fill pipeline
        num_warmup = min(self.pp_rank + 1, num_microbatches)
        for i in range(num_warmup):
            input_tensor = input_microbatches[i] if self.stage.is_first_stage else self._recv_forward()
            
            # Forward
            output_tensor = self.stage(input_tensor)
            
            if self.stage.is_last_stage:
                # Compute loss
                loss = self.loss_fn(output_tensor, label_microbatches[i])
                losses.append(loss)
            else:
                self._send_forward(output_tensor)
            
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
        
        # 1F1B steady state
        for i in range(num_warmup, num_microbatches):
            # Forward
            input_tensor = input_microbatches[i] if self.stage.is_first_stage else self._recv_forward()
            output_tensor = self.stage(input_tensor)
            
            if self.stage.is_last_stage:
                loss = self.loss_fn(output_tensor, label_microbatches[i])
                losses.append(loss)
            else:
                self._send_forward(output_tensor)
            
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
            
            # Backward for oldest microbatch
            grad_tensor = self._recv_backward() if not self.stage.is_last_stage else None
            
            oldest_output = self.output_tensors.pop(0)
            oldest_input = self.input_tensors.pop(0)
            
            if grad_tensor is not None:
                oldest_output.backward(grad_tensor)
            else:
                oldest_output.backward()
            
            if not self.stage.is_first_stage and oldest_input.grad is not None:
                self._send_backward(oldest_input.grad)
        
        # Cooldown phase: drain pipeline
        for i in range(len(self.output_tensors)):
            grad_tensor = self._recv_backward() if not self.stage.is_last_stage else None
            
            output_tensor = self.output_tensors.pop(0)
            input_tensor = self.input_tensors.pop(0)
            
            if grad_tensor is not None:
                output_tensor.backward(grad_tensor)
            else:
                output_tensor.backward()
            
            if not self.stage.is_first_stage and input_tensor.grad is not None:
                self._send_backward(input_tensor.grad)
        
        # Aggregate loss
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
        
        return {'loss': total_loss}
    
    def forward_backward_gpipe(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """GPipe schedule: all forward then all backward"""
        num_microbatches = self.config.num_microbatches
        input_microbatches = torch.chunk(input_ids, num_microbatches, dim=0)
        label_microbatches = torch.chunk(labels, num_microbatches, dim=0) if labels is not None else [None] * num_microbatches
        
        losses = []
        
        # Forward pass for all microbatches
        for i in range(num_microbatches):
            input_tensor = input_microbatches[i] if self.stage.is_first_stage else self._recv_forward()
            output_tensor = self.stage(input_tensor)
            
            if self.stage.is_last_stage:
                loss = self.loss_fn(output_tensor, label_microbatches[i])
                losses.append(loss)
            else:
                self._send_forward(output_tensor)
            
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
        
        # Backward pass for all microbatches
        for i in range(num_microbatches - 1, -1, -1):
            grad_tensor = self._recv_backward() if not self.stage.is_last_stage else None
            
            output_tensor = self.output_tensors[i]
            input_tensor = self.input_tensors[i]
            
            if grad_tensor is not None:
                output_tensor.backward(grad_tensor)
            else:
                output_tensor.backward()
            
            if not self.stage.is_first_stage and input_tensor.grad is not None:
                self._send_backward(input_tensor.grad)
        
        self.input_tensors.clear()
        self.output_tensors.clear()
        
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
        return {'loss': total_loss}
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute one training step with selected schedule"""
        if self.config.schedule == "1f1b":
            return self.forward_backward_1f1b(input_ids, labels)
        elif self.config.schedule == "gpipe":
            return self.forward_backward_gpipe(input_ids, labels)
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
