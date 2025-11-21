import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _tp_size(tensor_parallel_size: int) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return max(1, int(tensor_parallel_size))


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gather_output: bool = True, tensor_parallel_size: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tensor_parallel_size = _tp_size(tensor_parallel_size)
        if self.tensor_parallel_size == 1:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.weight = self.linear.weight
            self.bias = self.linear.bias
        else:
            part = out_features // self.tensor_parallel_size
            self.weight = nn.Parameter(torch.empty(part, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(part))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'linear') and isinstance(self.linear, nn.Linear):
            self.linear.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_size == 1:
            return self.linear(input)
        local = F.linear(input, self.weight, self.bias)
        if not self.gather_output or self.tensor_parallel_size == 1:
            return local
        if not dist.is_available() or not dist.is_initialized():
            return local
        outs = [torch.empty_like(local) for _ in range(self.tensor_parallel_size)]
        dist.all_gather(outs, local)
        return torch.cat(outs, dim=-1)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_is_parallel: bool = True, tensor_parallel_size: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tensor_parallel_size = _tp_size(tensor_parallel_size)
        if self.tensor_parallel_size == 1:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.weight = self.linear.weight
            self.bias = self.linear.bias
        else:
            part = in_features // self.tensor_parallel_size
            self.weight = nn.Parameter(torch.empty(out_features, part))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'linear') and isinstance(self.linear, nn.Linear):
            self.linear.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_size == 1:
            return self.linear(input)
        x = input
        if self.input_is_parallel and dist.is_available() and dist.is_initialized():
            pass
        out = F.linear(x, self.weight, None)
        if self.bias is not None:
            out = out + self.bias
        if self.input_is_parallel and dist.is_available() and dist.is_initialized():
            dist.all_reduce(out)
        return out


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, tensor_parallel_size: int = 1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_parallel_size = _tp_size(tensor_parallel_size)
        if self.tensor_parallel_size == 1:
            self.embed = nn.Embedding(num_embeddings, embedding_dim)
        else:
            part = num_embeddings // self.tensor_parallel_size
            self.embed = nn.Embedding(part, embedding_dim)
            self.vocab_start = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_size == 1:
            return self.embed(input_ids)
        return self.embed(input_ids)
