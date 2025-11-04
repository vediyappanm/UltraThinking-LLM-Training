from __future__ import annotations
"""
Serving integration stubs for vLLM and TGI.
- vLLM: use Python engine API when installed
- TGI: provide CLI hints (requires separate server install)
"""
from typing import Optional


def start_vllm_engine(model: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:
        raise ImportError("vLLM not installed. pip install vllm") from e
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
    return llm


def generate_with_vllm(llm, prompts, temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 128):
    from vllm import SamplingParams  # type: ignore
    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return llm.generate(prompts, params)


def tgi_hint():
    return (
        "To use TGI, install and run the server separately, e.g.:\n"
        "docker run -p 8080:80 -v $PWD:/data ghcr.io/huggingface/text-generation-inference:latest \\\n"
        " --model-id <model> --num-shard 2\n"
        "Then query via text-generation client."
    )
