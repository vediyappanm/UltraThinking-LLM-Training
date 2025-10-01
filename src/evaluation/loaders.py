"""
Minimal dataset loaders/adapters for evaluation benchmarks (GSM8K, HumanEval, MMLU).

These return lightweight Python iterables that yield dicts in the shapes expected by
`src/evaluation/benchmarks.py` evaluators.

Heavy datasets are optional; loaders handle missing datasets gracefully by
raising a clear exception the caller can catch and skip.
"""
from typing import Iterable, Dict, Any, Optional

from datasets import load_dataset  # type: ignore


def load_gsm8k(split: str = "test", subset: str = "main", max_samples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    ds = load_dataset("gsm8k", subset, split=split)
    def iterator():
        count = 0
        for row in ds:
            yield {"question": [row["question"]], "answer": [row["answer"]]}
            count += 1
            if max_samples and count >= max_samples:
                break
    return iterator()


essential_humaneval_fields = ["prompt", "test", "canonical_solution"]

def load_humaneval(split: str = "test", max_samples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    ds = load_dataset("openai_humaneval", split=split)
    def iterator():
        count = 0
        for row in ds:
            item = {k: [row.get(k)] for k in essential_humaneval_fields}
            yield item
            count += 1
            if max_samples and count >= max_samples:
                break
    return iterator()


def load_mmlu(split: str = "validation", subject: Optional[str] = None, max_samples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """
    Load MMLU-like multiple-choice QA. Try widely used sources in order.
    Returns dicts with fields: question, choices, answer.
    """
    ds = None
    try:
        # Newer mirror
        ds = load_dataset("cais/mmlu", subject or "abstract_algebra", split=split)
    except Exception:
        try:
            ds = load_dataset("hendrycks_test", subject or "abstract_algebra", split=split)
        except Exception as e:
            raise RuntimeError(f"Unable to load MMLU dataset: {e}")

    def iterator():
        count = 0
        for row in ds:
            choices = row.get("choices") or [row.get("A"), row.get("B"), row.get("C"), row.get("D")]
            yield {
                "question": [row.get("question", "")],
                "choices": [choices],
                "answer": [row.get("answer", "")],
            }
            count += 1
            if max_samples and count >= max_samples:
                break
    return iterator()
