import torch
from src.models.dynamic_reasoning import DynamicReasoningEngine


def test_dre_basic():
    # Minimal smoke test: instantiate and score complexity
    class DummyBase:
        def __init__(self):
            self.hidden_size = 16
    dre = DynamicReasoningEngine(base_model=DummyBase(), config={"hidden_dim": 16})
    x = torch.randn(1, 4, 16)
    score = dre.compute_complexity(x)
    assert score is not None
