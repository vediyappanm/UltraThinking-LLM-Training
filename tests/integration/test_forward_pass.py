import torch
from src.models.ultrathink import UltraThinkModel, UltraThinkConfig
from src.models.architecture import ModelConfig


def tiny_model():
    cfg = UltraThinkConfig(
        model_config=ModelConfig(
            vocab_size=256,
            n_positions=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            intermediate_size=128,
            activation="relu",
            dropout=0.0,
            attention_dropout=0.0,
            flash_attention=False,
            gradient_checkpointing=False,
        )
    )
    return UltraThinkModel(cfg)


def test_forward_smoke():
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 256, (2, 16))
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
    assert "loss" in out and torch.isfinite(out["loss"])    
