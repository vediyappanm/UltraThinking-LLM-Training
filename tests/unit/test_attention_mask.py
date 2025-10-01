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


def test_attention_mask_no_nan():
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 256, (1, 32))
    attention_mask = torch.ones_like(input_ids)
    # Pad last 4 tokens
    attention_mask[:, -4:] = 0
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    assert torch.isfinite(out["loss"]).item(), "Loss should be finite with key-side padding + causal mask"
