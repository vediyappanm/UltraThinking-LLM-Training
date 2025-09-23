import torch

from src.models.ultrathink import UltraThinkModel, UltraThinkConfig
from src.models.architecture import ModelConfig


def build_tiny_ultrathink():
    # Tiny config for quick CPU smoke test
    vocab_size = 50257
    n_embd = 128
    n_head = 8
    head_dim = n_embd // n_head  # 16

    model_config = ModelConfig(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=n_embd,
        n_layer=2,
        n_head=n_head,
        n_kv_head=4,
        rotary_dim=head_dim,  # must match head_dim for RoPE broadcasting
        intermediate_size=512,
        activation="swiglu",
        norm_type="rmsnorm",
        norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embed_dropout=0.0,
        tie_word_embeddings=True,
        use_cache=True,
        attention_bias=False,
        mlp_bias=False,
        flash_attention=False,  # CPU safe
        sliding_window=None,
        gradient_checkpointing=False,
        max_position_embeddings=128,
    )

    cfg = UltraThinkConfig(
        model_config=model_config,
        enable_dre=False,
        enable_constitutional=False,
        enable_moe=False,
        enable_multimodal=False,
        enable_rlhf=False,
        batch_size=2,
        gradient_accumulation=1,
        learning_rate=1e-4,
        warmup_steps=0,
        max_steps=10,
        gradient_checkpointing=False,
        mixed_precision="fp32",
        compile_model=False,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.0,
    )
    return UltraThinkModel(cfg)


def run_forward_pass():
    device = torch.device("cpu")
    model = build_tiny_ultrathink().to(device)

    batch_size = 2
    seq_len = 16
    vocab_size = model.config.model_config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"].item() if outputs["loss"] is not None else None
    logits_shape = tuple(outputs["logits"].shape)

    print("Smoke test OK")
    print(f"Loss: {loss}")
    print(f"Logits shape: {logits_shape}")


if __name__ == "__main__":
    run_forward_pass()
