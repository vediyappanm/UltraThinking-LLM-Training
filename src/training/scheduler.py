from transformers import get_linear_schedule_with_warmup


def build_scheduler(optimizer, args):
    """Linear schedule with warmup, default used across the project."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
