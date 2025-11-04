from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    start_seq: int = 512
    end_seq: int = 4096
    ramp_steps: int = 10000


def max_length_for_step(step: int, cfg: CurriculumConfig) -> int:
    if cfg.ramp_steps <= 0:
        return cfg.end_seq
    frac = min(1.0, max(0.0, step / cfg.ramp_steps))
    return int(cfg.start_seq + frac * (cfg.end_seq - cfg.start_seq))
