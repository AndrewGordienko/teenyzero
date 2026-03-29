from __future__ import annotations

import os
from pathlib import Path

import torch

from teenyzero.alphazero.runtime import RuntimeProfile


def unique_int_candidates(base_value: int, minimum: int, maximum: int) -> list[int]:
    points = {int(base_value), minimum, maximum}
    for ratio in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        points.add(int(round(base_value * ratio)))
    values = sorted(max(minimum, min(maximum, point)) for point in points)
    return [value for value in values if value >= minimum]


def baseline_actor_mode(device: str) -> str:
    return "inprocess" if device in {"mps", "cuda"} else "mp"


def precision_options(profile: RuntimeProfile, device: str) -> list[str]:
    if device != "cuda":
        return [profile.train_precision]
    values = [profile.train_precision, "fp32", "fp16", "bf16"]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def compile_options(profile: RuntimeProfile, device: str) -> list[bool]:
    if device != "cuda" or not hasattr(torch, "compile"):
        return [False]
    values = [bool(profile.train_compile), False, True]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def pin_memory_options(profile: RuntimeProfile, device: str) -> list[bool]:
    if device != "cuda":
        return [False]
    values = [bool(profile.train_pin_memory), False, True]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def build_apply_command(runtime_args: dict, config: dict) -> str:
    args = [
        "python3",
        "scripts/run_visualizers.py",
        f"--device {runtime_args['device']}",
        f"--profile {runtime_args['profile']}",
        f"--board-backend {runtime_args['board_backend']}",
        f"--actor-mode {config['actor_mode']}",
        f"--actor-workers {config['selfplay_workers']}",
        f"--selfplay-leaf-batch-size {config['selfplay_leaf_batch_size']}",
        f"--train-batch-size {config['train_batch_size']}",
        f"--train-num-workers {config['train_num_workers']}",
        f"--train-precision {config['train_precision']}",
    ]
    args.append("--train-pin-memory" if config["train_pin_memory"] else "--no-train-pin-memory")
    args.append("--train-compile" if config["train_compile"] else "--no-train-compile")
    return " ".join(args)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))
