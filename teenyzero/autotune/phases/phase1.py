from __future__ import annotations

import itertools
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass

import torch

from teenyzero.alphazero.runtime import RuntimeProfile, RuntimeSelection
from teenyzero.autotune.core.common import (
    baseline_actor_mode,
    build_apply_command,
    compile_options,
    pin_memory_options,
    precision_options,
    unique_int_candidates,
)
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs, save_autotune_run


@dataclass(frozen=True)
class Phase1Config:
    actor_mode: str
    selfplay_workers: int
    selfplay_leaf_batch_size: int
    train_batch_size: int
    train_num_workers: int
    train_pin_memory: bool
    train_precision: str
    train_compile: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, ValueError):
        return None


def _cuda_device_payload() -> dict | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(getattr(props, "multi_processor_count", 0) or 0),
            "major": int(getattr(props, "major", 0) or 0),
            "minor": int(getattr(props, "minor", 0) or 0),
        }
    except Exception:
        return {"name": "cuda", "total_memory_bytes": None}


def hardware_fingerprint(selection: RuntimeSelection) -> dict:
    return {
        "device": selection.device,
        "runtime_profile": selection.profile.name,
        "requested_device": selection.requested_device,
        "requested_profile": selection.requested_profile,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu_count": int(os.cpu_count() or 1),
        "physical_memory_bytes": _physical_memory_bytes(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_device": _cuda_device_payload(),
    }


def build_phase1_candidates(
    profile: RuntimeProfile,
    device: str,
    trial_count: int,
    seed: int = 0,
) -> list[dict]:
    cpu_count = max(1, int(os.cpu_count() or 1))
    worker_cap = max(2, min(cpu_count, max(profile.selfplay_workers * 2, 4)))
    train_worker_cap = max(0, min(cpu_count, max(profile.train_num_workers * 2 + 2, 4)))
    leaf_cap = max(8, profile.selfplay_leaf_batch_size * 2)
    batch_cap = max(16, min(512, profile.train_batch_size * 2))

    baseline = Phase1Config(
        actor_mode=baseline_actor_mode(device),
        selfplay_workers=max(1, int(profile.selfplay_workers)),
        selfplay_leaf_batch_size=max(1, int(profile.selfplay_leaf_batch_size)),
        train_batch_size=max(1, int(profile.train_batch_size)),
        train_num_workers=max(0, int(profile.train_num_workers)),
        train_pin_memory=bool(profile.train_pin_memory if device == "cuda" else False),
        train_precision=str(profile.train_precision),
        train_compile=bool(profile.train_compile if device == "cuda" and hasattr(torch, "compile") else False),
    )

    actor_modes = ["inprocess", "mp"] if device in {"mps", "cuda"} else ["mp", "inprocess"]
    combos = itertools.product(
        actor_modes,
        unique_int_candidates(baseline.selfplay_workers, minimum=1, maximum=worker_cap),
        unique_int_candidates(baseline.selfplay_leaf_batch_size, minimum=4, maximum=leaf_cap),
        unique_int_candidates(baseline.train_batch_size, minimum=8, maximum=batch_cap),
        unique_int_candidates(baseline.train_num_workers, minimum=0, maximum=train_worker_cap),
        pin_memory_options(profile, device),
        precision_options(profile, device),
        compile_options(profile, device),
    )

    candidates = []
    seen = set()
    for item in combos:
        config = Phase1Config(
            actor_mode=item[0],
            selfplay_workers=item[1],
            selfplay_leaf_batch_size=item[2],
            train_batch_size=item[3],
            train_num_workers=item[4],
            train_pin_memory=item[5],
            train_precision=item[6],
            train_compile=item[7],
        )
        key = tuple(config.to_dict().items())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(config)

    baseline_key = tuple(baseline.to_dict().items())
    sampled = [baseline]
    remainder = [candidate for candidate in candidates if tuple(candidate.to_dict().items()) != baseline_key]
    random.Random(seed).shuffle(remainder)
    sampled.extend(remainder[: max(0, int(trial_count) - 1)])

    total = len(sampled)
    return [
        {
            "index": idx,
            "label": f"T{idx + 1:02d}",
            "config": config.to_dict(),
            "is_baseline": idx == 0,
            "trial_count": total,
        }
        for idx, config in enumerate(sampled)
    ]


def _ratio(value: float | None, baseline: float | None) -> float:
    if value is None or baseline is None or baseline <= 0:
        return 0.0
    return float(value) / float(baseline)


def phase1_trial_score(trial: dict, baseline: dict | None, objective: str) -> float:
    if trial.get("status") != "ok":
        return -1.0

    selfplay = trial.get("selfplay", {})
    train = trial.get("train", {})
    baseline_selfplay = (baseline or {}).get("selfplay", {})
    baseline_train = (baseline or {}).get("train", {})

    positions_ratio = _ratio(selfplay.get("positions_per_s"), baseline_selfplay.get("positions_per_s"))
    searches_ratio = _ratio(selfplay.get("searches_per_s"), baseline_selfplay.get("searches_per_s"))
    latency_ratio = _ratio(
        baseline_selfplay.get("move_total_mean_ms"),
        selfplay.get("move_total_mean_ms"),
    )
    train_ratio = _ratio(train.get("samples_per_s"), baseline_train.get("samples_per_s"))

    if objective == "selfplay":
        return (0.65 * positions_ratio) + (0.20 * searches_ratio) + (0.15 * latency_ratio)
    if objective == "train":
        return train_ratio
    return (0.40 * positions_ratio) + (0.15 * searches_ratio) + (0.20 * latency_ratio) + (0.25 * train_ratio)


def finalize_phase1_run(run_payload: dict) -> dict:
    trials = list(run_payload.get("trials", []))
    baseline = next((item for item in trials if item.get("is_baseline")), None)
    objective = str(run_payload.get("objective", "balanced"))
    for trial in trials:
        trial["score"] = phase1_trial_score(trial, baseline, objective)
    ranked = sorted(trials, key=lambda item: float(item.get("score", -1.0)), reverse=True)
    best = ranked[0] if ranked else None
    run_payload["trials"] = ranked
    run_payload["best_trial"] = best
    if best is not None:
        run_payload["apply_command"] = build_apply_command(run_payload["runtime_args"], best["config"])
    return run_payload


def save_phase1_run(run_payload: dict, archive: bool = False) -> Path:
    finalized = finalize_phase1_run(dict(run_payload))
    return save_autotune_run(finalized, archive=archive)


def latest_phase1_run() -> dict | None:
    return latest_autotune_run(phase="phase1")


def list_phase1_runs(limit: int = 8) -> list[dict]:
    return list_autotune_runs(limit=limit, phase="phase1")
