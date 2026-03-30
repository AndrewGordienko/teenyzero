from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.logic.trainer import AlphaTrainer
from teenyzero.alphazero.runtime import RuntimeProfile
from teenyzero.autotune.core.benchmarks import run_selfplay_trial
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs, save_autotune_run
from teenyzero.autotune.phases.phase3_arena import play_phase3_match
from teenyzero.autotune.phases.phase3_data import (
    prepare_phase3_datasets,
    prepare_phase3_replay_source,
    resolve_phase3_base_checkpoint,
)


PHASE4_KEY_ORDER = (
    "train_lr",
    "train_weight_decay",
    "train_optimizer",
    "train_grad_accum_steps",
    "selfplay_simulations",
    "replay_window_samples",
    "train_samples_per_cycle",
)

PHASE4_COMBO_PAIRS = (
    ("train_lr", "train_weight_decay"),
    ("train_lr", "train_grad_accum_steps"),
    ("train_lr", "selfplay_simulations"),
    ("replay_window_samples", "train_samples_per_cycle"),
    ("train_optimizer", "train_lr"),
)


def _top_successful_trials(seed_run: dict | None, limit: int = 2) -> list[dict]:
    if not seed_run:
        return []
    ranked = [trial for trial in seed_run.get("trials", []) if trial.get("status") == "ok" and trial.get("config")]
    ranked.sort(key=lambda item: float(item.get("score", -1.0)), reverse=True)
    return ranked[: max(1, int(limit))]


def phase4_seed_run(seed_run: dict | None = None) -> dict | None:
    if seed_run:
        return seed_run
    phase3_run = latest_autotune_run(phase="phase3")
    if phase3_run:
        return phase3_run
    phase2_run = latest_autotune_run(phase="phase2")
    if phase2_run:
        return phase2_run
    phase1_run = latest_autotune_run(phase="phase1")
    if phase1_run:
        return phase1_run
    return latest_autotune_run()


def _int_candidates(base_value: int, *, minimum: int, maximum: int, scales: tuple[float, ...]) -> list[int]:
    points = {int(base_value), minimum, maximum}
    for scale in scales:
        points.add(int(round(float(base_value) * scale)))
    values = sorted(max(minimum, min(maximum, point)) for point in points)
    return [value for value in values if value >= minimum]


def _float_candidates(base_value: float, *, minimum: float, maximum: float, scales: tuple[float, ...]) -> list[float]:
    points = {float(base_value), float(minimum), float(maximum)}
    for scale in scales:
        points.add(float(base_value) * float(scale))
    values = []
    for point in sorted(points):
        clamped = max(float(minimum), min(float(maximum), float(point)))
        if not any(abs(existing - clamped) < 1e-12 for existing in values):
            values.append(clamped)
    return values


def _optimizer_candidates(base_value: str) -> list[str]:
    values = [str(base_value).lower()]
    for value in ("adamw", "adam"):
        if value not in values:
            values.append(value)
    return values


def _baseline_profile_overrides(profile: RuntimeProfile) -> dict:
    return {
        "train_optimizer": str(profile.train_optimizer).lower(),
        "train_lr": float(profile.train_lr),
        "train_weight_decay": float(profile.train_weight_decay),
        "train_grad_accum_steps": int(profile.train_grad_accum_steps),
        "replay_window_samples": int(profile.replay_window_samples),
        "train_samples_per_cycle": int(profile.train_samples_per_cycle),
        "selfplay_simulations": int(profile.selfplay_simulations),
    }


def _override_value_sets(profile: RuntimeProfile) -> dict:
    return {
        "train_optimizer": _optimizer_candidates(profile.train_optimizer),
        "train_lr": _float_candidates(
            profile.train_lr,
            minimum=max(1e-5, profile.train_lr * 0.4),
            maximum=max(5e-5, profile.train_lr * 2.0),
            scales=(0.5, 0.75, 1.0, 1.5, 2.0),
        ),
        "train_weight_decay": _float_candidates(
            profile.train_weight_decay,
            minimum=0.0,
            maximum=max(1e-6, profile.train_weight_decay * 4.0),
            scales=(0.0, 0.5, 1.0, 2.0, 4.0),
        ),
        "train_grad_accum_steps": _int_candidates(
            profile.train_grad_accum_steps,
            minimum=1,
            maximum=max(8, profile.train_grad_accum_steps * 2),
            scales=(0.5, 1.0, 2.0),
        ),
        "replay_window_samples": _int_candidates(
            profile.replay_window_samples,
            minimum=max(20_000, profile.replay_window_samples // 2),
            maximum=max(profile.replay_window_samples, int(profile.replay_window_samples * 1.5)),
            scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        ),
        "train_samples_per_cycle": _int_candidates(
            profile.train_samples_per_cycle,
            minimum=max(10_000, profile.train_samples_per_cycle // 2),
            maximum=max(profile.train_samples_per_cycle, int(profile.train_samples_per_cycle * 1.5)),
            scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        ),
        "selfplay_simulations": _int_candidates(
            profile.selfplay_simulations,
            minimum=max(16, profile.selfplay_simulations // 2),
            maximum=max(profile.selfplay_simulations, int(profile.selfplay_simulations * 1.5)),
            scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        ),
    }


def _unique_phase4_candidates(candidates: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(
            {
                "config": candidate["config"],
                "profile_overrides": candidate["profile_overrides"],
                "source_candidate_id": (candidate.get("source_trial") or {}).get("candidate_id"),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _source_trial_payload(source_trial: dict) -> dict:
    return {
        "label": source_trial.get("label"),
        "candidate_id": source_trial.get("candidate_id"),
        "score": source_trial.get("score"),
        "selfplay": dict(source_trial.get("selfplay") or {}),
        "train": dict(source_trial.get("train") or {}),
    }


def _ordered_numeric_values(base_value: float, values: list[float]) -> list[float]:
    below = sorted((value for value in values if value < base_value), key=lambda value: abs(math.log(max(value, 1e-12) / max(base_value, 1e-12))))
    above = sorted((value for value in values if value > base_value), key=lambda value: abs(math.log(max(value, 1e-12) / max(base_value, 1e-12))))
    ordered = []
    for index in range(max(len(below), len(above))):
        if index < len(below):
            ordered.append(below[index])
        if index < len(above):
            ordered.append(above[index])
    return ordered


def _ordered_axis_alternatives(baseline_overrides: dict, value_sets: dict) -> dict[str, list]:
    alternatives = {}
    for key in PHASE4_KEY_ORDER:
        base_value = baseline_overrides[key]
        axis_values = [value for value in value_sets[key] if value != base_value]
        if isinstance(base_value, str):
            alternatives[key] = axis_values
            continue
        alternatives[key] = _ordered_numeric_values(float(base_value), sorted(axis_values))
    return alternatives


def _combo_axis_values(base_value, alternatives: list) -> list:
    if not alternatives:
        return []
    if isinstance(base_value, str):
        return alternatives[:2]

    below = [value for value in alternatives if value < base_value]
    above = [value for value in alternatives if value > base_value]
    selected = []
    if below:
        selected.append(below[0])
    if above:
        selected.append(above[0])
    for value in alternatives:
        if value in selected:
            continue
        selected.append(value)
        if len(selected) >= 2:
            break
    return selected[:2]


def _phase4_candidate_record(
    runtime_config: dict,
    profile_overrides: dict,
    source_payload: dict,
    *,
    is_seed: bool,
    priority: tuple,
) -> dict:
    return {
        "config": dict(runtime_config),
        "profile_overrides": dict(profile_overrides),
        "source_trial": dict(source_payload),
        "is_seed": bool(is_seed),
        "_priority": tuple(priority),
    }


def build_phase4_candidate_pool(profile: RuntimeProfile, seed_run: dict | None, *, finalist_count: int) -> list[dict]:
    source_trials = _top_successful_trials(seed_run, limit=finalist_count)
    baseline_overrides = _baseline_profile_overrides(profile)
    value_sets = _override_value_sets(profile)
    axis_alternatives = _ordered_axis_alternatives(baseline_overrides, value_sets)
    combo_values = {
        key: _combo_axis_values(baseline_overrides[key], axis_alternatives[key])
        for key in PHASE4_KEY_ORDER
    }

    candidates = []
    for source_index, source_trial in enumerate(source_trials):
        runtime_config = dict(source_trial.get("config") or {})
        if not runtime_config:
            continue

        source_payload = _source_trial_payload(source_trial)
        candidates.append(
            _phase4_candidate_record(
                runtime_config,
                baseline_overrides,
                source_payload,
                is_seed=source_index == 0,
                priority=(0, 0, source_index, 0, 0),
            )
        )

        for axis_rank in range(max((len(values) for values in axis_alternatives.values()), default=0)):
            for key_index, key in enumerate(PHASE4_KEY_ORDER):
                values = axis_alternatives[key]
                if axis_rank >= len(values):
                    continue
                variant = dict(baseline_overrides)
                variant[key] = values[axis_rank]
                candidates.append(
                    _phase4_candidate_record(
                        runtime_config,
                        variant,
                        source_payload,
                        is_seed=False,
                        priority=(1, axis_rank, source_index, key_index, 0),
                    )
                )

        for pair_index, (left_key, right_key) in enumerate(PHASE4_COMBO_PAIRS):
            ranked_pairs = []
            for left_rank, left_value in enumerate(combo_values[left_key]):
                for right_rank, right_value in enumerate(combo_values[right_key]):
                    variant = dict(baseline_overrides)
                    variant[left_key] = left_value
                    variant[right_key] = right_value
                    ranked_pairs.append((left_rank + right_rank, left_rank, right_rank, variant))
            ranked_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            for combo_rank, (_, left_rank, right_rank, variant) in enumerate(ranked_pairs):
                candidates.append(
                    _phase4_candidate_record(
                        runtime_config,
                        variant,
                        source_payload,
                        is_seed=False,
                        priority=(2, combo_rank, source_index, pair_index, left_rank + right_rank),
                    )
                )

    pool = _unique_phase4_candidates(candidates)
    pool.sort(key=lambda item: item.get("_priority", (99, 99, 99, 99, 99)))
    return pool


def build_phase4_candidates(
    profile: RuntimeProfile,
    seed_run: dict | None,
    *,
    finalist_count: int,
    trial_count: int,
) -> list[dict]:
    selected = build_phase4_candidate_pool(profile, seed_run, finalist_count=finalist_count)[: max(1, int(trial_count))]
    return [
        {
            "index": index,
            "candidate_id": f"H{index + 1:02d}",
            "label": f"H{index + 1:02d}",
            "config": dict(item["config"]),
            "profile_overrides": dict(item["profile_overrides"]),
            "source_trial": dict(item["source_trial"]),
            "is_seed": bool(item.get("is_seed")),
            "trial_count": len(selected),
        }
        for index, item in enumerate(selected)
    ]


def select_phase4_candidates(
    profile: RuntimeProfile,
    seed_run: dict | None,
    *,
    finalist_count: int,
    trial_count: int,
    settings: dict,
    runtime_args: dict,
    objective: str,
    seen_signatures: set[str] | None = None,
) -> tuple[list[dict], dict]:
    seen_signatures = set(seen_signatures or ())
    pool = build_phase4_candidate_pool(profile, seed_run, finalist_count=finalist_count)

    signature_pool = []
    unseen_pool = []
    for pool_index, item in enumerate(pool):
        candidate = {
            "index": pool_index,
            "candidate_id": None,
            "label": None,
            "config": dict(item["config"]),
            "profile_overrides": dict(item["profile_overrides"]),
            "source_trial": dict(item["source_trial"]),
            "is_seed": bool(item.get("is_seed")),
            "trial_count": len(pool),
        }
        candidate["candidate_signature"] = phase4_candidate_signature(candidate, settings, runtime_args, objective)
        signature_pool.append(candidate)
        if candidate["candidate_signature"] not in seen_signatures:
            unseen_pool.append(candidate)

    selected = []
    starting_rank = max(0, len(signature_pool) - len(unseen_pool))
    for selected_index, candidate in enumerate(unseen_pool[: max(1, int(trial_count))], start=1):
        labeled = dict(candidate)
        candidate_rank = starting_rank + selected_index
        labeled["candidate_id"] = f"H{candidate_rank:02d}"
        labeled["label"] = labeled["candidate_id"]
        selected.append(labeled)
    progress = {
        "candidate_pool_size": len(signature_pool),
        "cached_candidate_count": max(0, len(signature_pool) - len(unseen_pool)),
        "selected_candidate_count": len(selected),
        "remaining_candidate_count": max(0, len(unseen_pool) - len(selected)),
        "pool_exhausted": not unseen_pool,
    }
    return selected, progress


def phase4_candidate_signature(candidate: dict, settings: dict, runtime_args: dict, objective: str) -> str:
    payload = {
        "phase": "phase4",
        "objective": str(objective),
        "runtime_args": dict(runtime_args),
        "config": dict(candidate.get("config") or {}),
        "profile_overrides": dict(candidate.get("profile_overrides") or {}),
        "searches_per_worker": int(settings.get("searches_per_worker", 0)),
        "train_window_fraction": float(settings.get("train_window_fraction", 0.0)),
        "train_samples_fraction": float(settings.get("train_samples_fraction", 0.0)),
        "max_window_samples": int(settings.get("max_window_samples", 0)),
        "max_train_samples": int(settings.get("max_train_samples", 0)),
        "eval_samples": int(settings.get("eval_samples", 0)),
        "train_epochs": int(settings.get("train_epochs", 0)),
        "arena_games": int(settings.get("arena_games", 0)),
        "arena_simulations": int(settings.get("arena_simulations", 0)),
        "replay_source": str(settings.get("replay_source", "auto")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def load_phase4_cache(limit_runs: int = 128) -> dict[str, dict]:
    cache = {}
    for run in list_autotune_runs(limit=limit_runs, phase="phase4"):
        for trial in run.get("trials", []):
            signature = str(trial.get("candidate_signature") or "").strip()
            if not signature or signature in cache or trial.get("status") != "ok":
                continue
            cache[signature] = trial
    return cache


def _trial_window_samples(overrides: dict, settings: dict) -> int:
    target = int(round(int(overrides.get("replay_window_samples", 0)) * float(settings.get("train_window_fraction", 0.05))))
    return max(512, min(int(settings.get("max_window_samples", 0)), target))


def _trial_train_samples(overrides: dict, settings: dict) -> int:
    target = int(round(int(overrides.get("train_samples_per_cycle", 0)) * float(settings.get("train_samples_fraction", 0.05))))
    return max(256, min(int(settings.get("max_train_samples", 0)), target))


def _build_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, prefetch_factor: int) -> DataLoader:
    loader_kwargs = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
        "persistent_workers": max(0, int(num_workers)) > 0,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def _evaluate_model(model, dataloader: DataLoader, *, device: str) -> dict:
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    batch_count = 0
    sample_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for states, target_pis, target_zs in dataloader:
            states = states.to(device, non_blocking=device == "cuda")
            target_pis = target_pis.to(device, non_blocking=device == "cuda")
            target_zs = torch.nan_to_num(
                target_zs.to(device, non_blocking=device == "cuda"),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp_(-1.0, 1.0)
            target_pis = torch.nan_to_num(target_pis, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)
            pi_sums = target_pis.sum(dim=1, keepdim=True)
            valid_rows = pi_sums.squeeze(1) > 0
            if valid_rows.any():
                target_pis[valid_rows] = target_pis[valid_rows] / pi_sums[valid_rows]
            if (~valid_rows).any():
                target_pis[~valid_rows] = 0.0

            out_pi_logits, out_v = model(states)
            log_probs = F.log_softmax(out_pi_logits, dim=1)
            loss_pi = -torch.sum(target_pis * log_probs, dim=1).mean()
            loss_v = F.mse_loss(out_v, target_zs)
            loss = loss_pi + loss_v
            total_loss += float(loss.item())
            total_policy += float(loss_pi.item())
            total_value += float(loss_v.item())
            batch_count += 1
            sample_count += int(states.size(0))

    elapsed_s = time.perf_counter() - started
    safe_batches = max(batch_count, 1)
    safe_elapsed = max(elapsed_s, 1e-9)
    return {
        "loss": total_loss / safe_batches,
        "policy_loss": total_policy / safe_batches,
        "value_loss": total_value / safe_batches,
        "batches": batch_count,
        "samples_seen": sample_count,
        "duration_s": elapsed_s,
        "batches_per_s": batch_count / safe_elapsed,
        "samples_per_s": sample_count / safe_elapsed,
        "avg_batch_time_ms": (elapsed_s / safe_batches) * 1000.0,
    }


def run_phase4_trial(
    candidate: dict,
    *,
    settings: dict,
    replay_info: dict,
    work_dir: Path,
    device: str,
    profile: RuntimeProfile,
    project_root: Path,
    python_executable: str,
    runtime_args: dict,
    objective: str,
) -> dict:
    started = time.time()
    config = dict(candidate["config"])
    profile_overrides = dict(candidate.get("profile_overrides") or {})
    source_trial = dict(candidate.get("source_trial") or {})
    candidate_signature = phase4_candidate_signature(candidate, settings, runtime_args, objective)

    train_window_samples = _trial_window_samples(profile_overrides, settings)
    train_samples = _trial_train_samples(profile_overrides, settings)
    datasets = prepare_phase3_datasets(
        replay_info["data_dir"],
        train_window_samples=train_window_samples,
        train_samples=train_samples,
        eval_samples=int(settings.get("eval_samples", 512)),
        seed=int(settings.get("seed", 0)) + int(candidate["index"]),
    )

    selfplay_settings = {
        "round_label": None,
        "searches_per_worker": int(settings.get("searches_per_worker", 8)),
        "selfplay_simulations": int(profile_overrides["selfplay_simulations"]),
        "train_batches": 0,
        "trial_timeout_s": float(settings.get("trial_timeout_s", 600.0)),
    }
    selfplay_result = run_selfplay_trial(
        config,
        selfplay_settings,
        project_root=project_root,
        python_executable=python_executable,
    )
    if not selfplay_result.get("ok"):
        raise RuntimeError(selfplay_result.get("error", "phase 4 self-play benchmark failed"))

    effective_profile = replace(
        profile,
        train_optimizer=str(profile_overrides["train_optimizer"]),
        train_lr=float(profile_overrides["train_lr"]),
        train_weight_decay=float(profile_overrides["train_weight_decay"]),
        train_grad_accum_steps=int(profile_overrides["train_grad_accum_steps"]),
        replay_window_samples=int(profile_overrides["replay_window_samples"]),
        train_samples_per_cycle=int(profile_overrides["train_samples_per_cycle"]),
        selfplay_simulations=int(profile_overrides["selfplay_simulations"]),
    )

    base_checkpoint = resolve_phase3_base_checkpoint(work_dir)
    candidate_checkpoint = work_dir / f"{candidate['candidate_id'].lower()}_phase4.pth"

    model = build_model()
    load_checkpoint(model, base_checkpoint, map_location=device, allow_partial=True)

    eval_loader_before = _build_loader(
        datasets["eval_dataset"],
        batch_size=min(max(16, int(config["train_batch_size"])), max(16, datasets["eval_sample_count"])),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=effective_profile.train_prefetch_factor,
    )
    pretrain_eval = _evaluate_model(model, eval_loader_before, device=device)

    trainer = AlphaTrainer(
        model,
        device=device,
        lr=effective_profile.train_lr,
        optimizer_name=effective_profile.train_optimizer,
        weight_decay=effective_profile.train_weight_decay,
        momentum=effective_profile.train_momentum,
        grad_accum_steps=effective_profile.train_grad_accum_steps,
        precision=config["train_precision"],
        use_compile=config["train_compile"],
        max_grad_norm=effective_profile.max_grad_norm,
    )
    train_loader = _build_loader(
        datasets["train_dataset"],
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config["train_num_workers"]),
        pin_memory=bool(config["train_pin_memory"]),
        prefetch_factor=effective_profile.train_prefetch_factor,
    )

    train_metrics = None
    for _ in range(max(1, int(settings.get("train_epochs", 1)))):
        train_metrics = trainer.train_epoch(train_loader)

    eval_loader_after = _build_loader(
        datasets["eval_dataset"],
        batch_size=min(max(16, int(config["train_batch_size"])), max(16, datasets["eval_sample_count"])),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=effective_profile.train_prefetch_factor,
    )
    posttrain_eval = _evaluate_model(trainer.model, eval_loader_after, device=device)
    save_checkpoint(
        trainer.model,
        candidate_checkpoint,
        extra_meta={
            "autotune_phase": "phase4",
            "candidate_id": candidate["candidate_id"],
        },
    )
    arena = play_phase3_match(
        base_checkpoint,
        candidate_checkpoint,
        device=device,
        simulations=int(settings.get("arena_simulations", 64)),
        games=int(settings.get("arena_games", 4)),
    )

    pre_loss = float(pretrain_eval.get("loss", 0.0))
    post_loss = float(posttrain_eval.get("loss", 0.0))
    loss_delta = pre_loss - post_loss
    loss_improvement = (loss_delta / pre_loss) if pre_loss > 0 else 0.0

    return {
        "index": int(candidate["index"]),
        "label": candidate["label"],
        "candidate_id": candidate["candidate_id"],
        "is_seed": bool(candidate.get("is_seed")),
        "config": config,
        "profile_overrides": profile_overrides,
        "candidate_signature": candidate_signature,
        "started_at": started,
        "finished_at": time.time(),
        "status": "ok",
        "errors": [],
        "source_trial": source_trial,
        "selfplay": selfplay_result,
        "train": train_metrics or {},
        "pretrain_eval": pretrain_eval,
        "posttrain_eval": posttrain_eval,
        "arena": arena,
        "dataset": {
            "window_samples": int(datasets["window_samples"]),
            "window_files": int(datasets["file_count"]),
            "train_sample_count": int(datasets["train_sample_count"]),
            "eval_sample_count": int(datasets["eval_sample_count"]),
        },
        "quality": {
            "loss_delta": loss_delta,
            "loss_improvement": loss_improvement,
            "candidate_checkpoint": str(candidate_checkpoint),
        },
    }


def _ratio(value: float | None, baseline: float | None) -> float:
    if value is None or baseline is None or baseline <= 0:
        return 1.0
    return float(value) / float(baseline)


def phase4_trial_score(trial: dict, objective: str) -> float:
    if trial.get("status") != "ok":
        return -1.0
    selfplay = trial.get("selfplay", {})
    train = trial.get("train", {})
    quality = trial.get("quality", {})
    arena = trial.get("arena", {})
    source = trial.get("source_trial", {})
    ref_selfplay = source.get("selfplay", {})
    ref_train = source.get("train", {})

    selfplay_ratio = _ratio(selfplay.get("positions_per_s"), ref_selfplay.get("positions_per_s"))
    train_ratio = _ratio(train.get("samples_per_s"), ref_train.get("samples_per_s"))
    quality_ratio = max(0.0, 1.0 + float(quality.get("loss_improvement", 0.0)))
    arena_ratio = max(0.0, float(arena.get("score", 0.0)) / 0.5) if arena.get("games") else 1.0

    if objective == "selfplay":
        return (0.35 * arena_ratio) + (0.20 * quality_ratio) + (0.35 * selfplay_ratio) + (0.10 * train_ratio)
    if objective == "train":
        return (0.25 * arena_ratio) + (0.40 * quality_ratio) + (0.35 * train_ratio)
    return (0.30 * arena_ratio) + (0.30 * quality_ratio) + (0.20 * selfplay_ratio) + (0.20 * train_ratio)


def finalize_phase4_run(run_payload: dict) -> dict:
    objective = str(run_payload.get("objective", "balanced"))
    trials = list(run_payload.get("trials", []))
    for trial in trials:
        trial["score"] = phase4_trial_score(trial, objective)
    ranked = sorted(trials, key=lambda item: float(item.get("score", -1.0)), reverse=True)
    run_payload["trials"] = ranked
    run_payload["best_trial"] = ranked[0] if ranked else None
    return run_payload


def save_phase4_run(run_payload: dict, archive: bool = False):
    finalized = finalize_phase4_run(dict(run_payload))
    return save_autotune_run(finalized, archive=archive)


def latest_phase4_run() -> dict | None:
    return latest_autotune_run(phase="phase4")


def list_phase4_runs(limit: int = 8) -> list[dict]:
    return list_autotune_runs(limit=limit, phase="phase4")
