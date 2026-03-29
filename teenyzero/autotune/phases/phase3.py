from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.logic.trainer import AlphaTrainer
from teenyzero.alphazero.runtime import RuntimeProfile
from teenyzero.autotune.phases.phase3_arena import play_phase3_match
from teenyzero.autotune.phases.phase3_data import (
    prepare_phase3_datasets,
    prepare_phase3_replay_source,
    resolve_phase3_base_checkpoint,
)
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs, save_autotune_run


def _top_successful_trials(seed_run: dict | None, limit: int = 3) -> list[dict]:
    if not seed_run:
        return []
    ranked = [trial for trial in seed_run.get("trials", []) if trial.get("status") == "ok"]
    ranked.sort(key=lambda item: float(item.get("score", -1.0)), reverse=True)
    return ranked[: max(1, int(limit))]


def phase3_seed_run(seed_run: dict | None = None) -> dict | None:
    if seed_run:
        return seed_run
    phase2_run = latest_autotune_run(phase="phase2")
    if phase2_run:
        return phase2_run
    phase1_run = latest_autotune_run(phase="phase1")
    if phase1_run:
        return phase1_run
    return latest_autotune_run()


def build_phase3_candidates(seed_run: dict | None, finalist_count: int) -> list[dict]:
    selected = _top_successful_trials(seed_run, limit=finalist_count)
    return [
        {
            "index": index,
            "candidate_id": f"Q{index + 1:02d}",
            "label": f"Q{index + 1:02d}",
            "config": dict(trial.get("config") or {}),
            "source_trial": {
                "label": trial.get("label"),
                "candidate_id": trial.get("candidate_id"),
                "score": trial.get("score"),
                "selfplay": dict(trial.get("selfplay") or {}),
                "train": dict(trial.get("train") or {}),
            },
            "is_seed": index == 0,
            "trial_count": len(selected),
        }
        for index, trial in enumerate(selected)
        if trial.get("config")
    ]


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


def run_phase3_trial(
    candidate: dict,
    *,
    seed_run: dict | None,
    settings: dict,
    datasets: dict,
    work_dir: Path,
    device: str,
    profile: RuntimeProfile,
) -> dict:
    started = time.time()
    config = dict(candidate["config"])
    source_trial = dict(candidate.get("source_trial") or {})
    base_checkpoint = resolve_phase3_base_checkpoint(work_dir)
    candidate_checkpoint = work_dir / f"{candidate['candidate_id'].lower()}_trained.pth"

    model = build_model()
    load_checkpoint(model, base_checkpoint, map_location=device, allow_partial=True)

    eval_loader_before = _build_loader(
        datasets["eval_dataset"],
        batch_size=min(max(16, int(config["train_batch_size"])), max(16, datasets["eval_sample_count"])),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=profile.train_prefetch_factor,
    )
    pretrain_eval = _evaluate_model(model, eval_loader_before, device=device)

    trainer = AlphaTrainer(
        model,
        device=device,
        lr=profile.train_lr,
        optimizer_name=profile.train_optimizer,
        weight_decay=profile.train_weight_decay,
        momentum=profile.train_momentum,
        grad_accum_steps=profile.train_grad_accum_steps,
        precision=config["train_precision"],
        use_compile=config["train_compile"],
        max_grad_norm=profile.max_grad_norm,
    )
    train_loader = _build_loader(
        datasets["train_dataset"],
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config["train_num_workers"]),
        pin_memory=bool(config["train_pin_memory"]),
        prefetch_factor=profile.train_prefetch_factor,
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
        prefetch_factor=profile.train_prefetch_factor,
    )
    posttrain_eval = _evaluate_model(trainer.model, eval_loader_after, device=device)
    save_checkpoint(
        trainer.model,
        candidate_checkpoint,
        extra_meta={
            "autotune_phase": "phase3",
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
        "started_at": started,
        "finished_at": time.time(),
        "status": "ok",
        "errors": [],
        "source_trial": source_trial,
        "selfplay": dict(source_trial.get("selfplay") or {}),
        "train": train_metrics or {},
        "pretrain_eval": pretrain_eval,
        "posttrain_eval": posttrain_eval,
        "arena": arena,
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


def phase3_trial_score(trial: dict, reference_trial: dict | None, objective: str) -> float:
    if trial.get("status") != "ok":
        return -1.0
    selfplay = trial.get("selfplay", {})
    train = trial.get("train", {})
    quality = trial.get("quality", {})
    arena = trial.get("arena", {})
    ref_selfplay = (reference_trial or {}).get("selfplay", {})
    ref_train = (reference_trial or {}).get("train", {})

    selfplay_ratio = _ratio(selfplay.get("positions_per_s"), ref_selfplay.get("positions_per_s"))
    train_ratio = _ratio(train.get("samples_per_s"), ref_train.get("samples_per_s"))
    quality_ratio = max(0.0, 1.0 + float(quality.get("loss_improvement", 0.0)))
    arena_ratio = max(0.0, float(arena.get("score", 0.0)) / 0.5) if arena.get("games") else 1.0

    if objective == "selfplay":
        return (0.35 * arena_ratio) + (0.20 * quality_ratio) + (0.35 * selfplay_ratio) + (0.10 * train_ratio)
    if objective == "train":
        return (0.35 * arena_ratio) + (0.40 * quality_ratio) + (0.25 * train_ratio)
    return (0.35 * arena_ratio) + (0.30 * quality_ratio) + (0.20 * selfplay_ratio) + (0.15 * train_ratio)


def finalize_phase3_run(run_payload: dict) -> dict:
    objective = str(run_payload.get("objective", "balanced"))
    reference_trial = ((run_payload.get("seed_run") or {}).get("best_trial")) or None
    trials = list(run_payload.get("trials", []))
    for trial in trials:
        trial["score"] = phase3_trial_score(trial, reference_trial, objective)
    ranked = sorted(trials, key=lambda item: float(item.get("score", -1.0)), reverse=True)
    best = ranked[0] if ranked else None
    run_payload["trials"] = ranked
    run_payload["best_trial"] = best
    return run_payload


def save_phase3_run(run_payload: dict, archive: bool = False):
    finalized = finalize_phase3_run(dict(run_payload))
    return save_autotune_run(finalized, archive=archive)


def latest_phase3_run() -> dict | None:
    return latest_autotune_run(phase="phase3")


def list_phase3_runs(limit: int = 8) -> list[dict]:
    return list_autotune_runs(limit=limit, phase="phase3")
