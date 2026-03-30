from __future__ import annotations

import time
from pathlib import Path

from teenyzero.autotune.phases.phase1 import hardware_fingerprint
from teenyzero.paths import runtime_paths_payload


def seed_payload(seed_run: dict | None) -> dict | None:
    if not seed_run:
        return None
    best_trial = (seed_run.get("best_trial") or {}).copy()
    return {
        "phase": seed_run.get("phase"),
        "run_id": seed_run.get("run_id"),
        "objective": seed_run.get("objective"),
        "best_trial": best_trial,
    }


def trial_settings(args, round_plan: dict | None = None) -> dict:
    if round_plan:
        return {
            "round_label": round_plan["label"],
            "searches_per_worker": int(round_plan["searches_per_worker"]),
            "selfplay_simulations": int(round_plan["selfplay_simulations"]),
            "train_batches": int(round_plan["train_batches"]),
            "trial_timeout_s": float(round_plan["trial_timeout_s"]),
        }
    return {
        "round_label": None,
        "searches_per_worker": int(args.searches_per_worker),
        "selfplay_simulations": int(args.selfplay_simulations),
        "train_batches": int(args.train_batches),
        "trial_timeout_s": float(args.trial_timeout_s),
    }


def phase3_settings(args, work_dir: Path) -> dict:
    return {
        "finalists": int(args.phase3_finalists),
        "train_window_samples": int(args.phase3_train_window_samples),
        "train_samples": int(args.phase3_train_samples),
        "eval_samples": int(args.phase3_eval_samples),
        "train_epochs": int(args.phase3_train_epochs),
        "arena_games": int(args.phase3_arena_games),
        "arena_simulations": int(args.phase3_arena_simulations),
        "replay_source": str(args.phase3_replay_source),
        "bootstrap_simulations": int(args.phase3_bootstrap_simulations),
        "work_dir": str(work_dir),
    }


def phase4_settings(args, work_dir: Path) -> dict:
    return {
        "trials": int(args.phase4_trials),
        "finalists": int(args.phase4_finalists),
        "train_window_fraction": float(args.phase4_train_window_fraction),
        "train_samples_fraction": float(args.phase4_train_samples_fraction),
        "max_window_samples": int(args.phase4_max_window_samples),
        "max_train_samples": int(args.phase4_max_train_samples),
        "eval_samples": int(args.phase4_eval_samples),
        "train_epochs": int(args.phase4_train_epochs),
        "arena_games": int(args.phase4_arena_games),
        "arena_simulations": int(args.phase4_arena_simulations),
        "replay_source": str(args.phase4_replay_source),
        "bootstrap_simulations": int(args.phase4_bootstrap_simulations),
        "searches_per_worker": int(args.searches_per_worker),
        "trial_timeout_s": float(args.trial_timeout_s),
        "work_dir": str(work_dir),
    }


def base_payload(phase: str, board_backend: str, started_at: float, run_id: str, *, runtime) -> dict:
    return {
        "phase": phase,
        "status": "running",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "objective": None,
        "runtime_args": {
            "device": runtime.device,
            "profile": runtime.profile.name,
            "board_backend": board_backend,
        },
        "hardware": hardware_fingerprint(runtime),
        "runtime_paths": runtime_paths_payload(),
    }


def phase1_payload(args, board_backend: str, started_at: float, run_id: str, *, runtime) -> dict:
    payload = base_payload("phase1", board_backend, started_at, run_id, runtime=runtime)
    payload.update(
        {
            "objective": args.objective,
            "search_settings": {
                "trials": int(args.trials),
                "searches_per_worker": int(args.searches_per_worker),
                "selfplay_simulations": int(args.selfplay_simulations),
                "train_batches": int(args.train_batches),
                "time_budget_minutes": float(args.time_budget_minutes),
                "trial_timeout_s": float(args.trial_timeout_s),
                "seed": int(args.seed),
            },
            "current_trial": None,
            "trials": [],
        }
    )
    return payload


def phase2_payload(args, board_backend: str, started_at: float, run_id: str, seed_run: dict | None, *, runtime) -> dict:
    payload = base_payload("phase2", board_backend, started_at, run_id, runtime=runtime)
    payload.update(
        {
            "objective": args.objective,
            "search_settings": {
                "trials": int(args.trials),
                "searches_per_worker": int(args.searches_per_worker),
                "selfplay_simulations": int(args.selfplay_simulations),
                "train_batches": int(args.train_batches),
                "time_budget_minutes": float(args.time_budget_minutes),
                "trial_timeout_s": float(args.trial_timeout_s),
                "seed": int(args.seed),
                "rounds": int(args.rounds),
                "halving_ratio": float(args.halving_ratio),
                "seed_run": str(args.seed_run) if args.seed_run else None,
            },
            "seed_run": seed_payload(seed_run),
            "current_round": None,
            "current_trial": None,
            "rounds": [],
            "trials": [],
        }
    )
    return payload


def phase3_payload(
    args,
    board_backend: str,
    started_at: float,
    run_id: str,
    seed_run: dict | None,
    work_dir: Path,
    *,
    runtime,
) -> dict:
    payload = base_payload("phase3", board_backend, started_at, run_id, runtime=runtime)
    payload.update(
        {
            "objective": args.objective,
            "search_settings": {
                "time_budget_minutes": float(args.time_budget_minutes),
                "seed": int(args.seed),
                **phase3_settings(args, work_dir),
            },
            "seed_run": seed_payload(seed_run),
            "current_trial": None,
            "replay_source": None,
            "replay_summary": None,
            "work_dir": str(work_dir),
            "trials": [],
        }
    )
    return payload


def phase4_payload(
    args,
    board_backend: str,
    started_at: float,
    run_id: str,
    seed_run: dict | None,
    work_dir: Path,
    *,
    runtime,
) -> dict:
    payload = base_payload("phase4", board_backend, started_at, run_id, runtime=runtime)
    payload.update(
        {
            "objective": args.objective,
            "search_settings": {
                "time_budget_minutes": float(args.time_budget_minutes),
                "seed": int(args.seed),
                **phase4_settings(args, work_dir),
            },
            "seed_run": seed_payload(seed_run),
            "current_trial": None,
            "replay_source": None,
            "replay_summary": None,
            "work_dir": str(work_dir),
            "trials": [],
        }
    )
    return payload


def failed_trial(candidate: dict, error: str) -> dict:
    now = time.time()
    return {
        "index": int(candidate["index"]),
        "label": candidate["label"],
        "candidate_id": candidate.get("candidate_id"),
        "is_baseline": bool(candidate.get("is_baseline")),
        "is_seed": bool(candidate.get("is_seed")),
        "config": dict(candidate["config"]),
        "profile_overrides": dict(candidate.get("profile_overrides") or {}),
        "started_at": now,
        "finished_at": now,
        "status": "failed",
        "errors": [str(error)],
        "selfplay": {},
        "train": {},
    }
