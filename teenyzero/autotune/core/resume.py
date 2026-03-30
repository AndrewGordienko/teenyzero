from __future__ import annotations

from teenyzero.autotune.core.storage import latest_autotune_run
from teenyzero.autotune.phases.phase1 import hardware_fingerprint


RESUMABLE_STATUSES = {"completed", "time_budget_reached"}


def _expected_search_settings(phase: str, args) -> dict:
    if phase == "phase1":
        return {
            "trials": int(args.trials),
            "searches_per_worker": int(args.searches_per_worker),
            "selfplay_simulations": int(args.selfplay_simulations),
            "train_batches": int(args.train_batches),
            "time_budget_minutes": float(args.time_budget_minutes),
            "trial_timeout_s": float(args.trial_timeout_s),
            "seed": int(args.seed),
        }
    if phase == "phase2":
        return {
            "trials": int(args.trials),
            "searches_per_worker": int(args.searches_per_worker),
            "selfplay_simulations": int(args.selfplay_simulations),
            "train_batches": int(args.train_batches),
            "time_budget_minutes": float(args.time_budget_minutes),
            "trial_timeout_s": float(args.trial_timeout_s),
            "seed": int(args.seed),
            "rounds": int(args.rounds),
            "halving_ratio": float(args.halving_ratio),
        }
    if phase == "phase3":
        return {
            "time_budget_minutes": float(args.time_budget_minutes),
            "seed": int(args.seed),
            "finalists": int(args.phase3_finalists),
            "train_window_samples": int(args.phase3_train_window_samples),
            "train_samples": int(args.phase3_train_samples),
            "eval_samples": int(args.phase3_eval_samples),
            "train_epochs": int(args.phase3_train_epochs),
            "arena_games": int(args.phase3_arena_games),
            "arena_simulations": int(args.phase3_arena_simulations),
            "replay_source": str(args.phase3_replay_source),
            "bootstrap_simulations": int(args.phase3_bootstrap_simulations),
        }
    if phase == "phase4":
        return {
            "time_budget_minutes": float(args.time_budget_minutes),
            "seed": int(args.seed),
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
        }
    return {}


def _settings_match(run_payload: dict, phase: str, args) -> bool:
    expected = _expected_search_settings(phase, args)
    observed = dict(run_payload.get("search_settings") or {})
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if isinstance(expected_value, float):
            if observed_value is None or abs(float(observed_value) - expected_value) > 1e-9:
                return False
            continue
        if observed_value != expected_value:
            return False
    return True


def run_is_compatible(run_payload: dict | None, phase: str, *, board_backend: str, runtime, objective: str, args) -> bool:
    if not run_payload:
        return False
    if str(run_payload.get("phase") or "").strip().lower() != str(phase).strip().lower():
        return False
    if str(run_payload.get("status") or "").strip().lower() not in RESUMABLE_STATUSES:
        return False
    if not run_payload.get("trials"):
        return False

    runtime_args = run_payload.get("runtime_args") or {}
    if runtime_args.get("device") != runtime.device:
        return False
    if runtime_args.get("profile") != runtime.profile.name:
        return False
    if runtime_args.get("board_backend") != board_backend:
        return False
    if str(run_payload.get("objective") or "balanced") != str(objective):
        return False
    if not _settings_match(run_payload, phase, args):
        return False

    current_hardware = hardware_fingerprint(runtime)
    run_hardware = run_payload.get("hardware") or {}
    current_platform = current_hardware.get("platform") or {}
    run_platform = run_hardware.get("platform") or {}
    if current_platform.get("system") != run_platform.get("system"):
        return False
    if current_platform.get("machine") != run_platform.get("machine"):
        return False
    if (current_hardware.get("cuda_device") or {}).get("name") != (run_hardware.get("cuda_device") or {}).get("name"):
        return False
    return True


def latest_compatible_phase_run(phase: str, *, board_backend: str, runtime, objective: str, args) -> dict | None:
    latest = latest_autotune_run(phase=phase)
    if not run_is_compatible(latest, phase, board_backend=board_backend, runtime=runtime, objective=objective, args=args):
        return None
    return latest
