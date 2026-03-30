from __future__ import annotations

import json
import os
import time
from pathlib import Path

from teenyzero.autotune.core.benchmarks import build_trial_record
from teenyzero.autotune.core.cli import print_auto_header, print_phase_header, print_trial_summary
from teenyzero.autotune.core.payloads import (
    failed_trial,
    phase1_payload,
    phase2_payload,
    phase3_payload,
    phase3_settings,
    phase4_payload,
    phase4_settings,
    trial_settings,
)
from teenyzero.autotune.core.resume import latest_compatible_phase_run
from teenyzero.autotune.phases.phase1 import build_phase1_candidates, save_phase1_run
from teenyzero.autotune.phases.phase2 import (
    build_phase2_candidates,
    build_phase2_round_plans,
    finalize_phase2_run,
    phase2_seed_run,
    phase2_survivor_count,
    save_phase2_run,
)
from teenyzero.autotune.core.storage import latest_autotune_run
from teenyzero.paths import AUTOTUNE_WORK_DIR


AUTO_PHASE_WEIGHTS = {
    "phase1": 3.0,
    "phase2": 4.0,
    "phase3": 5.0,
    "phase4": 6.0,
}


def board_backend_name() -> str:
    return (os.environ.get("TEENYZERO_BOARD_BACKEND", "") or "auto").strip().lower() or "auto"


def load_seed_run(path: str | None, *, default_phase: str | None = None) -> dict | None:
    if path:
        with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
            return json.load(handle)
    if default_phase == "phase2":
        return phase2_seed_run()
    if default_phase == "phase3":
        from teenyzero.autotune.phases.phase3 import phase3_seed_run

        return phase3_seed_run()
    if default_phase == "phase4":
        from teenyzero.autotune.phases.phase4 import phase4_seed_run

        return phase4_seed_run()
    return latest_autotune_run()


def run_phase1(args, board_backend: str, deadline: float, *, runtime, profile, project_root: Path, python_executable: str) -> dict | None:
    started_at = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    run_payload = phase1_payload(args, board_backend, started_at, run_id, runtime=runtime)
    save_phase1_run(run_payload)

    candidates = build_phase1_candidates(profile, runtime.device, trial_count=args.trials, seed=args.seed)
    print_phase_header("phase1", board_backend, args, len(candidates), device=runtime.device, profile_name=profile.name)

    try:
        for candidate in candidates:
            if time.time() >= deadline:
                run_payload["status"] = "time_budget_reached"
                print("\n[*] Time budget reached. Stopping phase 1 early.")
                break

            run_payload["current_trial"] = {
                "index": candidate["index"],
                "label": candidate["label"],
                "config": candidate["config"],
            }
            save_phase1_run(run_payload)
            trial = build_trial_record(
                candidate,
                trial_settings(args),
                project_root=project_root,
                python_executable=python_executable,
            )
            run_payload["trials"].append(trial)
            print_trial_summary(trial, phase="phase1")
            save_phase1_run(run_payload)
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["finished_at"] = time.time()
        save_phase1_run(run_payload, archive=True)
        print("\n[!] Autotune interrupted. Partial results were saved.")
        return latest_autotune_run(phase="phase1")

    if run_payload["status"] == "running":
        run_payload["status"] = "completed"
    run_payload["finished_at"] = time.time()
    archive_path = save_phase1_run(run_payload, archive=True)
    latest_path = save_phase1_run(run_payload)
    print(f"\n[*] Phase 1 results saved to {latest_path}")
    print(f"[*] Archived run saved to {archive_path}")
    return latest_autotune_run(phase="phase1")


def _survivor_candidates(round_payload: dict, round_index: int, round_count: int, halving_ratio: float) -> list[dict]:
    successful = [trial for trial in round_payload.get("trials", []) if trial.get("status") == "ok"]
    keep = phase2_survivor_count(len(successful), round_index, round_count, halving_ratio)
    survivors = []
    for trial in successful[:keep]:
        survivors.append(
            {
                "index": len(survivors),
                "candidate_id": trial.get("candidate_id") or trial.get("label"),
                "label": trial.get("candidate_id") or trial.get("label"),
                "config": dict(trial.get("config") or {}),
                "is_seed": bool(trial.get("is_seed")),
                "trial_count": keep,
            }
        )
    round_payload["survivors"] = [
        {
            "candidate_id": item["candidate_id"],
            "label": item["label"],
            "score": next(
                float(trial.get("score", -1.0))
                for trial in round_payload.get("trials", [])
                if (trial.get("candidate_id") or trial.get("label")) == item["candidate_id"]
            ),
        }
        for item in survivors
    ]
    return survivors


def run_phase2(args, board_backend: str, deadline: float, *, runtime, profile, project_root: Path, python_executable: str, seed_run: dict | None = None) -> dict | None:
    started_at = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    seed_run = seed_run or load_seed_run(args.seed_run, default_phase="phase2")
    run_payload = phase2_payload(args, board_backend, started_at, run_id, seed_run, runtime=runtime)
    save_phase2_run(run_payload)

    candidates = build_phase2_candidates(
        profile,
        runtime.device,
        trial_count=args.trials,
        seed_run=seed_run,
        seed=args.seed,
    )
    round_plans = build_phase2_round_plans(
        searches_per_worker=int(args.searches_per_worker),
        selfplay_simulations=int(args.selfplay_simulations),
        train_batches=int(args.train_batches),
        trial_timeout_s=float(args.trial_timeout_s),
        rounds=int(args.rounds),
    )
    print_phase_header("phase2", board_backend, args, len(candidates), device=runtime.device, profile_name=profile.name)

    try:
        for round_index, round_plan in enumerate(round_plans):
            if not candidates:
                break
            if time.time() >= deadline:
                run_payload["status"] = "time_budget_reached"
                print("\n[*] Time budget reached. Stopping phase 2 early.")
                break

            print(
                f"\n[{round_plan['label']}] {len(candidates)} candidates "
                f"at {round_plan['searches_per_worker']} searches/worker and "
                f"{round_plan['train_batches']} train batches"
            )
            round_payload = {
                **round_plan,
                "candidate_count": len(candidates),
                "current_trial": None,
                "trials": [],
                "survivors": [],
            }
            run_payload["current_round"] = {
                "index": round_index,
                "label": round_plan["label"],
                "candidate_count": len(candidates),
            }
            run_payload["current_trial"] = None
            run_payload["rounds"].append(round_payload)
            save_phase2_run(run_payload)

            for candidate in candidates:
                if time.time() >= deadline:
                    run_payload["status"] = "time_budget_reached"
                    break
                trial_candidate = dict(candidate)
                trial_candidate["label"] = f"{round_plan['label']}-{candidate['candidate_id']}"
                round_payload["current_trial"] = {
                    "candidate_id": candidate["candidate_id"],
                    "label": trial_candidate["label"],
                    "config": candidate["config"],
                }
                run_payload["current_trial"] = dict(round_payload["current_trial"])
                save_phase2_run(run_payload)
                trial = build_trial_record(
                    trial_candidate,
                    trial_settings(args, round_plan),
                    project_root=project_root,
                    python_executable=python_executable,
                )
                round_payload["trials"].append(trial)
                print_trial_summary(trial, phase="phase2")
                save_phase2_run(run_payload)

            run_payload["rounds"][-1] = round_payload
            run_payload = finalize_phase2_run(run_payload)
            round_payload = run_payload["rounds"][-1]
            if not round_payload.get("trials"):
                break

            candidates = _survivor_candidates(
                round_payload,
                round_index=round_index,
                round_count=len(round_plans),
                halving_ratio=float(args.halving_ratio),
            )
            run_payload["rounds"][-1] = round_payload
            best_trial = round_payload.get("best_trial") or {}
            print(
                f"[*] {round_plan['label']} best: "
                f"{best_trial.get('candidate_id', best_trial.get('label', 'n/a'))} "
                f"score={float(best_trial.get('score', 0.0)):.3f} "
                f"survivors={len(candidates)}"
            )
            save_phase2_run(run_payload)
            if run_payload["status"] == "time_budget_reached" or len(candidates) <= 1:
                break
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["finished_at"] = time.time()
        save_phase2_run(run_payload, archive=True)
        print("\n[!] Autotune interrupted. Partial results were saved.")
        return latest_autotune_run(phase="phase2")

    if run_payload["status"] == "running":
        run_payload["status"] = "completed"
    run_payload["finished_at"] = time.time()
    archive_path = save_phase2_run(run_payload, archive=True)
    latest_path = save_phase2_run(run_payload)
    print(f"\n[*] Phase 2 results saved to {latest_path}")
    print(f"[*] Archived run saved to {archive_path}")
    return latest_autotune_run(phase="phase2")


def run_phase3(args, board_backend: str, deadline: float, *, runtime, profile, seed_run: dict | None = None) -> dict | None:
    from teenyzero.autotune.phases.phase3 import (
        build_phase3_candidates,
        prepare_phase3_datasets,
        prepare_phase3_replay_source,
        run_phase3_trial,
        save_phase3_run,
    )

    started_at = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    seed_run = seed_run or load_seed_run(args.seed_run, default_phase="phase3")
    work_dir = AUTOTUNE_WORK_DIR / f"phase3_{run_id}"
    settings = phase3_settings(args, work_dir)
    run_payload = phase3_payload(args, board_backend, started_at, run_id, seed_run, work_dir, runtime=runtime)
    save_phase3_run(run_payload)

    candidates = build_phase3_candidates(seed_run, finalist_count=args.phase3_finalists)
    print_phase_header("phase3", board_backend, args, len(candidates), device=runtime.device, profile_name=profile.name)

    if not candidates:
        run_payload["status"] = "failed"
        run_payload["finished_at"] = time.time()
        run_payload["errors"] = ["No successful source trials were available for phase 3."]
        archive_path = save_phase3_run(run_payload, archive=True)
        latest_path = save_phase3_run(run_payload)
        print(f"\n[!] Phase 3 could not start: {run_payload['errors'][0]}")
        print(f"[*] Phase 3 results saved to {latest_path}")
        print(f"[*] Archived run saved to {archive_path}")
        return latest_autotune_run(phase="phase3")

    try:
        replay_info = prepare_phase3_replay_source(
            seed_run,
            work_dir=work_dir,
            device=runtime.device,
            settings=settings,
            deadline=deadline,
        )
        run_payload["replay_source"] = replay_info
        save_phase3_run(run_payload)

        datasets = prepare_phase3_datasets(
            replay_info["data_dir"],
            train_window_samples=int(args.phase3_train_window_samples),
            train_samples=int(args.phase3_train_samples),
            eval_samples=int(args.phase3_eval_samples),
            seed=int(args.seed),
        )
        run_payload["replay_summary"] = {
            "source": replay_info.get("source"),
            "data_dir": replay_info.get("data_dir"),
            "available_samples": replay_info.get("sample_count"),
            "available_files": replay_info.get("file_count"),
            "window_samples": datasets["window_samples"],
            "window_files": datasets["file_count"],
            "train_sample_count": datasets["train_sample_count"],
            "eval_sample_count": datasets["eval_sample_count"],
        }
        save_phase3_run(run_payload)

        for candidate in candidates:
            if time.time() >= deadline:
                run_payload["status"] = "time_budget_reached"
                print("\n[*] Time budget reached. Stopping phase 3 early.")
                break

            run_payload["current_trial"] = {
                "candidate_id": candidate["candidate_id"],
                "label": candidate["label"],
                "config": candidate["config"],
            }
            save_phase3_run(run_payload)
            try:
                trial = run_phase3_trial(
                    candidate,
                    seed_run=seed_run,
                    settings=settings,
                    datasets=datasets,
                    work_dir=work_dir,
                    device=runtime.device,
                    profile=profile,
                )
            except Exception as exc:
                trial = failed_trial(candidate, str(exc))
            run_payload["trials"].append(trial)
            print_trial_summary(trial, phase="phase3")
            save_phase3_run(run_payload)
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["finished_at"] = time.time()
        save_phase3_run(run_payload, archive=True)
        print("\n[!] Autotune interrupted. Partial results were saved.")
        return latest_autotune_run(phase="phase3")
    except Exception as exc:
        run_payload["status"] = "failed"
        run_payload["finished_at"] = time.time()
        run_payload["errors"] = [str(exc)]
        archive_path = save_phase3_run(run_payload, archive=True)
        latest_path = save_phase3_run(run_payload)
        print(f"\n[!] Phase 3 failed: {exc}")
        print(f"[*] Phase 3 results saved to {latest_path}")
        print(f"[*] Archived run saved to {archive_path}")
        return latest_autotune_run(phase="phase3")

    if run_payload["status"] == "running":
        run_payload["status"] = "completed"
    run_payload["finished_at"] = time.time()
    archive_path = save_phase3_run(run_payload, archive=True)
    latest_path = save_phase3_run(run_payload)
    print(f"\n[*] Phase 3 results saved to {latest_path}")
    print(f"[*] Archived run saved to {archive_path}")
    return latest_autotune_run(phase="phase3")


def _reuse_phase4_trial(candidate: dict, cached_trial: dict) -> dict:
    trial = dict(cached_trial)
    trial["index"] = int(candidate["index"])
    trial["label"] = candidate["label"]
    trial["candidate_id"] = candidate["candidate_id"]
    trial["is_seed"] = bool(candidate.get("is_seed"))
    trial["config"] = dict(candidate.get("config") or {})
    trial["profile_overrides"] = dict(candidate.get("profile_overrides") or {})
    trial["source_trial"] = dict(candidate.get("source_trial") or {})
    trial["reused"] = True
    trial["errors"] = list(trial.get("errors") or [])
    return trial


def run_phase4(
    args,
    board_backend: str,
    deadline: float,
    *,
    runtime,
    profile,
    project_root: Path,
    python_executable: str,
    seed_run: dict | None = None,
) -> dict | None:
    from teenyzero.autotune.phases.phase3_data import prepare_phase3_replay_source
    from teenyzero.autotune.phases.phase4 import (
        load_phase4_cache,
        select_phase4_candidates,
        run_phase4_trial,
        save_phase4_run,
    )

    started_at = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    seed_run = seed_run or load_seed_run(args.seed_run, default_phase="phase4")
    work_dir = AUTOTUNE_WORK_DIR / f"phase4_{run_id}"
    settings = phase4_settings(args, work_dir)
    run_payload = phase4_payload(args, board_backend, started_at, run_id, seed_run, work_dir, runtime=runtime)
    save_phase4_run(run_payload)

    phase4_cache = {} if args.no_resume else load_phase4_cache()
    runtime_args = dict(run_payload.get("runtime_args") or {})
    candidates, progress = select_phase4_candidates(
        profile,
        seed_run,
        finalist_count=args.phase4_finalists,
        trial_count=args.phase4_trials,
        settings=settings,
        runtime_args=runtime_args,
        objective=args.objective,
        seen_signatures=set(phase4_cache),
    )
    run_payload["search_progress"] = progress
    print_phase_header("phase4", board_backend, args, len(candidates), device=runtime.device, profile_name=profile.name)
    print(
        f"Candidate pool: {progress.get('candidate_pool_size', 0)} total, "
        f"{progress.get('cached_candidate_count', 0)} cached, "
        f"{progress.get('selected_candidate_count', 0)} selected, "
        f"{progress.get('remaining_candidate_count', 0)} left after this batch"
    )

    if progress.get("pool_exhausted"):
        latest = latest_autotune_run(phase="phase4")
        print("\n[*] Phase 4 search space is already exhausted for the current settings.")
        if latest and latest.get("best_trial"):
            best = latest["best_trial"]
            print(
                f"[*] Reusing latest phase 4 winner "
                f"{best.get('candidate_id', best.get('label', 'n/a'))} "
                f"from run {latest.get('run_id', 'unknown')} "
                f"score={float(best.get('score', 0.0)):.3f}"
            )
            return latest

    if not candidates:
        run_payload["status"] = "failed"
        run_payload["finished_at"] = time.time()
        run_payload["errors"] = ["No unseen phase 4 candidates were available for the current settings."]
        archive_path = save_phase4_run(run_payload, archive=True)
        latest_path = save_phase4_run(run_payload)
        print(f"\n[!] Phase 4 could not start: {run_payload['errors'][0]}")
        print(f"[*] Phase 4 results saved to {latest_path}")
        print(f"[*] Archived run saved to {archive_path}")
        return latest_autotune_run(phase="phase4")

    try:
        replay_info = prepare_phase3_replay_source(
            seed_run,
            work_dir=work_dir,
            device=runtime.device,
            settings={
                "replay_source": settings["replay_source"],
                "bootstrap_simulations": int(settings["bootstrap_simulations"]),
                "train_window_samples": int(settings["max_window_samples"]),
                "train_samples": int(settings["max_train_samples"]),
                "eval_samples": int(settings["eval_samples"]),
            },
            deadline=deadline,
        )
        run_payload["replay_source"] = replay_info
        run_payload["replay_summary"] = {
            "source": replay_info.get("source"),
            "data_dir": replay_info.get("data_dir"),
            "available_samples": replay_info.get("sample_count"),
            "available_files": replay_info.get("file_count"),
            "max_window_samples": int(settings["max_window_samples"]),
            "max_train_samples": int(settings["max_train_samples"]),
            "eval_sample_count": int(settings["eval_samples"]),
        }
        save_phase4_run(run_payload)

        for candidate in candidates:
            if time.time() >= deadline:
                run_payload["status"] = "time_budget_reached"
                print("\n[*] Time budget reached. Stopping phase 4 early.")
                break

            candidate_signature = candidate["candidate_signature"]
            run_payload["current_trial"] = {
                "candidate_id": candidate["candidate_id"],
                "label": candidate["label"],
                "config": candidate["config"],
                "profile_overrides": candidate.get("profile_overrides"),
                "candidate_signature": candidate_signature,
            }
            save_phase4_run(run_payload)

            try:
                trial = run_phase4_trial(
                    candidate,
                    settings=settings,
                    replay_info=replay_info,
                    work_dir=work_dir,
                    device=runtime.device,
                    profile=profile,
                    project_root=project_root,
                    python_executable=python_executable,
                    runtime_args=runtime_args,
                    objective=args.objective,
                )
            except Exception as exc:
                trial = failed_trial(candidate, str(exc))
                trial["candidate_signature"] = candidate_signature

            run_payload["trials"].append(trial)
            print_trial_summary(trial, phase="phase4")
            save_phase4_run(run_payload)

        executed_count = len(run_payload.get("trials", []))
        scheduled_count = int(progress.get("selected_candidate_count", 0))
        if executed_count < scheduled_count:
            run_payload["search_progress"] = dict(run_payload.get("search_progress") or {})
            run_payload["search_progress"]["remaining_candidate_count"] = int(run_payload["search_progress"].get("remaining_candidate_count", 0)) + (scheduled_count - executed_count)
            run_payload["search_progress"]["selected_candidate_count"] = executed_count
            save_phase4_run(run_payload)
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["finished_at"] = time.time()
        save_phase4_run(run_payload, archive=True)
        print("\n[!] Autotune interrupted. Partial results were saved.")
        return latest_autotune_run(phase="phase4")
    except Exception as exc:
        run_payload["status"] = "failed"
        run_payload["finished_at"] = time.time()
        run_payload["errors"] = [str(exc)]
        archive_path = save_phase4_run(run_payload, archive=True)
        latest_path = save_phase4_run(run_payload)
        print(f"\n[!] Phase 4 failed: {exc}")
        print(f"[*] Phase 4 results saved to {latest_path}")
        print(f"[*] Archived run saved to {archive_path}")
        return latest_autotune_run(phase="phase4")

    if run_payload["status"] == "running":
        run_payload["status"] = "completed"
    run_payload["finished_at"] = time.time()
    archive_path = save_phase4_run(run_payload, archive=True)
    latest_path = save_phase4_run(run_payload)
    print(f"\n[*] Phase 4 results saved to {latest_path}")
    print(f"[*] Archived run saved to {archive_path}")
    return latest_autotune_run(phase="phase4")


def allocated_deadline(overall_deadline: float, remaining_phases: list[str]) -> float:
    if not remaining_phases:
        return time.time()
    remaining_s = max(0.0, overall_deadline - time.time())
    if remaining_s <= 0.0:
        return time.time()
    current = remaining_phases[0]
    total_weight = sum(AUTO_PHASE_WEIGHTS[phase] for phase in remaining_phases)
    phase_budget_s = remaining_s * (AUTO_PHASE_WEIGHTS[current] / max(total_weight, 1e-9))
    return time.time() + phase_budget_s


def _reuse_candidate_run(phase: str, args, board_backend: str, runtime, *, resume_allowed: bool) -> dict | None:
    if not resume_allowed:
        return None
    return latest_compatible_phase_run(
        phase,
        board_backend=board_backend,
        runtime=runtime,
        objective=args.objective,
        args=args,
    )


def _remaining_auto_phases(phase_order: list[str], start_index: int, compatible_runs: dict[str, dict | None], *, resume_chain_alive: bool) -> list[str]:
    if not resume_chain_alive:
        return phase_order[start_index:]
    for index in range(start_index, len(phase_order)):
        phase = phase_order[index]
        if not compatible_runs.get(phase):
            return phase_order[index:]
    return []


def _print_reused_phase(run_payload: dict) -> None:
    best = run_payload.get("best_trial") or {}
    score = float(best.get("score", 0.0)) if best else 0.0
    label = best.get("candidate_id", best.get("label", "n/a"))
    print(
        f"[*] Reusing compatible {run_payload.get('phase', 'autotune')} run "
        f"{run_payload.get('run_id', 'unknown')} best={label} score={score:.3f}"
    )


def run_auto(args, board_backend: str, overall_deadline: float, *, runtime, profile, project_root: Path, python_executable: str) -> dict | None:
    from teenyzero.autotune.catalog.recommendations import (
        AUTOTUNE_RESULTS_DOC_PATH,
        RECOMMENDATIONS_PATH,
        promote_autotune_run,
    )

    print_auto_header(
        device=runtime.device,
        profile_name=profile.name,
        board_backend=board_backend,
        objective=args.objective,
    )

    phase_order = ["phase1", "phase2", "phase3", "phase4"]
    compatible_runs = {phase: None for phase in phase_order}
    resume_chain_alive = not args.no_resume and not args.seed_run

    phase1_run = _reuse_candidate_run("phase1", args, board_backend, runtime, resume_allowed=resume_chain_alive)
    compatible_runs["phase1"] = phase1_run
    if phase1_run:
        _print_reused_phase(phase1_run)
    else:
        phase1_deadline = allocated_deadline(
            overall_deadline,
            _remaining_auto_phases(phase_order, 0, compatible_runs, resume_chain_alive=resume_chain_alive),
        )
        phase1_run = run_phase1(
            args,
            board_backend,
            phase1_deadline,
            runtime=runtime,
            profile=profile,
            project_root=project_root,
            python_executable=python_executable,
        )
        resume_chain_alive = False

    phase2_seed = phase1_run or latest_autotune_run(phase="phase1")
    compatible_runs["phase2"] = _reuse_candidate_run("phase2", args, board_backend, runtime, resume_allowed=resume_chain_alive)
    phase2_run = compatible_runs["phase2"]
    if phase2_run:
        _print_reused_phase(phase2_run)
    else:
        phase2_deadline = allocated_deadline(
            overall_deadline,
            _remaining_auto_phases(phase_order, 1, compatible_runs, resume_chain_alive=resume_chain_alive),
        )
        phase2_run = run_phase2(
            args,
            board_backend,
            phase2_deadline,
            runtime=runtime,
            profile=profile,
            project_root=project_root,
            python_executable=python_executable,
            seed_run=phase2_seed,
        )
        resume_chain_alive = False

    phase3_seed = phase2_run or latest_autotune_run(phase="phase2") or phase2_seed
    compatible_runs["phase3"] = _reuse_candidate_run("phase3", args, board_backend, runtime, resume_allowed=resume_chain_alive)
    phase3_run = compatible_runs["phase3"]
    if phase3_run:
        _print_reused_phase(phase3_run)
    else:
        phase3_deadline = allocated_deadline(
            overall_deadline,
            _remaining_auto_phases(phase_order, 2, compatible_runs, resume_chain_alive=resume_chain_alive),
        )
        phase3_run = run_phase3(
            args,
            board_backend,
            phase3_deadline,
            runtime=runtime,
            profile=profile,
            seed_run=phase3_seed,
        )
        resume_chain_alive = False

    phase4_seed = phase3_run or latest_autotune_run(phase="phase3") or phase3_seed
    phase4_deadline = allocated_deadline(
        overall_deadline,
        _remaining_auto_phases(phase_order, 3, compatible_runs, resume_chain_alive=resume_chain_alive),
    )
    phase4_run = run_phase4(
        args,
        board_backend,
        phase4_deadline,
        runtime=runtime,
        profile=profile,
        project_root=project_root,
        python_executable=python_executable,
        seed_run=phase4_seed,
    )

    final_run = phase4_run or phase3_run or phase2_run or phase1_run
    if final_run and final_run.get("best_trial"):
        best = final_run["best_trial"]
        print(
            "\n[*] Auto pipeline best: "
            f"{best.get('candidate_id', best.get('label', 'n/a'))} "
            f"from {final_run.get('phase', 'autotune')} "
            f"score={float(best.get('score', 0.0)):.3f}"
        )
        if not args.no_auto_promote:
            try:
                entry = promote_autotune_run(final_run)
                print(f"[*] Updated shared recommendation: {entry.get('id', 'n/a')}")
                print(f"[*] Registry: {RECOMMENDATIONS_PATH}")
                print(f"[*] Docs: {AUTOTUNE_RESULTS_DOC_PATH}")
            except Exception as exc:
                print(f"[!] Auto-promotion failed: {exc}")
    return final_run
