from __future__ import annotations

import argparse


def parse_args(profile):
    parser = argparse.ArgumentParser(description="Run TeenyZero autotuning across runtime and profile-level settings.")
    parser.add_argument(
        "--phase",
        choices=["auto", "phase1", "phase2", "phase3", "phase4"],
        default="auto",
        help="Autotune phase to run. `auto` runs phases 1, 2, 3, and 4 in sequence.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Initial candidate count to try.")
    parser.add_argument(
        "--objective",
        choices=["balanced", "selfplay", "train"],
        default="balanced",
        help="Scoring objective for ranking runtime candidates.",
    )
    parser.add_argument("--searches-per-worker", type=int, default=8, help="Searches per worker for the self-play benchmark.")
    parser.add_argument("--selfplay-simulations", type=int, default=profile.selfplay_simulations)
    parser.add_argument("--train-batches", type=int, default=8, help="Measured batches for the synthetic trainer benchmark.")
    parser.add_argument("--time-budget-minutes", type=float, default=30.0, help="Stop early if the wall-clock budget is exceeded.")
    parser.add_argument("--trial-timeout-s", type=float, default=600.0, help="Timeout per benchmark subprocess.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for candidate ordering.")
    parser.add_argument("--rounds", type=int, default=3, help="Phase 2 halving rounds.")
    parser.add_argument("--halving-ratio", type=float, default=2.0, help="Phase 2 elimination ratio between rounds.")
    parser.add_argument("--seed-run", default=None, help="Optional prior autotune run JSON used to seed phase 2, phase 3, or phase 4.")
    parser.add_argument("--phase3-finalists", type=int, default=3, help="Finalists from phase 2 (or phase 1) to validate in phase 3.")
    parser.add_argument("--phase3-train-window-samples", type=int, default=min(4096, profile.replay_window_samples))
    parser.add_argument("--phase3-train-samples", type=int, default=min(2048, profile.train_samples_per_cycle))
    parser.add_argument("--phase3-eval-samples", type=int, default=512)
    parser.add_argument("--phase3-train-epochs", type=int, default=1)
    parser.add_argument("--phase3-arena-games", type=int, default=4)
    parser.add_argument("--phase3-arena-simulations", type=int, default=max(64, profile.arena_simulations // 2))
    parser.add_argument("--phase3-bootstrap-simulations", type=int, default=max(32, profile.selfplay_simulations // 2))
    parser.add_argument(
        "--phase3-replay-source",
        choices=["auto", "live", "bootstrap"],
        default="auto",
        help="Use the live replay buffer if it is large enough, or bootstrap a small isolated one for phase 3.",
    )
    parser.add_argument("--phase4-trials", type=int, default=10, help="Learning/search candidates to try in phase 4.")
    parser.add_argument("--phase4-finalists", type=int, default=2, help="Top phase 3 runtime finalists to seed into phase 4.")
    parser.add_argument("--phase4-train-window-fraction", type=float, default=0.05, help="Fraction of replay window samples to use during phase 4 trials.")
    parser.add_argument("--phase4-train-samples-fraction", type=float, default=0.05, help="Fraction of train samples per cycle to use during phase 4 trials.")
    parser.add_argument("--phase4-max-window-samples", type=int, default=min(16384, profile.replay_window_samples))
    parser.add_argument("--phase4-max-train-samples", type=int, default=min(4096, profile.train_samples_per_cycle))
    parser.add_argument("--phase4-eval-samples", type=int, default=512)
    parser.add_argument("--phase4-train-epochs", type=int, default=1)
    parser.add_argument("--phase4-arena-games", type=int, default=4)
    parser.add_argument("--phase4-arena-simulations", type=int, default=max(64, profile.arena_simulations // 2))
    parser.add_argument("--phase4-bootstrap-simulations", type=int, default=max(32, profile.selfplay_simulations // 2))
    parser.add_argument(
        "--phase4-replay-source",
        choices=["auto", "live", "bootstrap"],
        default="auto",
        help="Use the live replay buffer if it is large enough, or bootstrap a small isolated one for phase 4.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable reuse of compatible saved phase runs and phase 4 cached trials.")
    parser.add_argument("--no-auto-promote", action="store_true", help="Do not update the shared markdown/catalog automatically after `--phase auto`.")
    return parser.parse_args()


def print_trial_summary(trial: dict, *, phase: str) -> None:
    config = trial["config"]
    prefix = f"{trial.get('round_label')}-" if trial.get("round_label") else ""
    status_label = trial["status"].upper()
    if trial.get("reused"):
        status_label += " (cached)"
    print(f"\n[{prefix}{trial['candidate_id'] or trial['label']}] {status_label}")
    print(
        "  config: "
        f"mode={config['actor_mode']} "
        f"workers={config['selfplay_workers']} "
        f"leaf={config['selfplay_leaf_batch_size']} "
        f"train_batch={config['train_batch_size']} "
        f"train_workers={config['train_num_workers']} "
        f"precision={config['train_precision']} "
        f"compile={config['train_compile']}"
    )
    if trial["status"] != "ok":
        for error in trial.get("errors", []):
            print(f"  error: {error}")
        return

    if phase == "phase3":
        print(
            "  quality: "
            f"loss {trial['pretrain_eval']['loss']:.4f} -> {trial['posttrain_eval']['loss']:.4f}, "
            f"arena {trial['arena']['score']:.3f} "
            f"({trial['arena']['wins']}-{trial['arena']['draws']}-{trial['arena']['losses']})"
        )
        print(
            "  source runtime: "
            f"{trial['selfplay'].get('positions_per_s', 0.0):.1f} pos/s, "
            f"{trial['train'].get('samples_per_s', 0.0):.1f} samples/s"
        )
        return

    if phase == "phase4":
        profile_overrides = trial.get("profile_overrides") or {}
        print(
            "  profile: "
            f"sims={profile_overrides.get('selfplay_simulations')} "
            f"opt={profile_overrides.get('train_optimizer')} "
            f"lr={float(profile_overrides.get('train_lr') or 0.0):.6g} "
            f"wd={float(profile_overrides.get('train_weight_decay') or 0.0):.6g} "
            f"accum={profile_overrides.get('train_grad_accum_steps')} "
            f"replay={profile_overrides.get('replay_window_samples')} "
            f"train_samples={profile_overrides.get('train_samples_per_cycle')}"
        )
        print(
            "  quality: "
            f"loss {trial['pretrain_eval']['loss']:.4f} -> {trial['posttrain_eval']['loss']:.4f}, "
            f"arena {trial['arena']['score']:.3f} "
            f"({trial['arena']['wins']}-{trial['arena']['draws']}-{trial['arena']['losses']})"
        )
        print(
            "  runtime: "
            f"{trial['selfplay'].get('positions_per_s', 0.0):.1f} pos/s, "
            f"{trial['train'].get('samples_per_s', 0.0):.1f} samples/s"
        )
        return

    print(
        "  self-play: "
        f"{trial['selfplay']['positions_per_s']:.1f} pos/s, "
        f"{trial['selfplay']['searches_per_s']:.1f} searches/s, "
        f"{trial['selfplay']['move_total_mean_ms']:.1f} ms/move"
    )
    print(
        "  train: "
        f"{trial['train']['samples_per_s']:.1f} samples/s, "
        f"{trial['train']['batches_per_s']:.2f} batches/s, "
        f"{trial['train']['avg_batch_time_ms']:.1f} ms/batch"
    )


def print_phase_header(phase: str, board_backend: str, args, requested_trials: int, *, device: str, profile_name: str) -> None:
    labels = {
        "phase1": "Phase 1",
        "phase2": "Phase 2",
        "phase3": "Phase 3",
        "phase4": "Phase 4",
    }
    print(f"\nTeenyZero Autotune {labels[phase]}")
    print(f"Device: {device}")
    print(f"Profile: {profile_name}")
    print(f"Board backend: {board_backend}")
    print(f"Trials requested: {requested_trials}")
    print(f"Objective: {args.objective}")
    if phase == "phase2":
        print(f"Rounds: {max(2, int(args.rounds))}")
        print(f"Halving ratio: {float(args.halving_ratio):.2f}")
    if phase == "phase3":
        print(f"Finalists: {max(1, int(args.phase3_finalists))}")
        print(f"Arena games: {max(1, int(args.phase3_arena_games))}")
        print(f"Replay source: {args.phase3_replay_source}")
    if phase == "phase4":
        print(f"Trials: {max(1, int(args.phase4_trials))}")
        print(f"Runtime finalists: {max(1, int(args.phase4_finalists))}")
        print(f"Arena games: {max(1, int(args.phase4_arena_games))}")
        print(f"Replay source: {args.phase4_replay_source}")


def print_auto_header(*, device: str, profile_name: str, board_backend: str, objective: str) -> None:
    print("\nTeenyZero Autotune Auto")
    print("Pipeline: phase 1 -> phase 2 -> phase 3 -> phase 4")
    print(f"Device: {device}")
    print(f"Profile: {profile_name}")
    print(f"Board backend: {board_backend}")
    print(f"Objective: {objective}")


def print_autotune_footer() -> None:
    print("[*] Open http://localhost:5001/autotune to inspect the dashboard.")
    print("[*] Promote a saved run manually with: python3 scripts/promote_autotune.py")
    print("[*] `--phase auto` updates the shared catalog automatically unless you pass --no-auto-promote.")
