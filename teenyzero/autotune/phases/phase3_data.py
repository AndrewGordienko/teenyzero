from __future__ import annotations

import time
from pathlib import Path

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.logic.batched_selfplay import BatchedSelfPlayRunner
from teenyzero.alphazero.logic.trainer import ReplayWindowDataset, latest_replay_window, replay_buffer_summary
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.paths import BEST_MODEL_PATH, LATEST_MODEL_PATH, REPLAY_BUFFER_PATH


def _bootstrap_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    fallback = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else None
    if fallback and fallback.exists():
        model = build_model()
        load_result = load_checkpoint(model, fallback, map_location="cpu", allow_partial=True)
        if load_result.get("loaded"):
            save_checkpoint(model, path)
            return path
    model = build_model()
    save_checkpoint(model, path)
    return path


def resolve_phase3_base_checkpoint(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    if LATEST_MODEL_PATH.exists():
        return LATEST_MODEL_PATH
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    return _bootstrap_checkpoint(work_dir / "base_model.pth")


def _warmup_replay_buffer(
    replay_dir: Path,
    *,
    device: str,
    model_path: Path,
    target_samples: int,
    concurrent_games: int,
    simulations: int,
    leaf_batch_size: int,
    deadline: float | None = None,
) -> dict:
    replay_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    load_checkpoint(model, model_path, map_location="cpu", allow_partial=True)
    evaluator = AlphaZeroEvaluator(model=model, device=device, use_cache=True)
    engine = MCTS(
        evaluator=evaluator,
        params={
            "SIMULATIONS": max(1, int(simulations)),
            "C_PUCT": 1.5,
            "ALPHA": 0.3,
            "EPS": 0.30,
            "VIRTUAL_LOSS": 0.0,
            "PARALLEL_THREADS": 1,
            "FPU_REDUCTION": 0.4,
            "LEAF_BATCH_SIZE": max(1, int(leaf_batch_size)),
        },
    )
    runner = BatchedSelfPlayRunner(
        evaluator=evaluator,
        engine=engine,
        buffer_path=str(replay_dir),
        concurrent_games=max(1, int(concurrent_games)),
    )
    slots = [runner._new_game(slot_id) for slot_id in range(max(1, int(concurrent_games)))]
    started = time.perf_counter()
    games = 0

    summary = replay_buffer_summary(str(replay_dir))
    while int(summary.get("sample_count", 0)) < max(1, int(target_samples)):
        if deadline is not None and time.time() >= deadline:
            break
        runner._play_batch_move(slots, worker_id=0, stats_dict=None)
        finished_payloads = []
        for idx, slot in enumerate(slots):
            if not runner._is_finished(slot):
                continue
            finished_payloads.append((idx, runner._finalize_game(slot, stats_dict=None)))
            slots[idx] = runner._new_game(slot.slot_id)
        timestamp = int(time.time() * 1000)
        for game_idx, payload in finished_payloads:
            filename = f"phase3_bootstrap_{timestamp}_{game_idx}.npz"
            runner.helper.save_batch(payload, filename)
            games += 1
        summary = replay_buffer_summary(str(replay_dir))
        if not finished_payloads and (time.perf_counter() - started) > 120.0:
            break

    return {
        "source": "bootstrap",
        "data_dir": str(replay_dir),
        "sample_count": int(summary.get("sample_count", 0)),
        "file_count": int(summary.get("file_count", 0)),
        "games": int(games),
        "duration_s": time.perf_counter() - started,
        "target_samples": int(target_samples),
    }


def prepare_phase3_replay_source(
    seed_run: dict | None,
    *,
    work_dir: Path,
    device: str,
    settings: dict,
    deadline: float | None = None,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    target_samples = max(
        int(settings.get("train_window_samples", 0)),
        int(settings.get("train_samples", 0)) + int(settings.get("eval_samples", 0)),
    )
    replay_mode = str(settings.get("replay_source", "auto")).strip().lower()
    live_summary = replay_buffer_summary(str(REPLAY_BUFFER_PATH))
    if replay_mode in {"live", "auto"} and int(live_summary.get("sample_count", 0)) >= max(1, target_samples):
        return {
            "source": "live",
            "data_dir": str(REPLAY_BUFFER_PATH),
            "sample_count": int(live_summary.get("sample_count", 0)),
            "file_count": int(live_summary.get("file_count", 0)),
            "target_samples": int(target_samples),
        }

    seed_trial = (seed_run or {}).get("best_trial") or {}
    seed_config = dict(seed_trial.get("config") or {})
    replay_dir = work_dir / "replay_buffer"
    model_path = resolve_phase3_base_checkpoint(work_dir)
    return _warmup_replay_buffer(
        replay_dir,
        device=device,
        model_path=model_path,
        target_samples=target_samples,
        concurrent_games=max(2, int(seed_config.get("selfplay_workers", 4))),
        simulations=max(16, int(settings.get("bootstrap_simulations", seed_config.get("selfplay_leaf_batch_size", 32)))),
        leaf_batch_size=max(4, int(seed_config.get("selfplay_leaf_batch_size", 16))),
        deadline=deadline,
    )


def prepare_phase3_datasets(
    data_dir: str,
    *,
    train_window_samples: int,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> dict:
    files, window_samples = latest_replay_window(data_dir, max_samples=max(1, int(train_window_samples)))
    if not files or window_samples <= 0:
        raise ValueError("Phase 3 could not find replay data to train on.")
    train_dataset = ReplayWindowDataset(
        files,
        sample_size=min(int(train_samples), int(window_samples)),
        rng_seed=int(seed),
    )
    eval_dataset = ReplayWindowDataset(
        files,
        sample_size=min(int(eval_samples), int(window_samples)),
        rng_seed=int(seed) + 1,
    )
    return {
        "files": files,
        "window_samples": int(window_samples),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "file_count": len(files),
        "train_sample_count": len(train_dataset),
        "eval_sample_count": len(eval_dataset),
    }
