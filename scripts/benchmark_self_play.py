import argparse
import multiprocessing as mp
import os
import time

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

import chess
import numpy as np

from teenyzero.alphazero.backend import create_board
from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.logic.batched_selfplay import BatchedSelfPlayRunner
from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.alphazero.servers.inference import inference_worker
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.paths import BEST_MODEL_PATH, LATEST_MODEL_PATH, ensure_runtime_dirs


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


def benchmark_fens():
    lines = [
        [],
        ["e4", "e5", "Nf3", "Nc6"],
        ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"],
        ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6"],
        ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"],
    ]
    boards = []
    for san_line in lines:
        board = chess.Board()
        for san in san_line:
            board.push_san(san)
        boards.append(board.fen())
    return boards


def bootstrap_model(path, fallback_path=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = build_model()
    if os.path.exists(path):
        load_result = load_checkpoint(model, path, map_location="cpu", allow_partial=True)
        if load_result["loaded"]:
            return
        print(f"[*] Replacing unusable checkpoint at {path} ({load_result['reason']})...")
    elif fallback_path and os.path.exists(fallback_path):
        fallback_result = load_checkpoint(model, fallback_path, map_location="cpu", allow_partial=True)
        if fallback_result["loaded"]:
            print(f"[*] Initializing {path} from fallback checkpoint {fallback_path}...")
            save_checkpoint(model, path)
            return
    save_checkpoint(model, path)


def benchmark_worker(worker_id, fens, task_queue, response_queue, result_queue, simulations, leaf_batch_size):
    evaluator = AlphaZeroEvaluator(
        task_queue=task_queue,
        response_queue=response_queue,
        worker_id=worker_id,
        use_cache=True,
    )
    engine = MCTS(
        evaluator=evaluator,
        params={
            "SIMULATIONS": simulations,
            "C_PUCT": 1.5,
            "FPU_REDUCTION": 0.4,
            "LEAF_BATCH_SIZE": leaf_batch_size,
        },
    )

    totals = {
        "count": 0,
        "wall_s": 0.0,
        "simulations": 0,
        "positions": 0,
    }
    samples = {
        "total_ms": [],
        "selection_ms": [],
        "leaf_eval_ms": [],
        "backprop_ms": [],
        "wait_ms": [],
        "forward_ms": [],
    }

    for fen in fens:
        board = create_board(fen)
        started = time.perf_counter()
        engine.search(board, is_training=False)
        wall_s = time.perf_counter() - started
        stats = engine.last_search_stats
        timings = stats.get("timings_ms", {})
        evaluator_stats = stats.get("evaluator", {})

        totals["count"] += 1
        totals["wall_s"] += wall_s
        totals["simulations"] += int(stats.get("simulations_completed", 0))
        totals["positions"] += int(evaluator_stats.get("positions_evaluated", 0))
        samples["total_ms"].append(float(timings.get("total", 0.0)))
        samples["selection_ms"].append(float(timings.get("selection", 0.0)))
        samples["leaf_eval_ms"].append(float(timings.get("leaf_eval", 0.0)))
        samples["backprop_ms"].append(float(timings.get("backprop", 0.0)))
        samples["wait_ms"].append(float(timings.get("inference_wait", 0.0)))
        samples["forward_ms"].append(float(timings.get("inference_forward", 0.0)))

    result_queue.put({
        "totals": totals,
        "samples": samples,
    })


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def run_inprocess_benchmark(device, workers, searches_per_worker, simulations, leaf_batch_size):
    model = build_model()
    load_checkpoint(model, LATEST_MODEL_PATH, map_location="cpu", allow_partial=True)
    evaluator = AlphaZeroEvaluator(model=model, device=device, use_cache=True)
    engine = MCTS(
        evaluator=evaluator,
        params={
            "SIMULATIONS": simulations,
            "C_PUCT": 1.5,
            "FPU_REDUCTION": 0.4,
            "LEAF_BATCH_SIZE": leaf_batch_size,
        },
    )
    runner = BatchedSelfPlayRunner(
        evaluator=evaluator,
        engine=engine,
        buffer_path="var/data/replay_buffer",
        concurrent_games=workers,
    )
    base_fens = benchmark_fens()
    slots = []
    for idx in range(workers):
        slot = runner._new_game(idx)
        slot.board = create_board(base_fens[idx % len(base_fens)])
        slots.append(slot)

    totals = {
        "count": 0,
        "wall_s": 0.0,
        "simulations": 0,
        "positions": 0,
    }
    samples = {
        "total_ms": [],
        "selection_ms": [],
        "leaf_eval_ms": [],
        "backprop_ms": [],
        "wait_ms": [],
        "forward_ms": [],
    }

    for _ in range(searches_per_worker):
        started = time.perf_counter()
        runner._play_batch_move(slots)
        wall_s = time.perf_counter() - started
        stats = runner.helper.last_profile or {}
        timings = stats.get("timings_ms", {})
        evaluator_stats = stats.get("evaluator", {})
        active = max(1, len(slots))
        totals["count"] += active
        totals["wall_s"] += wall_s
        totals["simulations"] += int(stats.get("simulations_completed", 0))
        totals["positions"] += int(evaluator_stats.get("positions_evaluated", 0))
        for key, target in (
            ("total", "total_ms"),
            ("selection", "selection_ms"),
            ("leaf_eval", "leaf_eval_ms"),
            ("backprop", "backprop_ms"),
            ("inference_wait", "wait_ms"),
            ("inference_forward", "forward_ms"),
        ):
            per_move_value = float(timings.get(key, 0.0)) / float(active)
            samples[target].extend([per_move_value] * active)

    return totals, samples


def main():
    parser = argparse.ArgumentParser(description="Benchmark the TeenyZero self-play inference/MCTS path.")
    parser.add_argument("--workers", type=int, default=PROFILE.selfplay_workers)
    parser.add_argument("--searches-per-worker", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=PROFILE.selfplay_simulations)
    parser.add_argument("--leaf-batch-size", type=int, default=PROFILE.selfplay_leaf_batch_size)
    parser.add_argument(
        "--actor-mode",
        choices=["auto", "inprocess", "mp"],
        default="auto",
        help="Benchmark the in-process batched self-play path or the legacy mp queue/server path.",
    )
    args = parser.parse_args()

    device = RUNTIME.device
    ensure_runtime_dirs()
    bootstrap_model(str(BEST_MODEL_PATH))
    model_path = str(LATEST_MODEL_PATH)
    bootstrap_model(model_path, fallback_path=str(BEST_MODEL_PATH))
    actor_mode = args.actor_mode
    if actor_mode == "auto":
        actor_mode = "inprocess" if device in {"cuda", "mps"} else "mp"

    if actor_mode == "inprocess":
        started = time.perf_counter()
        totals, merged = run_inprocess_benchmark(
            device=device,
            workers=args.workers,
            searches_per_worker=args.searches_per_worker,
            simulations=args.simulations,
            leaf_batch_size=args.leaf_batch_size,
        )
        elapsed = time.perf_counter() - started
    else:
        base_fens = benchmark_fens()
        fens = [base_fens[i % len(base_fens)] for i in range(args.searches_per_worker)]

        mp.set_start_method("spawn", force=True)
        task_queue = mp.Queue(maxsize=4096)
        response_queues = [mp.Queue(maxsize=1024) for _ in range(args.workers)]
        result_queue = mp.Queue()

        inference = mp.Process(
            target=inference_worker,
            args=(model_path, device, task_queue, response_queues, None),
        )
        inference.start()

        workers = []
        started = time.perf_counter()
        for worker_id in range(args.workers):
            proc = mp.Process(
                target=benchmark_worker,
                args=(
                    worker_id,
                    fens,
                    task_queue,
                    response_queues[worker_id],
                    result_queue,
                    args.simulations,
                    args.leaf_batch_size,
                ),
            )
            proc.start()
            workers.append(proc)

        results = [result_queue.get() for _ in workers]
        elapsed = time.perf_counter() - started

        for proc in workers:
            proc.join()
        inference.terminate()
        inference.join(timeout=1.0)

        totals = {
            "count": sum(item["totals"]["count"] for item in results),
            "wall_s": sum(item["totals"]["wall_s"] for item in results),
            "simulations": sum(item["totals"]["simulations"] for item in results),
            "positions": sum(item["totals"]["positions"] for item in results),
        }
        merged = {
            key: [value for item in results for value in item["samples"][key]]
            for key in results[0]["samples"].keys()
        }

    print("\nTeenyZero Benchmark")
    print(f"Device: {device}")
    print(f"Profile: {PROFILE.name}")
    print(f"Mode: {actor_mode}")
    print(f"Workers: {args.workers}")
    print(f"Searches per worker: {args.searches_per_worker}")
    print(f"Simulations: {args.simulations}")
    print(f"Leaf batch size: {args.leaf_batch_size}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Searches: {totals['count']}")
    print(f"Searches/sec: {totals['count'] / max(elapsed, 1e-9):.2f}")
    print(f"Simulations/sec: {totals['simulations'] / max(elapsed, 1e-9):.2f}")
    print(f"Positions/sec: {totals['positions'] / max(elapsed, 1e-9):.2f}")

    for key, label in (
        ("total_ms", "Move Total"),
        ("selection_ms", "Selection"),
        ("leaf_eval_ms", "Leaf Eval"),
        ("backprop_ms", "Backprop"),
        ("wait_ms", "Inference Wait"),
        ("forward_ms", "Model Forward"),
    ):
        stats = summarize(merged[key])
        print(
            f"{label}: mean {stats['mean']:.1f} ms | "
            f"p50 {stats['p50']:.1f} ms | p95 {stats['p95']:.1f} ms"
        )


if __name__ == "__main__":
    main()
