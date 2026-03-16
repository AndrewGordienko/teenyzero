import os
import time
import torch
import argparse
import multiprocessing as mp

from teenyzero.alphazero.model import AlphaNet
from teenyzero.alphazero.servers.inference import inference_worker
from teenyzero.alphazero.logic.collector import DataCollector
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.visualizers.training_data_dashboard.dashboard import run_dashboard


def bootstrap_model(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[*] Initializing fresh AlphaNet at {path}...")
        model = AlphaNet(num_res_blocks=10, channels=128)
        torch.save(model.state_dict(), path)


def worker_task(worker_id, task_queue, response_queue, shared_stats):
    """
    Self-play worker loop.
    """
    evaluator = AlphaZeroEvaluator(
        task_queue=task_queue,
        response_queue=response_queue,
        worker_id=worker_id,
        use_cache=True,
    )

    mcts_params = {
        "SIMULATIONS": 32,
        "C_PUCT": 1.2,
        "VIRTUAL_LOSS": 0.0,
        "PARALLEL_THREADS": 1,
        "FPU_REDUCTION": 0.4,
        "LEAF_BATCH_SIZE": 8,
    }
    engine = MCTS(evaluator=evaluator, params=mcts_params)

    collector = DataCollector(
        evaluator,
        engine,
        buffer_path="teenyzero/alphazero/data/replay_buffer",
    )

    print(f"[Worker {worker_id}] Online and searching...")

    while True:
        game_data = collector.collect_game(worker_id=worker_id, stats_dict=shared_stats)

        timestamp = int(time.time() * 1000)
        filename = f"game_{worker_id}_{timestamp}.npz"
        collector.save_batch(game_data, filename)

        print(f"[Worker {worker_id}] Saved {len(game_data)} positions to {filename}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of self-play processes (defaults to 8 on accelerator-backed runs, else 4 on CPU)",
    )
    args = parser.parse_args()
    worker_count = args.workers if args.workers is not None else (8 if DEVICE != "cpu" else 4)

    bootstrap_model(MODEL_PATH)

    with mp.Manager() as manager:
        shared_stats = manager.dict()
        shared_stats["__cluster__"] = {
            "config": {
                "device": DEVICE,
                "workers": worker_count,
                "model_path": MODEL_PATH,
                "simulations": 32,
                "leaf_batch_size": 8,
            }
        }

        # Shared request queue into inference server
        task_queue = mp.Queue(maxsize=4096)

        # One direct response queue per worker
        response_queues = [mp.Queue(maxsize=1024) for _ in range(worker_count)]

        processes = []

        # Inference server
        inf_p = mp.Process(
            target=inference_worker,
            args=(MODEL_PATH, DEVICE, task_queue, response_queues, shared_stats),
        )
        inf_p.daemon = True
        inf_p.start()
        processes.append(inf_p)

        # Dashboard
        dash_p = mp.Process(target=run_dashboard, args=(shared_stats,))
        dash_p.daemon = True
        dash_p.start()
        processes.append(dash_p)

        # Self-play workers
        for i in range(worker_count):
            p = mp.Process(
                target=worker_task,
                args=(i, task_queue, response_queues[i], shared_stats),
            )
            p.daemon = True
            p.start()
            processes.append(p)

        print("\n" + "=" * 50)
        print("  TEENYZERO FACTORY ONLINE")
        print("=" * 50)
        print(f"[*] Workers:   {worker_count}")
        print(f"[*] Device:    {DEVICE}")
        print(f"[*] Dashboard: http://localhost:5002")
        print(f"[*] Data:      teenyzero/alphazero/data/replay_buffer/")
        print("Press Ctrl+C to stop cluster.")
        print("=" * 50 + "\n")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[!] Shutting down cluster...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=1.0)
