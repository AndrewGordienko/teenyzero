import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.paths import runtime_free_bytes, runtime_low_disk_watermark_bytes, runtime_paths_payload


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-actors", action="store_true", help="Do not start the self-play actor process.")
    parser.add_argument("--no-trainer", action="store_true", help="Do not start the trainer process.")
    parser.add_argument("--no-arena", action="store_true", help="Do not start the arena process.")
    parser.add_argument(
        "--actor-mode",
        choices=["auto", "inprocess", "mp"],
        default=None,
        help="Override the self-play execution model used by run_actors.py.",
    )
    parser.add_argument("--actor-workers", type=int, default=None, help="Override self-play concurrent games/workers.")
    parser.add_argument("--selfplay-simulations", type=int, default=None, help="Override self-play simulations per move.")
    parser.add_argument(
        "--selfplay-leaf-batch-size",
        type=int,
        default=None,
        help="Override self-play leaf evaluation batch size.",
    )
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override trainer batch size.")
    parser.add_argument("--train-num-workers", type=int, default=None, help="Override trainer DataLoader worker count.")
    parser.add_argument("--train-optimizer", choices=["sgd", "adam", "adamw"], default=None, help="Override trainer optimizer.")
    parser.add_argument("--train-lr", type=float, default=None, help="Override trainer learning rate.")
    parser.add_argument("--train-weight-decay", type=float, default=None, help="Override trainer weight decay.")
    parser.add_argument("--train-grad-accum-steps", type=int, default=None, help="Override trainer grad accumulation steps.")
    parser.add_argument("--replay-window-samples", type=int, default=None, help="Override trainer replay window size.")
    parser.add_argument("--train-samples-per-cycle", type=int, default=None, help="Override trainer sampled positions per cycle.")
    parser.add_argument(
        "--train-precision",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="Override trainer precision mode.",
    )
    parser.add_argument("--train-compile", action="store_true", help="Enable trainer torch.compile mode.")
    parser.add_argument("--no-train-compile", action="store_true", help="Disable trainer torch.compile mode.")
    parser.add_argument("--train-pin-memory", action="store_true", help="Enable trainer DataLoader pin memory.")
    parser.add_argument("--no-train-pin-memory", action="store_true", help="Disable trainer DataLoader pin memory.")
    parser.add_argument("--stockfish-path", default=None, help="Path to a Stockfish binary for arena anchor matches.")
    parser.add_argument("--promotion-games", type=int, default=None, help="Override arena promotion games.")
    parser.add_argument("--baseline-games", type=int, default=None, help="Override arena baseline games per opponent.")
    parser.add_argument("--arena-simulations", type=int, default=None, help="Override arena MCTS simulations.")
    parser.add_argument("--stockfish-time-ms", type=int, default=None, help="Override Stockfish move time in milliseconds.")
    parser.add_argument(
        "--play-simulations",
        type=int,
        default=None,
        help="Override the gameplay board MCTS simulations in the visualizer app.",
    )
    args = parser.parse_args()

    if args.play_simulations is not None:
        os.environ["TEENYZERO_PLAY_SIMULATIONS"] = str(max(1, int(args.play_simulations)))
    if args.actor_workers is not None:
        os.environ["TEENYZERO_SELFPLAY_WORKERS"] = str(max(1, int(args.actor_workers)))
    if args.actor_mode is not None:
        os.environ["TEENYZERO_ACTOR_MODE"] = str(args.actor_mode)
    if args.selfplay_simulations is not None:
        os.environ["TEENYZERO_SELFPLAY_SIMULATIONS"] = str(max(1, int(args.selfplay_simulations)))
    if args.selfplay_leaf_batch_size is not None:
        os.environ["TEENYZERO_SELFPLAY_LEAF_BATCH_SIZE"] = str(max(1, int(args.selfplay_leaf_batch_size)))
    if args.train_batch_size is not None:
        os.environ["TEENYZERO_TRAIN_BATCH_SIZE"] = str(max(1, int(args.train_batch_size)))
    if args.train_num_workers is not None:
        os.environ["TEENYZERO_TRAIN_NUM_WORKERS"] = str(max(0, int(args.train_num_workers)))
    if args.train_optimizer is not None:
        os.environ["TEENYZERO_TRAIN_OPTIMIZER"] = str(args.train_optimizer)
    if args.train_lr is not None:
        os.environ["TEENYZERO_TRAIN_LR"] = str(max(1e-8, float(args.train_lr)))
    if args.train_weight_decay is not None:
        os.environ["TEENYZERO_TRAIN_WEIGHT_DECAY"] = str(max(0.0, float(args.train_weight_decay)))
    if args.train_grad_accum_steps is not None:
        os.environ["TEENYZERO_TRAIN_GRAD_ACCUM_STEPS"] = str(max(1, int(args.train_grad_accum_steps)))
    if args.replay_window_samples is not None:
        os.environ["TEENYZERO_REPLAY_WINDOW_SAMPLES"] = str(max(1, int(args.replay_window_samples)))
    if args.train_samples_per_cycle is not None:
        os.environ["TEENYZERO_TRAIN_SAMPLES_PER_CYCLE"] = str(max(1, int(args.train_samples_per_cycle)))
    if args.train_precision is not None:
        os.environ["TEENYZERO_TRAIN_PRECISION"] = str(args.train_precision)
    if args.train_compile:
        os.environ["TEENYZERO_TRAIN_COMPILE"] = "1"
    if args.no_train_compile:
        os.environ["TEENYZERO_TRAIN_COMPILE"] = "0"
    if args.train_pin_memory:
        os.environ["TEENYZERO_TRAIN_PIN_MEMORY"] = "1"
    if args.no_train_pin_memory:
        os.environ["TEENYZERO_TRAIN_PIN_MEMORY"] = "0"
    if args.stockfish_path:
        os.environ["TEENYZERO_STOCKFISH_PATH"] = str(args.stockfish_path)
    if args.promotion_games is not None:
        os.environ["TEENYZERO_ARENA_PROMOTION_GAMES"] = str(max(1, int(args.promotion_games)))
    if args.baseline_games is not None:
        os.environ["TEENYZERO_ARENA_BASELINE_GAMES"] = str(max(1, int(args.baseline_games)))
    if args.arena_simulations is not None:
        os.environ["TEENYZERO_ARENA_SIMULATIONS"] = str(max(1, int(args.arena_simulations)))
    if args.stockfish_time_ms is not None:
        os.environ["TEENYZERO_STOCKFISH_TIME_MS"] = str(max(1, int(args.stockfish_time_ms)))

    runtime = get_runtime_selection()
    profile = runtime.profile
    from teenyzero.visualizers.app import app

    actor_process = None
    trainer_process = None
    arena_process = None

    print(f"[*] Launching TeenyZero dashboard stack with runtime profile: {profile.name} on {runtime.device}")
    free_bytes = runtime_free_bytes()
    low_disk_watermark = runtime_low_disk_watermark_bytes()
    if free_bytes <= low_disk_watermark:
        paths_payload = runtime_paths_payload()
        print(
            "[*] Warning: low disk headroom detected "
            f"({free_bytes / (1024 * 1024):.0f} MiB free, target {low_disk_watermark / (1024 * 1024):.0f} MiB). "
            f"Runtime root: {paths_payload['runtime_root']}"
        )
        print("[*] If MPS crashes while creating temp graphs, relaunch with `--runtime-root` and `--tmpdir` on another volume.")
    if runtime.device in {"mps", "cuda"} and not (args.no_actors and args.no_trainer and args.no_arena):
        print("[*] Warning: background actors/trainer/arena share the same device and will increase play-board latency.")

    if not args.no_actors and not _port_in_use(5002):
        run_actors = _project_root() / "scripts" / "run_actors.py"
        actor_process = subprocess.Popen(
            [sys.executable, str(run_actors)],
            cwd=str(_project_root()),
        )

    if not args.no_trainer:
        run_trainer = _project_root() / "scripts" / "train.py"
        trainer_process = subprocess.Popen(
            [sys.executable, str(run_trainer)],
            cwd=str(_project_root()),
        )

    if not args.no_arena:
        run_arena = _project_root() / "scripts" / "run_arena.py"
        arena_process = subprocess.Popen(
            [sys.executable, str(run_arena)],
            cwd=str(_project_root()),
        )

    try:
        app.run(debug=False, port=5001, host="0.0.0.0")
    finally:
        if actor_process is not None and actor_process.poll() is None:
            actor_process.terminate()
        if trainer_process is not None and trainer_process.poll() is None:
            trainer_process.terminate()
        if arena_process is not None and arena_process.poll() is None:
            arena_process.terminate()
