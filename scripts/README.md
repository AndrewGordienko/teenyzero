# Scripts

The runnable surface is intentionally flat now. Use the top-level scripts only.

- `scripts/benchmark_self_play.py`
  Benchmarks move latency and positions/sec for the current runtime/device/profile.
- `scripts/run_actors.py`
  Starts the self-play factory. On `mps`/`cuda` it now defaults to in-process batched self-play instead of the legacy queue/server layout.
- `scripts/run_arena.py`
  Runs checkpoint promotion, arena Elo tracking, and optional Stockfish anchors.
- `scripts/run_chess.py`
  Starts terminal self-play or the gameplay UI.
- `scripts/run_visualizers.py`
  Starts the full dashboard stack.
- `scripts/train.py`
  Runs the continuous trainer loop.

Common flags:

- `--device mps`
- `--device h200 --profile h200`
- `--profile mps`
- `--board-backend native`
- `--runtime-root /Volumes/External/teenyzero-runtime`
- `--tmpdir /Volumes/External/teenyzero-tmp`
- `--actor-workers 64`
- `--stockfish-path /path/to/stockfish`
- `--promotion-games 40 --baseline-games 24 --arena-simulations 640`
- `--stockfish-time-ms 100`
- `--no-dashboard` on `run_actors.py`
- `--actor-mode inprocess` or `--actor-mode mp`

Runtime state lives under `var/`:

- `var/models/` for live and archived checkpoints
- `var/data/` for replay shards and trainer/arena state
