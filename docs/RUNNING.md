# Running TeenyZero

## Python

Use the same interpreter you used to build the optional native extension.

Example:

```bash
python3.11 setup.py build_ext --inplace
```

## MPS

Benchmark:

```bash
python3.11 scripts/benchmark_self_play.py --device mps --profile mps --board-backend native --actor-mode inprocess --workers 4 --searches-per-worker 8
```

Fast self-play benchmark on MPS with the reduced-size profile:

```bash
python3.11 scripts/benchmark_self_play.py --device mps --profile mps_fast --board-backend native --actor-mode inprocess --workers 4 --searches-per-worker 8
```

Gameplay UI:

```bash
python3.11 scripts/run_chess.py --device mps --profile mps --board-backend native --mode visualize
```

Self-play factory:

```bash
python3.11 scripts/run_actors.py --device mps --profile mps --board-backend native --actor-mode inprocess --no-dashboard
```

Full dashboard stack:

```bash
python3.11 scripts/run_visualizers.py --device mps --profile mps --board-backend native
```

Gameplay-only visualizer on MPS without actors, trainer, or arena contending for the device:

```bash
python3.11 scripts/run_visualizers.py --device mps --profile mps --board-backend native --no-actors --no-trainer --no-arena --play-simulations 64
```

Gameplay-only visualizer on the reduced-size MPS profile:

```bash
python3.11 scripts/run_visualizers.py --device mps --profile mps_fast --board-backend native --no-actors --no-trainer --no-arena --play-simulations 48
```

## H200

Benchmark:

```bash
python3.11 scripts/benchmark_self_play.py --device h200 --profile h200 --board-backend native --actor-mode inprocess --workers 16 --searches-per-worker 8
```

Factory:

```bash
python3.11 scripts/run_actors.py --device h200 --profile h200 --board-backend native --actor-mode inprocess --no-dashboard
```

Full stack:

```bash
python3.11 scripts/run_visualizers.py --device h200 --profile h200 --board-backend native
```

## Runtime Data

Runtime artifacts now live under `var/` by default instead of under the source package.

- Live checkpoints: `var/models/best_model.pth`, `var/models/latest_model.pth`
- Replay buffer: `var/data/replay_buffer/`
- Trainer state: `var/data/training_state.json`
- Arena state: `var/data/arena_state.json`
- Archive checkpoints: `var/models/archive/`

Override with:

```bash
TEENYZERO_RUNTIME_ROOT=/abs/path/to/runtime python3.11 scripts/run_actors.py ...
```
