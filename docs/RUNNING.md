# Running TeenyZero

This guide is meant to answer two questions:

1. "How do I get this repo working on my machine?"
2. "Which command should I run for the thing I actually want to do?"

The short version is:

- install the package in editable mode
- optionally build the native extension
- pick one launcher depending on whether you want gameplay, self-play,
  training, or the full dashboard stack

If you are brand new to the repo, start with the quick-start tutorial below.

In the commands below, `python3` means "the exact interpreter you chose for
this repo." If your environment is on `python3.14`, use `python3.14`
everywhere. If it is on `python3.11`, use `python3.11` everywhere.

## First-Time Setup

### 1. Pick one Python interpreter and stick to it

Use the same interpreter for:

- `pip install`
- `setup.py build_ext --inplace`
- every `python ...` command you run afterward

That matters because the optional native extension is built for a specific
CPython version. If you build it with `python3.14` and run scripts with
`python3.11`, you are using two different environments.

### 2. Install the repo in editable mode

From the repo root:

```bash
python3 -m pip install -e .
```

If you skip this step and run a script directly, Python may fail to import
`teenyzero` from the repo root.

### 3. Optionally build the native extension

TeenyZero works without the native extension, but the optional speedups are
worth building if you plan to use `--board-backend native`.

```bash
python3 setup.py build_ext --inplace
```

The build is optional by design. If the extension is unavailable, TeenyZero can
fall back to the plain Python board path.

## Quick Start

If you want the smoothest "show me the system" experience, follow this path.

### Step 1. Launch the gameplay-only visualizer

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps_fast \
  --board-backend native \
  --no-actors \
  --no-trainer \
  --no-arena \
  --play-simulations 48
```

What this does:

- starts the main Flask app on `http://localhost:5001`
- loads a local search/evaluation stack for the board UI
- does not start self-play, training, or arena background processes

Use this when you want the lightest possible interactive experience.

### Step 2. Launch the full dashboard stack

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps \
  --board-backend native
```

What this does:

- starts the main app on `http://localhost:5001`
- starts self-play actors
- starts the trainer
- starts the arena
- exposes the actor dashboard on `http://localhost:5002`

Use this when you want the actual learning loop running in the background while
you watch or interact with the UI.

### Step 3. Watch the runtime state on disk

Once the system is running, most of the important state shows up under `var/`:

- `var/models/latest_model.pth`
- `var/models/best_model.pth`
- `var/models/archive/`
- `var/data/replay_buffer/`
- `var/data/training_state.json`
- `var/data/arena_state.json`

If you want to understand whether the loop is healthy, looking in `var/` is
often more useful than staring at logs.

## Which Command Should I Run?

This is the practical routing table.

### I want the whole lab

Run:

```bash
python3 scripts/run_visualizers.py --device mps --profile mps --board-backend native
```

This is the best default command when you want:

- the main UI
- self-play
- training
- arena evaluation
- live dashboards

Think of it as the "bring the whole project online" launcher.

### I want just the board UI

Run:

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps_fast \
  --board-backend native \
  --no-actors \
  --no-trainer \
  --no-arena
```

This keeps the UI responsive because the device is not being shared with
background self-play, training, and arena work.

### I want terminal self-play without the web app

Run:

```bash
python3 scripts/run_chess.py --device mps --profile mps --board-backend native --mode play
```

This is a simple terminal loop where the engine plays through a game and prints
the board after each move.

### I want the web gameplay UI only, without the orchestrator

Run:

```bash
python3 scripts/run_chess.py --device mps --profile mps --board-backend native --mode visualize
```

This launches the gameplay app directly on port `5001`.
It is useful when you want a simpler single-purpose path than
`run_visualizers.py`.

### I want to generate self-play data

Run:

```bash
python3 scripts/run_actors.py --device mps --profile mps --board-backend native --actor-mode inprocess
```

What this does:

- loads the current checkpoint
- starts self-play workers
- writes replay shards into `var/data/replay_buffer/`
- optionally starts the actor dashboard on `5002`

Use this when you are tuning self-play throughput or debugging data generation.

### I want max self-play throughput without the dashboard

Run:

```bash
python3 scripts/run_actors.py \
  --device mps \
  --profile mps \
  --board-backend native \
  --actor-mode inprocess \
  --no-dashboard
```

This removes the live stats process and shared manager overhead.

### I want to train from the replay buffer

Run:

```bash
python3 scripts/train.py --device mps --profile mps --board-backend native
```

The trainer is a long-running watcher, not a one-shot command. It waits for
enough replay data, trains a cycle, writes new checkpoints, updates
`training_state.json`, and goes back to waiting.

### I want to evaluate and promote new checkpoints

Run:

```bash
python3 scripts/run_arena.py --device mps --profile mps --board-backend native
```

The arena compares `latest_model.pth` against the current `best_model.pth` and
updates ratings, promotion state, and archives.

### I want to benchmark self-play/search throughput

Run:

```bash
python3 scripts/benchmark_self_play.py \
  --device mps \
  --profile mps \
  --board-backend native \
  --actor-mode inprocess \
  --workers 4 \
  --searches-per-worker 8
```

This is the command to use when you are measuring:

- searches per second
- simulations per second
- time spent in selection, leaf eval, and backprop
- inference wait and forward timings

It is a profiling tool, not part of the normal training loop.

## Runtime Profiles

Profiles are named operating modes defined in
[`teenyzero/alphazero/runtime.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/runtime.py).

The important ones for day-to-day use are:

- `local`
  conservative CPU-friendly defaults
- `mps`
  fuller Apple Silicon profile
- `mps_fast`
  lighter Apple Silicon profile for quick iteration
- `cuda_fast`
  general CUDA profile
- `h100`
  large CUDA profile for H100-class hardware
- `h200`
  larger CUDA profile for H200-class hardware

You can specify both `--device` and `--profile`, but in practice:

- `--device` chooses where tensors run
- `--profile` chooses how aggressively the whole loop behaves

Examples:

```bash
python3 scripts/run_visualizers.py --device mps --profile mps_fast
python3 scripts/run_visualizers.py --device h200 --profile h200
python3 scripts/run_actors.py --device cpu --profile local
```

## Important Flags

These flags show up across the launcher scripts because
[`teenyzero/runtime_bootstrap.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/runtime_bootstrap.py)
handles them early.

### `--device`

Examples:

```bash
--device cpu
--device mps
--device cuda
--device h100
--device h200
```

This selects the target compute device.
`h100` and `h200` are treated as CUDA devices plus matching profile hints.

### `--profile`

Examples:

```bash
--profile local
--profile mps
--profile mps_fast
--profile h200
```

This selects the runtime profile: batch sizes, worker counts, model scale,
training thresholds, and other linked settings.

### `--board-backend`

Examples:

```bash
--board-backend python
--board-backend native
--board-backend auto
```

Use `native` if you built the extension and want the optional speedups.
Use `python` if you want the simplest possible debugging path.

### `--runtime-root`

Example:

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps \
  --runtime-root /Volumes/FastSSD/teenyzero-runtime
```

This moves the whole runtime state tree off the default `var/` directory.
It is useful when replay data or temp files should live on a bigger or faster
volume.

### `--tmpdir`

Example:

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps \
  --tmpdir /Volumes/FastSSD/tmp
```

This is mainly helpful on macOS when MPS graph compilation or temp-file churn
is stressing your default disk.

## Common Workflows

### Workflow: interactive UI work

If you are tweaking the board UI or search behavior, use:

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps_fast \
  --board-backend native \
  --no-actors \
  --no-trainer \
  --no-arena
```

That keeps latency down and removes background noise.

### Workflow: self-play tuning

If you are tuning actor counts or leaf batch sizes, use:

```bash
python3 scripts/run_actors.py \
  --device mps \
  --profile mps \
  --board-backend native \
  --actor-mode inprocess \
  --workers 4 \
  --leaf-batch-size 24
```

This isolates the self-play side of the system.

### Workflow: end-to-end training run

If you want the whole loop alive, use:

```bash
python3 scripts/run_visualizers.py --device mps --profile mps --board-backend native
```

This is the best "just run the project" command.

## Runtime Data

By default, runtime artifacts live under `var/` instead of inside the package.

- Live checkpoints: `var/models/best_model.pth`, `var/models/latest_model.pth`
- Replay buffer: `var/data/replay_buffer/`
- Trainer state: `var/data/training_state.json`
- Trainer history: `var/data/training_history.json`
- Arena state: `var/data/arena_state.json`
- Arena history: `var/data/arena_history.json`
- Match log: `var/data/arena_matches.json`
- Archive checkpoints: `var/models/archive/`

This is intentional. The package code stays relatively clean, and the runtime
state stays easy to inspect, move, and delete.

## Troubleshooting

### `ModuleNotFoundError: No module named 'teenyzero'`

Install the repo in editable mode:

```bash
python3 -m pip install -e .
```

### Native warnings or missing native module

Rebuild the extension with the same interpreter you plan to run:

```bash
python3 setup.py build_ext --inplace
```

If you do not care about native speedups, run with:

```bash
--board-backend python
```

### The UI feels sluggish on MPS

If actors, trainer, and arena are sharing the same device, the gameplay board
will feel slower.

Use the gameplay-only mode:

```bash
python3 scripts/run_visualizers.py \
  --device mps \
  --profile mps_fast \
  --board-backend native \
  --no-actors \
  --no-trainer \
  --no-arena
```

### I am not sure which command to start with

Start here:

```bash
python3 scripts/run_visualizers.py --device mps --profile mps_fast --board-backend native --no-actors --no-trainer --no-arena
```

Then, once that works, move up to the full stack command.
