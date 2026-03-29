# TeenyZero Architecture

This document is the "how the machine fits together" view of TeenyZero.
If [RUNNING.md](./RUNNING.md) is about getting the system on its feet,
this file is about understanding the moving parts once it is alive.

TeenyZero is an AlphaZero-style training loop built around a few simple ideas:

1. Self-play generates fresh games.
2. The trainer turns those games into better weights.
3. The arena decides whether the new weights deserve promotion.
4. The visualizers make the whole loop inspectable while it runs.

The codebase is still intentionally direct. Most behavior is driven by a small
number of scripts and plain files under `var/`, not by a complex service layer.

## The Big Picture

The easiest way to think about the system is as a data pipeline:

```text
latest_model.pth
        |
        v
self-play actors -----> replay buffer (.npz game shards)
        |                            |
        |                            v
        |                       trainer
        |                            |
        |                            v
        |                   latest_model.pth / best_model.pth
        |                            |
        +----------------------------+
                                     |
                                     v
                                  arena
                                     |
                                     v
                         promotion / archive / ratings
```

At the edge of that loop sit two UIs:

- the main Flask app on port `5001`
- the cluster dashboard on port `5002`

Those UIs are not the source of truth. The source of truth is the runtime state
under `var/`: checkpoints, replay shards, JSON state files, and model archives.

## The Runtime Model

One of the most important design choices in this repo is that runtime selection
happens before the heavy imports.

### 1. CLI bootstrap

[`teenyzero/runtime_bootstrap.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/runtime_bootstrap.py)
parses a small set of common flags early:

- `--device`
- `--profile`
- `--board-backend`
- `--runtime-root`
- `--tmpdir`

It translates those flags into environment variables before modules like the
model config or runtime profile are imported. That keeps the rest of the code
simple: later modules can just read the active runtime settings.

### 2. Runtime profiles

[`teenyzero/alphazero/runtime.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/runtime.py)
defines named profiles such as `local`, `mps`, `mps_fast`, `cuda_fast`,
`h100`, and `h200`.

Each profile bundles the knobs that usually need to move together:

- model size
- replay window sizes
- train batch sizes
- self-play worker budgets
- MCTS simulation counts
- inference precision and compile settings

The important idea is that a profile is not just a model architecture preset.
It is a whole operating mode for the training loop.

### 3. Runtime paths

[`teenyzero/paths.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/paths.py)
keeps all runtime artifacts under a single root, which defaults to `var/`.

That gives the project one boring, inspectable place for state:

- `var/models/latest_model.pth`
- `var/models/best_model.pth`
- `var/models/archive/`
- `var/data/replay_buffer/`
- `var/data/training_state.json`
- `var/data/arena_state.json`

This is why TeenyZero feels closer to a lab bench than a hidden framework:
most of the important state is just sitting on disk in obvious places.

## Search And Inference

The search stack is split into a few focused modules.

### `AlphaZeroEvaluator`

[`teenyzero/mcts/evaluator.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/evaluator.py)
is the bridge between chess positions and the neural network.

It is responsible for:

- encoding board state into input planes
- mapping legal moves into policy indices
- masking illegal policy logits
- running local inference or queue-based batched inference
- caching repeated evaluations when appropriate

This is where "what does the model think about this position?" becomes a usable
policy/value pair for search.

### `MCTS`

[`teenyzero/mcts/search.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/search.py)
implements the actual tree search.

The implementation is still pretty explicit:

- prepare the root
- collect a batch of leaves
- evaluate those leaves
- expand and backpropagate
- repeat for the requested number of simulations

That straight-line structure is useful when profiling because selection,
evaluation, and backprop timings remain visible as separate phases.

### `SearchSession`

[`teenyzero/alphazero/search_session.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/search_session.py)
holds onto the current root across moves.

That matters for interactive play and arena matches. Instead of rebuilding the
entire tree every move, the session can advance the root after a move is played
and reuse part of the previous search work.

## Python Board Path vs Native Helpers

TeenyZero has optional native speedups, but it is not pretending to be a full
native engine yet.

[`teenyzero/alphazero/backend.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/backend.py)
chooses whether to use:

- plain Python helpers via `python-chess`
- optional native helpers from `teenyzero.native._speedups`

[`native/speedups.cpp`](/Users/andrewgordienko/Documents/TeenyZero/native/speedups.cpp)
and [`native/board.cpp`](/Users/andrewgordienko/Documents/TeenyZero/native/board.cpp)
speed up pieces of board handling and move-index plumbing, but legal move
generation and the overall chess API still fundamentally follow `python-chess`.

That tradeoff is deliberate. The project gets some low-level acceleration
without forcing the whole engine into a harder-to-debug codepath.

## Self-Play

Self-play is launched by
[`scripts/run_actors.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_actors.py).

This script has two operating modes:

- `inprocess`
- `mp`

### In-process mode

On `mps` and `cuda`, the default is usually `inprocess`.

In this mode:

- one process owns the model on the target device
- self-play runs as a batched loop in that process
- inference stays local instead of bouncing through multiprocessing queues

This is the lower-overhead path for a single GPU or Apple Silicon setup.

### Multiprocess mode

In `mp` mode:

- worker processes generate search requests
- an inference server process batches those requests
- results flow back through response queues

This is closer to a classic actor/inference split. It is easier to reason about
as a distributed design, but it costs more in queueing and process overhead.

### Output

Self-play writes replay shards into:

- `var/data/replay_buffer/`

Those shards are the trainer's raw material.

## Training

[`scripts/train.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/train.py)
is a long-running loop that watches the replay buffer and trains whenever
enough fresh samples have accumulated.

Conceptually it does five things:

1. Scan the replay buffer.
2. Decide whether enough new data exists for a training cycle.
3. Build a replay window.
4. Train for one cycle.
5. Save state, checkpoints, history, and prune old data.

The trainer writes status to JSON so the rest of the system can inspect it
without RPC plumbing:

- `training_state.json` for current state
- `training_history.json` for recent completed cycles

The trainer also updates checkpoints under `var/models/`.

## Arena And Promotion

[`scripts/run_arena.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_arena.py)
decides whether `latest_model.pth` should replace `best_model.pth`.

It compares a candidate checkpoint against:

- the current champion
- recent archived champions
- optionally a Stockfish anchor, if configured

This gives the project a promotion gate instead of blindly assuming the newest
weights are better.

The arena keeps a small amount of persistent bookkeeping:

- ratings
- recent matches
- promotion thresholds
- archive paths
- external engine availability

That state is recorded in `arena_state.json`, `arena_history.json`, and
`arena_matches.json`.

## Visualizers

[`scripts/run_visualizers.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_visualizers.py)
is the convenience launcher for the "full lab" experience.

It starts the main Flask app and, unless told otherwise, also starts:

- self-play actors
- the trainer
- the arena

The main app lives in
[`teenyzero/visualizers/app.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/visualizers/app.py).
It is less a static frontend and more a control room:

- gameplay board
- trainer state
- arena state
- runtime metadata
- process controls

The cluster monitor in
[`teenyzero/visualizers/cluster_monitor/dashboard.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/visualizers/cluster_monitor/dashboard.py)
is a separate Flask app on port `5002` for live actor-side stats.

## File-Level Map

If you want to start reading the codebase, this order usually works well:

1. [`scripts/run_visualizers.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_visualizers.py)
   for the top-level process model
2. [`teenyzero/alphazero/runtime.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/runtime.py)
   for profile selection
3. [`teenyzero/paths.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/paths.py)
   for runtime state layout
4. [`scripts/run_actors.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_actors.py)
   for self-play generation
5. [`scripts/train.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/train.py)
   for checkpoint production
6. [`scripts/run_arena.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_arena.py)
   for promotion logic
7. [`teenyzero/mcts/evaluator.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/evaluator.py)
   and [`teenyzero/mcts/search.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/search.py)
   for the core search path

## Where To Hack

If your goal is to change behavior, here is the shortest routing table:

- Want a different model shape or training scale:
  edit [`teenyzero/alphazero/runtime.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/alphazero/runtime.py)
- Want to change search behavior:
  edit [`teenyzero/mcts/search.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/search.py)
- Want to change encoding or policy masking:
  edit [`teenyzero/mcts/evaluator.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/mcts/evaluator.py)
- Want to change self-play orchestration:
  edit [`scripts/run_actors.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_actors.py)
- Want to change training cadence or checkpointing:
  edit [`scripts/train.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/train.py)
- Want to change promotion rules:
  edit [`scripts/run_arena.py`](/Users/andrewgordienko/Documents/TeenyZero/scripts/run_arena.py)
- Want to change the UI/control room:
  edit [`teenyzero/visualizers/app.py`](/Users/andrewgordienko/Documents/TeenyZero/teenyzero/visualizers/app.py)

## Current Limits

The current architecture is practical, not mystical.

- Native support is optional rather than foundational.
- Runtime coordination is mostly file-based and process-based.
- The main launchers are scripts, not a packaged service supervisor.
- Visualizers read state produced by the training loop; they do not own it.

That simplicity is a strength right now. The system is small enough to follow in
a debugger, and most of the important runtime state can be inspected by opening
files under `var/`.
