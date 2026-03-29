# Autotune Results

This file tracks promoted autotune runtime recommendations that can be shared in the repo.
Each entry comes from an autotune run and captures a recommended hardware/runtime setup for a workload.

The source of truth is the checked-in registry at `teenyzero/autotune/catalog/recommendations.json`.

| ID | Title | Workload | Seed | Best Trial | Self-Play Pos/Sec | Train Samples/Sec |
| --- | --- | --- | --- | --- | ---: | ---: |
| apple_silicon_mps_balanced | Apple Silicon MPS 24GB Balanced | balanced | mps/mps/native | Q02 | 891.9 | 381.0 |

## Apple Silicon MPS 24GB Balanced

- `id`: `apple_silicon_mps_balanced`
- `workload`: `balanced`
- `device family`: `apple_silicon_mps`
- `score`: `1.013`
- `self-play positions/sec`: `891.9`
- `train samples/sec`: `381.0`
- `best config`: `mode=inprocess`, `workers=8`, `leaf=48`, `train_batch=144`, `train_workers=4`, `precision=fp16`, `compile=False`
- `summary`: Promoted from phase3 autotune run 20260329_162359.

```bash
python3 scripts/run_visualizers.py --device mps --profile mps --board-backend native --actor-mode inprocess --actor-workers 8 --selfplay-leaf-batch-size 48 --train-batch-size 144 --train-num-workers 4 --train-precision fp16 --no-train-pin-memory --no-train-compile
```
