# Autotune Results

This file tracks promoted autotune recommendations that can be shared in the repo.
Each entry comes from an autotune run and captures a recommended hardware, runtime, and profile-level setup for a workload.

The source of truth is the checked-in registry at `teenyzero/autotune/catalog/recommendations.json`.

| ID | Title | Workload | Phase | Seed | Best Trial | Self-Play Pos/Sec | Train Samples/Sec |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| apple_silicon_mps_balanced | Apple Silicon MPS 24GB Balanced | balanced | phase4 | mps/mps/native | H16 | 990.7 | 663.1 |

## Apple Silicon MPS 24GB Balanced

- `id`: `apple_silicon_mps_balanced`
- `workload`: `balanced`
- `device family`: `apple_silicon_mps`
- `source phase`: `phase4`
- `score`: `1.220`
- `self-play positions/sec`: `990.7`
- `train samples/sec`: `663.1`
- `best runtime config`: `mode=inprocess workers=6 leaf=48 train_batch=144 train_workers=4 precision=fp16 compile=False`
- `best profile overrides`: `sims=128 opt=adamw lr=0.00015 wd=0.0001 accum=4 replay=250000 train_samples=80000`
- `arena score`: `0.500`
- `loss delta`: `0.0849`
- `summary`: Promoted from phase 4 autotune run 20260329_211802.

```bash
python3 scripts/run_visualizers.py --device mps --profile mps --board-backend native --actor-mode inprocess --actor-workers 6 --selfplay-leaf-batch-size 48 --train-batch-size 144 --train-num-workers 4 --train-precision fp16 --no-train-pin-memory --no-train-compile --selfplay-simulations 128 --train-optimizer adamw --train-lr 0.00015 --train-weight-decay 0.0001 --train-grad-accum-steps 4 --replay-window-samples 250000 --train-samples-per-cycle 80000
```

