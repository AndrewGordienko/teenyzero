# autotune

This package owns the autotune flow for TeenyZero.

The layout is grouped by responsibility now instead of keeping every file flat at the top level:

- `core/`
  CLI wiring, orchestration, storage, payload shaping, benchmark launchers, and small shared helpers.
- `phases/`
  The actual search stages: phase 1, phase 2, phase 3, phase 4, plus the replay and arena helpers used by the quality phases.
- `catalog/`
  The shared recommendation registry and promotion code used to turn local autotune runs into open-source defaults.

The most important files are:

- `core/pipeline.py`
  Runs phase 1, phase 2, phase 3, phase 4, auto-resume, and auto-promotion.
- `phases/phase1.py`
  Broad candidate generation and first-pass scoring.
- `phases/phase2.py`
  Successive-halving refinement around the best region from phase 1.
- `phases/phase3.py`
  Short real-training validation and final scoring for the best phase 2 finalists.
- `phases/phase4.py`
  Searches profile-level learning and search knobs on top of the best runtime finalists while keeping the network shape fixed.
- `catalog/recommendations.py`
  Promotion of local results into the shared recommendation catalog and markdown report.

The normal entrypoint is:

```bash
python3 scripts/autotune.py --device mps --profile mps --board-backend native
```

That single command is the intended UX:

- it runs the full phase 1 -> phase 4 pipeline
- it reuses compatible saved phases instead of re-running identical work
- it promotes the final result into the shared catalog and markdown file unless you pass `--no-auto-promote`

The script stays small on purpose. It bootstraps the runtime, then hands off to `pipeline.py`.

If you are trying to change behavior, a good rule of thumb is:

- edit `phases/phase1.py` or `phases/phase2.py` for runtime throughput search behavior
- edit `phases/phase3.py`, `phases/phase3_data.py`, or `phases/phase3_arena.py` for short quality validation
- edit `phases/phase4.py` for profile-level learning/search sweeps
- edit `catalog/recommendations.py` for the shared catalog and markdown results page
- edit `core/cli.py` only for UX and flags
