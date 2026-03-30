from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.autotune.core.cli import parse_args, print_autotune_footer
from teenyzero.autotune.core.pipeline import board_backend_name, run_auto, run_phase1, run_phase2, run_phase3, run_phase4
from teenyzero.paths import ensure_runtime_dirs


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


def main() -> None:
    args = parse_args(PROFILE)
    ensure_runtime_dirs()
    board_backend = board_backend_name()
    overall_deadline = time.time() + max(0.0, float(args.time_budget_minutes)) * 60.0

    if args.phase == "phase1":
        run_phase1(
            args,
            board_backend,
            overall_deadline,
            runtime=RUNTIME,
            profile=PROFILE,
            project_root=PROJECT_ROOT,
            python_executable=sys.executable,
        )
    elif args.phase == "phase2":
        run_phase2(
            args,
            board_backend,
            overall_deadline,
            runtime=RUNTIME,
            profile=PROFILE,
            project_root=PROJECT_ROOT,
            python_executable=sys.executable,
        )
    elif args.phase == "phase3":
        run_phase3(
            args,
            board_backend,
            overall_deadline,
            runtime=RUNTIME,
            profile=PROFILE,
        )
    elif args.phase == "phase4":
        run_phase4(
            args,
            board_backend,
            overall_deadline,
            runtime=RUNTIME,
            profile=PROFILE,
            project_root=PROJECT_ROOT,
            python_executable=sys.executable,
        )
    else:
        run_auto(
            args,
            board_backend,
            overall_deadline,
            runtime=RUNTIME,
            profile=PROFILE,
            project_root=PROJECT_ROOT,
            python_executable=sys.executable,
        )

    print_autotune_footer()


if __name__ == "__main__":
    main()
