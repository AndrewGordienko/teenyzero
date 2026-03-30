from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teenyzero.autotune.catalog.recommendations import (
    AUTOTUNE_RESULTS_DOC_PATH,
    RECOMMENDATIONS_PATH,
    promote_autotune_run,
)
from teenyzero.autotune.core.storage import latest_autotune_run


def _load_run(path: str | None) -> dict:
    if path:
        run_path = Path(path).expanduser()
        with open(run_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    payload = latest_autotune_run()
    if not payload:
        raise ValueError("No latest autotune run found. Pass --run to select a saved run file.")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Promote an autotune run into the shared recommendations registry.")
    parser.add_argument("--run", default=None, help="Path to a saved autotune run JSON. Defaults to the latest local run.")
    parser.add_argument("--name", default=None, help="Optional user-facing title for the recommendation.")
    parser.add_argument("--workload", default=None, help="Optional workload label. Defaults to the run objective.")
    parser.add_argument("--notes", default=None, help="Optional summary or notes to include with the recommendation.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_payload = _load_run(args.run)
    entry = promote_autotune_run(
        run_payload,
        name=args.name,
        workload=args.workload,
        notes=args.notes,
    )

    print("[*] Promoted autotune recommendation")
    print(f"[*] ID: {entry['id']}")
    print(f"[*] Title: {entry['title']}")
    print(f"[*] Registry: {RECOMMENDATIONS_PATH}")
    print(f"[*] Docs: {AUTOTUNE_RESULTS_DOC_PATH}")
    if entry.get("apply_command"):
        print("[*] Apply Command:")
        print(entry["apply_command"])


if __name__ == "__main__":
    main()
