from __future__ import annotations

import json
import os
import time
from pathlib import Path

from teenyzero.autotune.core.common import build_apply_command, json_safe
from teenyzero.paths import (
    AUTOTUNE_LATEST_PATH,
    AUTOTUNE_PHASE1_LATEST_PATH,
    AUTOTUNE_PHASE2_LATEST_PATH,
    AUTOTUNE_PHASE3_LATEST_PATH,
    AUTOTUNE_PHASE4_LATEST_PATH,
    AUTOTUNE_RUNS_DIR,
    ensure_runtime_dirs,
)


PHASE_LATEST_PATHS = {
    "phase1": AUTOTUNE_PHASE1_LATEST_PATH,
    "phase2": AUTOTUNE_PHASE2_LATEST_PATH,
    "phase3": AUTOTUNE_PHASE3_LATEST_PATH,
    "phase4": AUTOTUNE_PHASE4_LATEST_PATH,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def normalized_autotune_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    runtime_args = normalized.get("runtime_args") or {}
    best = normalized.get("best_trial") or {}
    config = best.get("config") or {}
    if runtime_args and config:
        try:
            normalized["apply_command"] = build_apply_command(runtime_args, config, best.get("profile_overrides"))
        except KeyError:
            pass
    return normalized


def save_autotune_run(run_payload: dict, archive: bool = False) -> Path:
    ensure_runtime_dirs()
    payload = normalized_autotune_payload(dict(run_payload))
    phase = str(payload.get("phase") or "phase1").strip().lower()
    latest_path = PHASE_LATEST_PATHS.get(phase, AUTOTUNE_LATEST_PATH)
    _write_json(AUTOTUNE_LATEST_PATH, payload)
    _write_json(latest_path, payload)
    if not archive:
        return latest_path
    run_id = payload.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    archive_path = AUTOTUNE_RUNS_DIR / f"{phase}_{run_id}.json"
    _write_json(archive_path, payload)
    return archive_path


def latest_autotune_run(phase: str | None = None) -> dict | None:
    if phase:
        path = PHASE_LATEST_PATHS.get(str(phase).strip().lower(), AUTOTUNE_LATEST_PATH)
    else:
        path = AUTOTUNE_LATEST_PATH
        if not path.exists():
            for fallback in (AUTOTUNE_PHASE4_LATEST_PATH, AUTOTUNE_PHASE3_LATEST_PATH, AUTOTUNE_PHASE2_LATEST_PATH, AUTOTUNE_PHASE1_LATEST_PATH):
                if fallback.exists():
                    path = fallback
                    break
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return normalized_autotune_payload(json.load(handle))
    except Exception:
        return None


def list_autotune_runs(limit: int = 8, phase: str | None = None) -> list[dict]:
    ensure_runtime_dirs()
    if phase:
        pattern = f"{str(phase).strip().lower()}_*.json"
    else:
        pattern = "phase*_*.json"
    payloads = []
    for path in sorted(AUTOTUNE_RUNS_DIR.glob(pattern), reverse=True)[: max(1, int(limit))]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = normalized_autotune_payload(json.load(handle))
        except Exception:
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads
