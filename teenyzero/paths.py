from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "var"
RUNTIME_ROOT = Path(os.environ.get("TEENYZERO_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT)).expanduser()
DATA_DIR = RUNTIME_ROOT / "data"
MODELS_DIR = RUNTIME_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LATEST_MODEL_PATH = MODELS_DIR / "latest_model.pth"
REPLAY_BUFFER_PATH = DATA_DIR / "replay_buffer"
AUTOTUNE_DIR = DATA_DIR / "autotune"
AUTOTUNE_RUNS_DIR = AUTOTUNE_DIR / "runs"
AUTOTUNE_WORK_DIR = AUTOTUNE_DIR / "work"
AUTOTUNE_LATEST_PATH = AUTOTUNE_DIR / "latest.json"
AUTOTUNE_PHASE1_LATEST_PATH = AUTOTUNE_DIR / "phase1_latest.json"
AUTOTUNE_PHASE2_LATEST_PATH = AUTOTUNE_DIR / "phase2_latest.json"
AUTOTUNE_PHASE3_LATEST_PATH = AUTOTUNE_DIR / "phase3_latest.json"
AUTOTUNE_PHASE4_LATEST_PATH = AUTOTUNE_DIR / "phase4_latest.json"
TRAINING_STATE_PATH = DATA_DIR / "training_state.json"
TRAINING_HISTORY_PATH = DATA_DIR / "training_history.json"
TRAINER_LOCK_PATH = DATA_DIR / "trainer.lock"
ARENA_STATE_PATH = DATA_DIR / "arena_state.json"
ARENA_HISTORY_PATH = DATA_DIR / "arena_history.json"
ARENA_MATCHES_PATH = DATA_DIR / "arena_matches.json"
ARENA_LOCK_PATH = DATA_DIR / "arena.lock"
MODEL_ARCHIVE_PATH = RUNTIME_ROOT / "models" / "archive"
LEGACY_BEST_MODEL_PATH = LEGACY_MODELS_DIR / "best_model.pth"
LEGACY_LATEST_MODEL_PATH = LEGACY_MODELS_DIR / "latest_model.pth"
DEFAULT_MIN_FREE_DISK_MB = 1024


def _runtime_stat_path() -> Path:
    candidate = RUNTIME_ROOT
    while not candidate.exists():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def runtime_free_bytes() -> int:
    try:
        usage = shutil.disk_usage(_runtime_stat_path())
        return int(usage.free)
    except Exception:
        return 0


def runtime_tree_bytes() -> int:
    total = 0
    if not RUNTIME_ROOT.exists():
        return total
    try:
        for entry in RUNTIME_ROOT.rglob("*"):
            if not entry.is_file():
                continue
            try:
                total += int(entry.stat().st_size)
            except OSError:
                continue
    except OSError:
        return total
    return total


def runtime_low_disk_watermark_bytes() -> int:
    raw_value = os.environ.get("TEENYZERO_MIN_FREE_DISK_MB", "").strip()
    if raw_value.isdigit():
        return max(0, int(raw_value)) * 1024 * 1024
    return DEFAULT_MIN_FREE_DISK_MB * 1024 * 1024


def _migrate_legacy_checkpoint(legacy_path: Path, target_path: Path):
    if target_path.exists() or not legacy_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.replace(target_path)


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    AUTOTUNE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    AUTOTUNE_WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_checkpoint(LEGACY_BEST_MODEL_PATH, BEST_MODEL_PATH)
    _migrate_legacy_checkpoint(LEGACY_LATEST_MODEL_PATH, LATEST_MODEL_PATH)


def runtime_paths_payload():
    return {
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "replay_buffer_path": str(REPLAY_BUFFER_PATH),
        "autotune_dir": str(AUTOTUNE_DIR),
        "autotune_runs_dir": str(AUTOTUNE_RUNS_DIR),
        "autotune_work_dir": str(AUTOTUNE_WORK_DIR),
        "autotune_latest_path": str(AUTOTUNE_LATEST_PATH),
        "autotune_phase1_latest_path": str(AUTOTUNE_PHASE1_LATEST_PATH),
        "autotune_phase2_latest_path": str(AUTOTUNE_PHASE2_LATEST_PATH),
        "autotune_phase3_latest_path": str(AUTOTUNE_PHASE3_LATEST_PATH),
        "autotune_phase4_latest_path": str(AUTOTUNE_PHASE4_LATEST_PATH),
        "training_state_path": str(TRAINING_STATE_PATH),
        "training_history_path": str(TRAINING_HISTORY_PATH),
        "arena_state_path": str(ARENA_STATE_PATH),
        "arena_history_path": str(ARENA_HISTORY_PATH),
        "arena_matches_path": str(ARENA_MATCHES_PATH),
        "model_archive_path": str(MODEL_ARCHIVE_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "latest_model_path": str(LATEST_MODEL_PATH),
        "runtime_tree_bytes": runtime_tree_bytes(),
        "runtime_free_bytes": runtime_free_bytes(),
        "min_free_disk_bytes": runtime_low_disk_watermark_bytes(),
    }
