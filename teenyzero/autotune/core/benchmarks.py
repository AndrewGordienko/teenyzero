from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from teenyzero.paths import AUTOTUNE_DIR, ensure_runtime_dirs


def subprocess_env(project_root: Path) -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "").strip()
    root = str(project_root)
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def temp_json_path(prefix: str) -> str:
    ensure_runtime_dirs()
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=str(AUTOTUNE_DIR))
    os.close(fd)
    return path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cleanup_temp(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def run_command(command: list[str], timeout_s: float, project_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(project_root),
        env=subprocess_env(project_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def run_selfplay_trial(
    trial_config: dict,
    settings: dict,
    *,
    project_root: Path,
    python_executable: str,
) -> dict:
    output_path = temp_json_path("selfplay_")
    command = [
        python_executable,
        str(project_root / "scripts" / "benchmark_self_play.py"),
        "--workers",
        str(trial_config["selfplay_workers"]),
        "--searches-per-worker",
        str(settings["searches_per_worker"]),
        "--simulations",
        str(settings["selfplay_simulations"]),
        "--leaf-batch-size",
        str(trial_config["selfplay_leaf_batch_size"]),
        "--actor-mode",
        str(trial_config["actor_mode"]),
        "--json-output",
        output_path,
    ]
    try:
        completed = run_command(command, timeout_s=float(settings["trial_timeout_s"]), project_root=project_root)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or completed.stdout.strip() or "self-play benchmark failed",
                "command": command,
            }
        payload = load_json(output_path)
        summary = payload.get("summary", {})
        return {
            "ok": True,
            "command": command,
            "raw": payload,
            "searches_per_s": float(payload.get("searches_per_s", 0.0)),
            "simulations_per_s": float(payload.get("simulations_per_s", 0.0)),
            "positions_per_s": float(payload.get("positions_per_s", 0.0)),
            "move_total_mean_ms": float(summary.get("move_total_ms", {}).get("mean", 0.0)),
            "selection_mean_ms": float(summary.get("selection_ms", {}).get("mean", 0.0)),
            "leaf_eval_mean_ms": float(summary.get("leaf_eval_ms", {}).get("mean", 0.0)),
            "duration_s": float(payload.get("duration_s", 0.0)),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"self-play benchmark timed out after {float(settings['trial_timeout_s']):.0f}s", "command": command}
    finally:
        cleanup_temp(output_path)


def run_train_trial(
    trial_config: dict,
    settings: dict,
    *,
    project_root: Path,
    python_executable: str,
) -> dict:
    output_path = temp_json_path("train_")
    command = [
        python_executable,
        str(project_root / "scripts" / "benchmark_train.py"),
        "--batch-size",
        str(trial_config["train_batch_size"]),
        "--batches",
        str(settings["train_batches"]),
        "--num-workers",
        str(trial_config["train_num_workers"]),
        "--precision",
        str(trial_config["train_precision"]),
        "--json-output",
        output_path,
    ]
    if trial_config["train_compile"]:
        command.append("--compile")
    if trial_config["train_pin_memory"]:
        command.append("--pin-memory")
    try:
        completed = run_command(command, timeout_s=float(settings["trial_timeout_s"]), project_root=project_root)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or completed.stdout.strip() or "train benchmark failed",
                "command": command,
            }
        payload = load_json(output_path)
        metrics = payload.get("metrics", {})
        return {
            "ok": True,
            "command": command,
            "raw": payload,
            "samples_per_s": float(metrics.get("samples_per_s", 0.0)),
            "batches_per_s": float(metrics.get("batches_per_s", 0.0)),
            "avg_batch_time_ms": float(metrics.get("avg_batch_time_ms", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
            "duration_s": float(payload.get("duration_s", 0.0)),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"train benchmark timed out after {float(settings['trial_timeout_s']):.0f}s", "command": command}
    finally:
        cleanup_temp(output_path)


def build_trial_record(
    candidate: dict,
    settings: dict,
    *,
    project_root: Path,
    python_executable: str,
) -> dict:
    started = time.time()
    config = dict(candidate["config"])
    selfplay_result = run_selfplay_trial(
        config,
        settings,
        project_root=project_root,
        python_executable=python_executable,
    )
    train_result = run_train_trial(
        config,
        settings,
        project_root=project_root,
        python_executable=python_executable,
    )
    errors = []
    if not selfplay_result.get("ok"):
        errors.append(selfplay_result.get("error", "self-play benchmark failed"))
    if not train_result.get("ok"):
        errors.append(train_result.get("error", "train benchmark failed"))

    return {
        "index": int(candidate["index"]),
        "label": candidate["label"],
        "candidate_id": candidate.get("candidate_id"),
        "round_label": settings.get("round_label"),
        "is_baseline": bool(candidate.get("is_baseline")),
        "is_seed": bool(candidate.get("is_seed")),
        "config": config,
        "started_at": started,
        "finished_at": time.time(),
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "selfplay": selfplay_result if selfplay_result.get("ok") else {},
        "train": train_result if train_result.get("ok") else {},
    }
