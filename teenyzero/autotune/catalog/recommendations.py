from __future__ import annotations

import json
import re
import time
from pathlib import Path

from teenyzero.autotune.core.common import build_apply_command
from teenyzero.autotune.core.storage import latest_autotune_run
from teenyzero.paths import PROJECT_ROOT


RECOMMENDATIONS_PATH = Path(__file__).resolve().with_name("recommendations.json")
AUTOTUNE_RESULTS_DOC_PATH = PROJECT_ROOT / "docs" / "autotune_results.md"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_recommendations() -> dict:
    payload = _read_json(RECOMMENDATIONS_PATH, {"version": 1, "updated_at": None, "recommendations": []})
    payload.setdefault("version", 1)
    payload.setdefault("updated_at", None)
    payload.setdefault("recommendations", [])
    return payload


def save_recommendations(payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.time()
    payload.setdefault("version", 1)
    payload.setdefault("recommendations", [])
    _write_json(RECOMMENDATIONS_PATH, payload)


def recommendation_device_family(run_payload: dict) -> str:
    hardware = run_payload.get("hardware", {})
    device = str(hardware.get("device", "unknown"))
    system = str(hardware.get("platform", {}).get("system", "")).lower()
    machine = str(hardware.get("platform", {}).get("machine", "")).lower()
    cuda_name = str((hardware.get("cuda_device") or {}).get("name", "")).strip()

    if device == "mps" and system == "darwin" and machine == "arm64":
        return "apple_silicon_mps"
    if cuda_name:
        return _slugify(cuda_name)
    return _slugify(device)


def recommendation_title(run_payload: dict, workload: str, name: str | None = None) -> str:
    if name:
        return name.strip()

    hardware = run_payload.get("hardware", {})
    family = recommendation_device_family(run_payload)
    memory_bytes = hardware.get("physical_memory_bytes")
    memory_gb = None
    if isinstance(memory_bytes, int) and memory_bytes > 0:
        memory_gb = int(round(memory_bytes / (1024 ** 3)))

    if family == "apple_silicon_mps":
        suffix = f" {memory_gb}GB" if memory_gb else ""
        return f"Apple Silicon MPS{suffix} {workload.title()}"
    if (hardware.get("cuda_device") or {}).get("name"):
        return f"{hardware['cuda_device']['name']} {workload.title()}"
    return f"{family.replace('_', ' ').title()} {workload.title()}"


def _phase_summary_label(value: str | None) -> str:
    phase = str(value or "autotune").strip().lower()
    if phase == "phase1":
        return "phase 1"
    if phase == "phase2":
        return "phase 2"
    return phase


def build_recommendation_entry(
    run_payload: dict,
    name: str | None = None,
    workload: str | None = None,
    notes: str | None = None,
) -> dict:
    best_trial = run_payload.get("best_trial") or {}
    if not best_trial:
        raise ValueError("Run payload does not contain a best_trial.")

    workload_name = (workload or run_payload.get("objective") or "balanced").strip().lower()
    title = recommendation_title(run_payload, workload_name, name=name)
    family = recommendation_device_family(run_payload)
    entry_id = _slugify(f"{family}_{workload_name}")
    hardware = run_payload.get("hardware", {})
    runtime_args = run_payload.get("runtime_args", {})
    best_config = best_trial.get("config", {})
    apply_command = run_payload.get("apply_command")
    if runtime_args and best_config:
        try:
            apply_command = build_apply_command(runtime_args, best_config)
        except KeyError:
            pass
    phase_label = _phase_summary_label(run_payload.get("phase"))

    return {
        "id": entry_id,
        "title": title,
        "workload": workload_name,
        "device_family": family,
        "summary": notes or f"Promoted from {phase_label} autotune run {run_payload.get('run_id', 'unknown')}.",
        "hardware_match": {
            "device": hardware.get("device"),
            "platform_system": (hardware.get("platform") or {}).get("system"),
            "platform_machine": (hardware.get("platform") or {}).get("machine"),
            "cuda_device_name": (hardware.get("cuda_device") or {}).get("name"),
        },
        "runtime_seed": {
            "device": runtime_args.get("device"),
            "profile": runtime_args.get("profile"),
            "board_backend": runtime_args.get("board_backend"),
        },
        "config": best_config,
        "apply_command": apply_command,
        "metrics": {
            "score": best_trial.get("score"),
            "selfplay_positions_per_s": (best_trial.get("selfplay") or {}).get("positions_per_s"),
            "selfplay_searches_per_s": (best_trial.get("selfplay") or {}).get("searches_per_s"),
            "selfplay_move_total_mean_ms": (best_trial.get("selfplay") or {}).get("move_total_mean_ms"),
            "train_samples_per_s": (best_trial.get("train") or {}).get("samples_per_s"),
            "train_avg_batch_time_ms": (best_trial.get("train") or {}).get("avg_batch_time_ms"),
        },
        "source": {
            "phase": run_payload.get("phase"),
            "objective": run_payload.get("objective"),
            "run_id": run_payload.get("run_id"),
            "run_started_at": run_payload.get("started_at"),
            "run_finished_at": run_payload.get("finished_at"),
            "best_trial_label": best_trial.get("label"),
        },
        "updated_at": time.time(),
    }


def upsert_recommendation(entry: dict, payload: dict | None = None) -> dict:
    payload = load_recommendations() if payload is None else dict(payload)
    recommendations = [item for item in payload.get("recommendations", []) if item.get("id") != entry.get("id")]
    recommendations.append(entry)
    recommendations.sort(key=lambda item: str(item.get("id", "")))
    payload["recommendations"] = recommendations
    return payload


def recommendations_markdown(payload: dict) -> str:
    recommendations = list(payload.get("recommendations", []))
    lines = [
        "# Autotune Results",
        "",
        "This file tracks promoted autotune runtime recommendations that can be shared in the repo.",
        "Each entry comes from an autotune run and captures a recommended hardware/runtime setup for a workload.",
        "",
        "The source of truth is the checked-in registry at `teenyzero/autotune/catalog/recommendations.json`.",
        "",
    ]

    if not recommendations:
        lines.extend(
            [
                "No promoted recommendations yet.",
                "",
                "Generate a local run with:",
                "",
                "```bash",
                "python3 scripts/autotune.py --device mps --profile mps --board-backend native",
                "```",
                "",
                "Then promote it with:",
                "",
                "```bash",
                "python3 scripts/promote_autotune.py",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| ID | Title | Workload | Seed | Best Trial | Self-Play Pos/Sec | Train Samples/Sec |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for entry in recommendations:
        seed = entry.get("runtime_seed", {})
        source = entry.get("source", {})
        metrics = entry.get("metrics", {})
        lines.append(
            "| {id} | {title} | {workload} | {seed} | {trial} | {pos:.1f} | {train:.1f} |".format(
                id=entry.get("id", "n/a"),
                title=entry.get("title", "n/a"),
                workload=entry.get("workload", "n/a"),
                seed="/".join(
                    str(seed.get(key, "n/a")) for key in ("device", "profile", "board_backend")
                ),
                trial=source.get("best_trial_label", "n/a"),
                pos=float(metrics.get("selfplay_positions_per_s") or 0.0),
                train=float(metrics.get("train_samples_per_s") or 0.0),
            )
        )

    lines.append("")
    for entry in recommendations:
        metrics = entry.get("metrics", {})
        config = entry.get("config", {})
        lines.extend(
            [
                f"## {entry.get('title', entry.get('id', 'Recommendation'))}",
                "",
                f"- `id`: `{entry.get('id', 'n/a')}`",
                f"- `workload`: `{entry.get('workload', 'n/a')}`",
                f"- `device family`: `{entry.get('device_family', 'n/a')}`",
                f"- `score`: `{float(metrics.get('score', metrics.get('phase1_score')) or 0.0):.3f}`",
                f"- `self-play positions/sec`: `{float(metrics.get('selfplay_positions_per_s') or 0.0):.1f}`",
                f"- `train samples/sec`: `{float(metrics.get('train_samples_per_s') or 0.0):.1f}`",
                f"- `best config`: `mode={config.get('actor_mode')}`, `workers={config.get('selfplay_workers')}`, `leaf={config.get('selfplay_leaf_batch_size')}`, `train_batch={config.get('train_batch_size')}`, `train_workers={config.get('train_num_workers')}`, `precision={config.get('train_precision')}`, `compile={config.get('train_compile')}`",
                f"- `summary`: {entry.get('summary', 'n/a')}",
                "",
                "```bash",
                entry.get("apply_command") or "# no apply command recorded",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_recommendations_markdown(payload: dict) -> None:
    AUTOTUNE_RESULTS_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTOTUNE_RESULTS_DOC_PATH.write_text(recommendations_markdown(payload), encoding="utf-8")


def promote_latest_autotune_run(name: str | None = None, workload: str | None = None, notes: str | None = None) -> dict:
    run_payload = latest_autotune_run()
    if not run_payload:
        raise ValueError("No autotune run is available to promote.")
    entry = build_recommendation_entry(run_payload, name=name, workload=workload, notes=notes)
    payload = upsert_recommendation(entry)
    save_recommendations(payload)
    write_recommendations_markdown(payload)
    return entry


def promote_latest_phase1_run(name: str | None = None, workload: str | None = None, notes: str | None = None) -> dict:
    return promote_latest_autotune_run(name=name, workload=workload, notes=notes)
