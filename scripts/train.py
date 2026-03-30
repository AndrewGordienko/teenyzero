import json
import os
import time
import atexit
import math
import errno
import subprocess
from pathlib import Path

import torch

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.logic.trainer import (
    AlphaTrainer,
    dataloader_for_replay_window,
    prune_replay_buffer,
    replay_buffer_summary,
)
from teenyzero.alphazero.runtime import get_runtime_selection, runtime_profile_payload
from teenyzero.paths import (
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    MODEL_ARCHIVE_PATH,
    REPLAY_BUFFER_PATH,
    TRAINER_LOCK_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_STATE_PATH,
    ensure_runtime_dirs,
    runtime_free_bytes,
    runtime_low_disk_watermark_bytes,
    runtime_paths_payload,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = str(REPLAY_BUFFER_PATH)
MODEL_DIR = str(BEST_MODEL_PATH.parent)
MODEL_PATH = str(BEST_MODEL_PATH)
LATEST_MODEL_PATH = str(LATEST_MODEL_PATH)
TRAINING_STATE_PATH = str(TRAINING_STATE_PATH)
TRAINER_LOCK_PATH = str(TRAINER_LOCK_PATH)
TRAINING_HISTORY_PATH = str(TRAINING_HISTORY_PATH)

RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile
PROFILE_SETTINGS = runtime_profile_payload(PROFILE)

def _env_int(name, default, minimum=1):
    raw_value = os.environ.get(name, "").strip()
    if raw_value.isdigit():
        return max(minimum, int(raw_value))
    return int(default)


def _env_bool(name, default):
    raw_value = os.environ.get(name, "").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_float(name, default, minimum=0.0):
    raw_value = os.environ.get(name, "").strip()
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return float(default)


def _env_choice(name, default, allowed):
    raw_value = os.environ.get(name, "").strip().lower()
    if raw_value in allowed:
        return raw_value
    return str(default).lower()


MIN_SAMPLES_READY = PROFILE.min_samples_ready
TRAIN_INCREMENT = PROFILE.train_increment
REPLAY_WINDOW_SAMPLES = _env_int("TEENYZERO_REPLAY_WINDOW_SAMPLES", PROFILE.replay_window_samples)
TRAIN_SAMPLES_PER_CYCLE = _env_int("TEENYZERO_TRAIN_SAMPLES_PER_CYCLE", PROFILE.train_samples_per_cycle)
BOOTSTRAP_WINDOW_SAMPLES = PROFILE.bootstrap_window_samples
MAX_RETAINED_SAMPLES = PROFILE.max_retained_samples
BATCH_SIZE = _env_int("TEENYZERO_TRAIN_BATCH_SIZE", PROFILE.train_batch_size)
TRAIN_NUM_WORKERS = _env_int("TEENYZERO_TRAIN_NUM_WORKERS", PROFILE.train_num_workers, minimum=0)
TRAIN_PIN_MEMORY = _env_bool("TEENYZERO_TRAIN_PIN_MEMORY", PROFILE.train_pin_memory)
TRAIN_PRECISION = _env_choice("TEENYZERO_TRAIN_PRECISION", PROFILE.train_precision, {"fp32", "fp16", "bf16"})
TRAIN_COMPILE = _env_bool("TEENYZERO_TRAIN_COMPILE", PROFILE.train_compile)
TRAIN_OPTIMIZER = _env_choice("TEENYZERO_TRAIN_OPTIMIZER", PROFILE.train_optimizer, {"sgd", "adam", "adamw"})
TRAIN_LR = _env_float("TEENYZERO_TRAIN_LR", PROFILE.train_lr, minimum=1e-8)
TRAIN_WEIGHT_DECAY = _env_float("TEENYZERO_TRAIN_WEIGHT_DECAY", PROFILE.train_weight_decay, minimum=0.0)
TRAIN_GRAD_ACCUM_STEPS = _env_int("TEENYZERO_TRAIN_GRAD_ACCUM_STEPS", PROFILE.train_grad_accum_steps, minimum=1)
EPOCHS_PER_CYCLE = PROFILE.train_epochs_per_cycle
POLL_INTERVAL_S = PROFILE.train_poll_interval_s


def _state_defaults():
    return {
        "status": "waiting",
        "stage_started_at": None,
        "last_trained_sample_count": 0,
        "last_train_cutoff_mtime": 0.0,
        "training_cycles": 0,
        "last_loss": 0.0,
        "last_policy_loss": 0.0,
        "last_value_loss": 0.0,
        "last_cycle_samples": 0,
        "last_window_samples": 0,
        "last_window_files": 0,
        "last_train_started_at": None,
        "last_train_finished_at": None,
        "last_train_duration_s": 0.0,
        "last_scan_duration_s": 0.0,
        "last_window_build_duration_s": 0.0,
        "last_train_phase_duration_s": 0.0,
        "last_checkpoint_duration_s": 0.0,
        "last_prune_duration_s": 0.0,
        "last_avg_batch_time_ms": 0.0,
        "last_batches_per_s": 0.0,
        "last_samples_per_s": 0.0,
        "buffer_sample_count": 0,
        "buffer_file_count": 0,
        "new_samples_since_last_train": 0,
        "min_samples_ready": MIN_SAMPLES_READY,
        "train_increment": TRAIN_INCREMENT,
        "replay_window_samples": REPLAY_WINDOW_SAMPLES,
        "bootstrap_window_samples": BOOTSTRAP_WINDOW_SAMPLES,
        "train_samples_per_cycle": TRAIN_SAMPLES_PER_CYCLE,
        "max_retained_samples": MAX_RETAINED_SAMPLES,
        "active_window_samples": 0,
        "loaded_files": 0,
        "total_window_files": 0,
        "loaded_window_samples": 0,
        "window_load_elapsed_s": 0.0,
        "window_files_per_s": 0.0,
        "window_samples_per_s": 0.0,
        "completed_batches": 0,
        "total_batches": 0,
        "train_elapsed_s": 0.0,
        "avg_batch_time_ms": 0.0,
        "batches_per_s": 0.0,
        "samples_per_s": 0.0,
        "trained_samples": 0,
        "running_loss": 0.0,
        "running_policy_loss": 0.0,
        "running_value_loss": 0.0,
        "heartbeat_at": None,
        "latest_model_path": None,
        "last_pruned_files": 0,
        "device": None,
        "runtime_profile": PROFILE.name,
        "runtime_profile_settings": PROFILE_SETTINGS,
        "train_batch_size": BATCH_SIZE,
        "train_num_workers": TRAIN_NUM_WORKERS,
        "train_pin_memory": TRAIN_PIN_MEMORY,
        "train_precision": TRAIN_PRECISION,
        "train_compile": TRAIN_COMPILE,
        "train_optimizer": TRAIN_OPTIMIZER,
        "train_lr": TRAIN_LR,
        "train_weight_decay": TRAIN_WEIGHT_DECAY,
        "train_grad_accum_steps": TRAIN_GRAD_ACCUM_STEPS,
        "runtime_paths": runtime_paths_payload(),
        "training_history_path": TRAINING_HISTORY_PATH,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _load_training_state():
    defaults = _state_defaults()
    if not os.path.exists(TRAINING_STATE_PATH):
        return defaults

    with open(TRAINING_STATE_PATH, "r", encoding="utf-8") as handle:
        loaded = _json_safe(json.load(handle))
    defaults.update(loaded)
    return defaults


def _write_training_state(state):
    os.makedirs(os.path.dirname(TRAINING_STATE_PATH), exist_ok=True)
    tmp_path = TRAINING_STATE_PATH + ".tmp"
    payload = _json_safe(state)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp_path, TRAINING_STATE_PATH)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        _reclaim_runtime_space()
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True, allow_nan=False)
            os.replace(tmp_path, TRAINING_STATE_PATH)
        except OSError as retry_exc:
            if retry_exc.errno != errno.ENOSPC:
                raise
            # Last-resort fallback: overwrite the target in place with compact JSON.
            with open(TRAINING_STATE_PATH, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _mark_stage(state, stage, **extra):
    now = time.time()
    if state.get("status") != stage:
        state["stage_started_at"] = now
    state["status"] = stage
    state["heartbeat_at"] = now
    for key, value in extra.items():
        state[key] = value
    _write_training_state(state)


def _append_training_history(entry):
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    history = []
    if os.path.exists(TRAINING_HISTORY_PATH):
        try:
            with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as handle:
                history = _json_safe(json.load(handle))
        except Exception:
            history = []

    history.append(_json_safe(entry))
    history = history[-200:]

    tmp_path = TRAINING_HISTORY_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(history), handle, indent=2, allow_nan=False)
        os.replace(tmp_path, TRAINING_HISTORY_PATH)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        _reclaim_runtime_space()

def _runtime_under_disk_pressure():
    return runtime_free_bytes() <= runtime_low_disk_watermark_bytes()


def _reclaim_runtime_space():
    try:
        prune_replay_buffer(DATA_DIR, max_samples_to_keep=max(1, MAX_RETAINED_SAMPLES // 2))
    except Exception:
        pass

    try:
        archive_paths = sorted(MODEL_ARCHIVE_PATH.glob("*.pth"))
        while len(archive_paths) > 1:
            oldest = archive_paths.pop(0)
            oldest.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        history_path = Path(TRAINING_HISTORY_PATH)
        if history_path.exists():
            history_path.unlink(missing_ok=True)
    except Exception:
        pass

    for suffix in (".tmp",):
        for path in Path(os.path.dirname(TRAINING_STATE_PATH)).glob(f"*{suffix}"):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
        if result.stdout.strip().upper().startswith("Z"):
            return False
    except Exception:
        pass
    return True


def _acquire_trainer_lock():
    os.makedirs(os.path.dirname(TRAINER_LOCK_PATH), exist_ok=True)
    if os.path.exists(TRAINER_LOCK_PATH):
        try:
            with open(TRAINER_LOCK_PATH, "r", encoding="utf-8") as handle:
                existing_pid = int(handle.read().strip())
            if _pid_is_running(existing_pid):
                return False
        except Exception:
            pass
        try:
            os.remove(TRAINER_LOCK_PATH)
        except OSError:
            return False

    with open(TRAINER_LOCK_PATH, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


def _release_trainer_lock():
    if os.path.exists(TRAINER_LOCK_PATH):
        try:
            os.remove(TRAINER_LOCK_PATH)
        except OSError:
            pass


def main():
    if not _acquire_trainer_lock():
        print("[Trainer] Another trainer instance is already running. Exiting.")
        return

    atexit.register(_release_trainer_lock)
    device = RUNTIME.device
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_runtime_dirs()

    model = build_model()
    load_result = load_checkpoint(model, LATEST_MODEL_PATH, map_location=device, allow_partial=True)
    if not load_result["loaded"]:
        fallback_result = load_checkpoint(model, MODEL_PATH, map_location=device, allow_partial=True)
        if fallback_result["loaded"]:
            load_result = fallback_result

    if load_result["loaded"]:
        flavor = "partial" if load_result["partial"] else "full"
        print(f"[Trainer] Loaded {flavor} checkpoint ({load_result['reason']}).")
    else:
        print(f"[Trainer] Starting from fresh weights ({load_result['reason']}).")

    trainer = AlphaTrainer(
        model,
        device=device,
        lr=TRAIN_LR,
        optimizer_name=TRAIN_OPTIMIZER,
        weight_decay=TRAIN_WEIGHT_DECAY,
        momentum=PROFILE.train_momentum,
        grad_accum_steps=TRAIN_GRAD_ACCUM_STEPS,
        precision=TRAIN_PRECISION,
        use_compile=TRAIN_COMPILE,
        max_grad_norm=PROFILE.max_grad_norm,
    )
    state = _load_training_state()
    state["device"] = device
    state["runtime_profile"] = PROFILE.name
    state["runtime_profile_settings"] = PROFILE_SETTINGS
    state["train_batch_size"] = BATCH_SIZE
    state["train_num_workers"] = TRAIN_NUM_WORKERS
    state["train_pin_memory"] = TRAIN_PIN_MEMORY
    state["train_precision"] = TRAIN_PRECISION
    state["train_compile"] = TRAIN_COMPILE
    state["train_optimizer"] = TRAIN_OPTIMIZER
    state["train_lr"] = TRAIN_LR
    state["train_weight_decay"] = TRAIN_WEIGHT_DECAY
    state["train_grad_accum_steps"] = TRAIN_GRAD_ACCUM_STEPS
    state["runtime_paths"] = runtime_paths_payload()
    state["training_history_path"] = TRAINING_HISTORY_PATH
    _write_training_state(state)

    print("[Trainer] Continuous trainer online.")

    while True:
        if _runtime_under_disk_pressure():
            _reclaim_runtime_space()

        scan_started = time.perf_counter()
        _mark_stage(
            state,
            "scanning_replay_buffer",
            loaded_files=0,
            total_window_files=0,
            loaded_window_samples=0,
            window_load_elapsed_s=0.0,
            window_files_per_s=0.0,
            window_samples_per_s=0.0,
            train_elapsed_s=0.0,
            avg_batch_time_ms=0.0,
            batches_per_s=0.0,
            samples_per_s=0.0,
            trained_samples=0,
        )
        summary = replay_buffer_summary(DATA_DIR)
        scan_duration_s = time.perf_counter() - scan_started
        files = summary["files"]
        total_samples = int(summary["sample_count"])
        cutoff_mtime = float(state.get("last_train_cutoff_mtime", 0.0) or 0.0)
        if cutoff_mtime <= 0.0:
            new_samples = total_samples
        else:
            new_samples = sum(info.sample_count for info in files if info.mtime > cutoff_mtime)

        state["buffer_sample_count"] = total_samples
        state["buffer_file_count"] = int(summary["file_count"])
        state["new_samples_since_last_train"] = new_samples
        state["last_scan_duration_s"] = float(scan_duration_s)
        if state.get("status") != "waiting":
            state["stage_started_at"] = time.time()
        state["status"] = "waiting"
        state["heartbeat_at"] = time.time()
        _write_training_state(state)

        if total_samples < MIN_SAMPLES_READY or new_samples < TRAIN_INCREMENT:
            time.sleep(POLL_INTERVAL_S)
            continue

        train_started = time.time()
        state["last_train_started_at"] = train_started
        window_started = time.perf_counter()
        _mark_stage(
            state,
            "building_replay_window",
            loaded_files=0,
            total_window_files=0,
            loaded_window_samples=0,
            window_load_elapsed_s=0.0,
            window_files_per_s=0.0,
            window_samples_per_s=0.0,
            completed_batches=0,
            total_batches=0,
            train_elapsed_s=0.0,
            avg_batch_time_ms=0.0,
            batches_per_s=0.0,
            samples_per_s=0.0,
            trained_samples=0,
        )
        current_window = BOOTSTRAP_WINDOW_SAMPLES if int(state.get("training_cycles", 0)) == 0 else REPLAY_WINDOW_SAMPLES
        sample_target = min(TRAIN_SAMPLES_PER_CYCLE, current_window)
        state["active_window_samples"] = current_window
        state["train_samples_per_cycle"] = sample_target
        _write_training_state(state)

        def on_window_progress(progress):
            _mark_stage(
                state,
                progress.get("stage", "loading_replay_window"),
                loaded_files=int(progress.get("loaded_files", 0)),
                total_window_files=int(progress.get("total_files", 0)),
                loaded_window_samples=int(progress.get("loaded_samples", 0)),
                window_load_elapsed_s=float(progress.get("elapsed_s", 0.0)),
                window_files_per_s=float(progress.get("files_per_s", 0.0)),
                window_samples_per_s=float(progress.get("samples_per_s", 0.0)),
            )

        loader, window_samples, files = dataloader_for_replay_window(
            DATA_DIR,
            max_samples=current_window,
            sample_size=sample_target,
            batch_size=BATCH_SIZE,
            shuffle=True,
            progress_callback=on_window_progress,
            rng_seed=int(time.time() * 1000) & 0xFFFFFFFF,
            num_workers=TRAIN_NUM_WORKERS,
            pin_memory=TRAIN_PIN_MEMORY,
            prefetch_factor=PROFILE.train_prefetch_factor,
        )
        window_build_duration_s = time.perf_counter() - window_started
        state["last_window_build_duration_s"] = float(window_build_duration_s)

        last_metrics = None
        train_phase_started = time.perf_counter()
        for _ in range(EPOCHS_PER_CYCLE):
            def on_train_progress(progress):
                _mark_stage(
                    state,
                    progress.get("stage", "training_batches"),
                    completed_batches=int(progress.get("completed_batches", 0)),
                    total_batches=int(progress.get("total_batches", 0)),
                    running_loss=float(progress.get("running_loss", 0.0)),
                    running_policy_loss=float(progress.get("running_policy_loss", 0.0)),
                    running_value_loss=float(progress.get("running_value_loss", 0.0)),
                    train_elapsed_s=float(progress.get("elapsed_s", 0.0)),
                    avg_batch_time_ms=float(progress.get("avg_batch_time_ms", 0.0)),
                    batches_per_s=float(progress.get("batches_per_s", 0.0)),
                    samples_per_s=float(progress.get("samples_per_s", 0.0)),
                    trained_samples=int(progress.get("samples_seen", 0)),
                )

            last_metrics = trainer.train_epoch(loader, progress_callback=on_train_progress)
        train_phase_duration_s = time.perf_counter() - train_phase_started
        state["last_train_phase_duration_s"] = float(train_phase_duration_s)

        checkpoint_started = time.perf_counter()
        _mark_stage(state, "saving_checkpoint")
        trainer.save_checkpoint(LATEST_MODEL_PATH)
        checkpoint_duration_s = time.perf_counter() - checkpoint_started
        state["last_checkpoint_duration_s"] = float(checkpoint_duration_s)

        prune_started = time.perf_counter()
        prune_result = prune_replay_buffer(DATA_DIR, max_samples_to_keep=MAX_RETAINED_SAMPLES)
        prune_duration_s = time.perf_counter() - prune_started
        state["last_prune_duration_s"] = float(prune_duration_s)

        train_finished = time.time()
        if state.get("status") != "waiting":
            state["stage_started_at"] = train_finished
        state["status"] = "waiting"
        state["last_trained_sample_count"] = total_samples
        state["last_train_cutoff_mtime"] = max((info.mtime for info in files), default=cutoff_mtime)
        state["training_cycles"] = int(state.get("training_cycles", 0)) + 1
        state["last_loss"] = float(last_metrics["loss"]) if last_metrics else 0.0
        state["last_policy_loss"] = float(last_metrics["policy_loss"]) if last_metrics else 0.0
        state["last_value_loss"] = float(last_metrics["value_loss"]) if last_metrics else 0.0
        state["last_cycle_samples"] = new_samples
        state["last_window_samples"] = int(window_samples)
        state["last_window_files"] = len(files)
        state["last_train_finished_at"] = train_finished
        state["last_train_duration_s"] = float(train_finished - train_started)
        state["last_avg_batch_time_ms"] = float(last_metrics["avg_batch_time_ms"]) if last_metrics else 0.0
        state["last_batches_per_s"] = float(last_metrics["batches_per_s"]) if last_metrics else 0.0
        state["last_samples_per_s"] = float(last_metrics["samples_per_s"]) if last_metrics else 0.0
        state["buffer_sample_count"] = int(prune_result["remaining_samples"])
        state["buffer_file_count"] = int(prune_result["remaining_files"])
        state["new_samples_since_last_train"] = 0
        state["latest_model_path"] = LATEST_MODEL_PATH
        state["heartbeat_at"] = time.time()
        state["last_pruned_files"] = len(prune_result["removed_files"])
        _write_training_state(state)

        _append_training_history(
            {
                "finished_at": train_finished,
                "runtime_profile": PROFILE.name,
                "loss": float(last_metrics["loss"]) if last_metrics else 0.0,
                "policy_loss": float(last_metrics["policy_loss"]) if last_metrics else 0.0,
                "value_loss": float(last_metrics["value_loss"]) if last_metrics else 0.0,
                "window_samples": int(window_samples),
                "window_files": len(files),
                "train_samples_per_cycle": int(sample_target),
                "train_optimizer": TRAIN_OPTIMIZER,
                "train_lr": float(TRAIN_LR),
                "train_weight_decay": float(TRAIN_WEIGHT_DECAY),
                "train_grad_accum_steps": int(TRAIN_GRAD_ACCUM_STEPS),
                "new_samples": int(new_samples),
                "buffer_samples": int(state["buffer_sample_count"]),
                "buffer_files": int(state["buffer_file_count"]),
                "batches": int(last_metrics["batches"]) if last_metrics else 0,
                "samples_per_s": float(last_metrics["samples_per_s"]) if last_metrics else 0.0,
                "batches_per_s": float(last_metrics["batches_per_s"]) if last_metrics else 0.0,
                "avg_batch_time_ms": float(last_metrics["avg_batch_time_ms"]) if last_metrics else 0.0,
                "scan_duration_s": float(scan_duration_s),
                "window_build_duration_s": float(window_build_duration_s),
                "train_duration_s": float(train_phase_duration_s),
                "checkpoint_duration_s": float(checkpoint_duration_s),
                "prune_duration_s": float(prune_duration_s),
                "duration_s": float(train_finished - train_started),
            }
        )


if __name__ == "__main__":
    main()
