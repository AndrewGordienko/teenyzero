import json
import os
import time
import atexit

import torch

from teenyzero.alphazero.logic.trainer import (
    AlphaTrainer,
    dataloader_for_replay_window,
    prune_replay_buffer,
    replay_buffer_summary,
)
from teenyzero.alphazero.model import AlphaNet


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "teenyzero", "alphazero", "data", "replay_buffer")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pth")
TRAINING_STATE_PATH = os.path.join(PROJECT_ROOT, "teenyzero", "alphazero", "data", "training_state.json")
TRAINER_LOCK_PATH = os.path.join(PROJECT_ROOT, "teenyzero", "alphazero", "data", "trainer.lock")
TRAINING_HISTORY_PATH = os.path.join(PROJECT_ROOT, "teenyzero", "alphazero", "data", "training_history.json")

MIN_SAMPLES_READY = 20_000
TRAIN_INCREMENT = 20_000
REPLAY_WINDOW_SAMPLES = 25_000
BOOTSTRAP_WINDOW_SAMPLES = 200_000
MAX_RETAINED_SAMPLES = 300_000
BATCH_SIZE = 64
EPOCHS_PER_CYCLE = 1
POLL_INTERVAL_S = 10.0


def _load_training_state():
    if not os.path.exists(TRAINING_STATE_PATH):
        return {
            "status": "waiting",
            "last_trained_sample_count": 0,
            "last_train_cutoff_mtime": 0.0,
            "training_cycles": 0,
            "last_loss": 0.0,
            "last_policy_loss": 0.0,
            "last_value_loss": 0.0,
            "last_cycle_samples": 0,
            "last_window_samples": 0,
            "last_train_started_at": None,
            "last_train_finished_at": None,
            "last_train_duration_s": 0.0,
            "buffer_sample_count": 0,
            "buffer_file_count": 0,
            "new_samples_since_last_train": 0,
            "min_samples_ready": MIN_SAMPLES_READY,
            "train_increment": TRAIN_INCREMENT,
            "replay_window_samples": REPLAY_WINDOW_SAMPLES,
            "bootstrap_window_samples": BOOTSTRAP_WINDOW_SAMPLES,
            "max_retained_samples": MAX_RETAINED_SAMPLES,
        }

    with open(TRAINING_STATE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_training_state(state):
    os.makedirs(os.path.dirname(TRAINING_STATE_PATH), exist_ok=True)
    tmp_path = TRAINING_STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, TRAINING_STATE_PATH)


def _mark_stage(state, stage, **extra):
    state["status"] = stage
    state["heartbeat_at"] = time.time()
    for key, value in extra.items():
        state[key] = value
    _write_training_state(state)


def _append_training_history(entry):
    os.makedirs(os.path.dirname(TRAINING_HISTORY_PATH), exist_ok=True)
    history = []
    if os.path.exists(TRAINING_HISTORY_PATH):
        try:
            with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as handle:
                history = json.load(handle)
        except Exception:
            history = []

    history.append(entry)
    history = history[-200:]

    tmp_path = TRAINING_HISTORY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    os.replace(tmp_path, TRAINING_HISTORY_PATH)


def _acquire_trainer_lock():
    os.makedirs(os.path.dirname(TRAINER_LOCK_PATH), exist_ok=True)
    if os.path.exists(TRAINER_LOCK_PATH):
        try:
            with open(TRAINER_LOCK_PATH, "r", encoding="utf-8") as handle:
                existing_pid = int(handle.read().strip())
            os.kill(existing_pid, 0)
            return False
        except Exception:
            pass

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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = AlphaNet(num_res_blocks=10, channels=128)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    trainer = AlphaTrainer(model, device=device)
    state = _load_training_state()
    state["device"] = device
    state["training_history_path"] = TRAINING_HISTORY_PATH
    _write_training_state(state)

    print("[Trainer] Continuous trainer online.")

    while True:
        _mark_stage(state, "scanning_replay_buffer")
        summary = replay_buffer_summary(DATA_DIR)
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
        state["status"] = "waiting"
        state["heartbeat_at"] = time.time()
        _write_training_state(state)

        if total_samples < MIN_SAMPLES_READY or new_samples < TRAIN_INCREMENT:
            time.sleep(POLL_INTERVAL_S)
            continue

        train_started = time.time()
        state["last_train_started_at"] = train_started
        _mark_stage(state, "building_replay_window")
        current_window = BOOTSTRAP_WINDOW_SAMPLES if int(state.get("training_cycles", 0)) == 0 else REPLAY_WINDOW_SAMPLES
        state["active_window_samples"] = current_window
        _write_training_state(state)

        def on_window_progress(progress):
            _mark_stage(
                state,
                progress.get("stage", "loading_replay_window"),
                loaded_files=int(progress.get("loaded_files", 0)),
                total_window_files=int(progress.get("total_files", 0)),
                loaded_window_samples=int(progress.get("loaded_samples", 0)),
            )

        loader, window_samples, files = dataloader_for_replay_window(
            DATA_DIR,
            max_samples=current_window,
            batch_size=BATCH_SIZE,
            shuffle=True,
            progress_callback=on_window_progress,
        )

        last_metrics = None
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
                )

            last_metrics = trainer.train_epoch(loader, progress_callback=on_train_progress)

        _mark_stage(state, "saving_checkpoint")
        trainer.save_checkpoint(MODEL_PATH)
        trainer.save_checkpoint(LATEST_MODEL_PATH)
        prune_result = prune_replay_buffer(DATA_DIR, max_samples_to_keep=MAX_RETAINED_SAMPLES)

        train_finished = time.time()
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
                "loss": float(last_metrics["loss"]) if last_metrics else 0.0,
                "policy_loss": float(last_metrics["policy_loss"]) if last_metrics else 0.0,
                "value_loss": float(last_metrics["value_loss"]) if last_metrics else 0.0,
                "window_samples": int(window_samples),
                "duration_s": float(train_finished - train_started),
            }
        )


if __name__ == "__main__":
    main()
