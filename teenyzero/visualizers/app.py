import os
import math
import sys
import time
import json
import atexit
import socket
import signal
import threading
import subprocess
from pathlib import Path
import chess
import torch
from flask import Flask, request, jsonify, render_template, redirect
from teenyzero.autotune.catalog.recommendations import load_recommendations
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs
from teenyzero.alphazero.backend import create_board, move_from_uci
from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile, get_runtime_selection, runtime_profile_payload
from teenyzero.alphazero.search_session import SearchSession
from teenyzero.mcts.search import MCTS
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.paths import (
    ARENA_HISTORY_PATH,
    ARENA_LOCK_PATH,
    ARENA_MATCHES_PATH,
    ARENA_STATE_PATH,
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    MODEL_ARCHIVE_PATH,
    REPLAY_BUFFER_PATH,
    TRAINER_LOCK_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_STATE_PATH,
    ensure_runtime_dirs,
)

VISUALIZERS_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(VISUALIZERS_ROOT),
    static_folder=str(VISUALIZERS_ROOT),
    static_url_path="/static",
)


def _request_host_with_port(port: int) -> str:
    host = request.host
    if host.startswith("["):
        end = host.find("]")
        hostname = host[: end + 1] if end != -1 else host
    else:
        hostname = host.rsplit(":", 1)[0] if ":" in host else host
    return f"{request.scheme}://{hostname}:{port}/"

# Fallback engine (will be overwritten if launched via run.py)
RUNTIME_SELECTION = get_runtime_selection()
model = build_model()
evaluator = AlphaZeroEvaluator(model=model, device=RUNTIME_SELECTION.device)


def _resolve_play_simulations():
    raw_value = os.environ.get("TEENYZERO_PLAY_SIMULATIONS", "").strip()
    if raw_value.isdigit():
        return max(1, int(raw_value))
    return max(160, get_runtime_profile().arena_simulations)


engine = MCTS(
    evaluator=evaluator,
    params={
        "SIMULATIONS": _resolve_play_simulations(),
        "PARALLEL_THREADS": 1,
        "VIRTUAL_LOSS": 0.0,
        "LEAF_BATCH_SIZE": max(8, get_runtime_profile().selfplay_leaf_batch_size // 2),
    },
)
search_session = SearchSession(engine)

board = create_board()
_actor_process = None
_actor_lock = threading.Lock()
_arena_process = None
_arena_lock = threading.Lock()
_trainer_process = None
_trainer_lock = threading.Lock()
_play_model_mtime = None
RUNTIME_PROFILE = get_runtime_profile()
RUNTIME_PROFILE_SETTINGS = runtime_profile_payload(RUNTIME_PROFILE)


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts"


def _script_path(*parts: str) -> Path:
    return _scripts_dir().joinpath(*parts)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _load_json_payload(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _json_safe(json.load(handle))
    except Exception:
        return default


def _enrich_runtime_payload(payload):
    if not isinstance(payload, dict):
        return payload
    enriched = dict(payload)
    enriched["runtime_profile"] = enriched.get("runtime_profile") or RUNTIME_PROFILE.name
    enriched["runtime_profile_settings"] = enriched.get("runtime_profile_settings") or RUNTIME_PROFILE_SETTINGS
    return enriched


def _trainer_pid():
    return _pid_from_lock(TRAINER_LOCK_PATH)


def _arena_pid():
    return _pid_from_lock(ARENA_LOCK_PATH)


def _pid_from_lock(lock_path: Path):
    if not lock_path.exists():
        return None
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
        return pid if pid > 0 else None
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
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
        state = result.stdout.strip().upper()
        if state.startswith("Z"):
            return False
    except Exception:
        pass

    return True


def _trainer_running() -> bool:
    pid = _trainer_pid()
    return pid is not None and _pid_is_running(pid)


def _arena_running() -> bool:
    pid = _arena_pid()
    return pid is not None and _pid_is_running(pid)


def _wait_for_trainer_running(expected, timeout_s=8.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _trainer_running() == expected:
            return True
        time.sleep(0.1)
    return _trainer_running() == expected


def _wait_for_arena_running(expected, timeout_s=8.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _arena_running() == expected:
            return True
        time.sleep(0.1)
    return _arena_running() == expected


def _cleanup_trainer_process():
    global _trainer_process
    if _trainer_process is not None and _trainer_process.poll() is None:
        _trainer_process.terminate()
        try:
            _trainer_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _trainer_process.kill()
    _trainer_process = None


def _cleanup_arena_process():
    global _arena_process
    if _arena_process is not None and _arena_process.poll() is None:
        _arena_process.terminate()
        try:
            _arena_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _arena_process.kill()
    _arena_process = None


def _stop_trainer_process():
    global _trainer_process
    with _trainer_lock:
        pid = _trainer_pid()
        was_running = pid is not None and _pid_is_running(pid)
        forced = False

        if pid is not None and was_running:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass

            if was_running and not _wait_for_trainer_running(False, timeout_s=5.0):
                forced = True
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
                _wait_for_trainer_running(False, timeout_s=2.0)

        if _trainer_process is not None and _trainer_process.poll() is None:
            _cleanup_trainer_process()

        stopped = not _trainer_running()
        if stopped:
            TRAINER_LOCK_PATH.unlink(missing_ok=True)
            _trainer_process = None

        return {
            "was_running": was_running,
            "forced": forced,
            "stopped": stopped,
        }


def _ensure_trainer_running():
    global _trainer_process

    if _trainer_running():
        return True

    with _trainer_lock:
        if _trainer_running():
            return True

        run_trainer = _script_path("train.py")
        _trainer_process = subprocess.Popen(
            [sys.executable, str(run_trainer)],
            cwd=str(_scripts_dir().parent),
        )

    return _wait_for_trainer_running(True, timeout_s=8.0)


def _stop_arena_process():
    global _arena_process
    with _arena_lock:
        pid = _arena_pid()
        was_running = pid is not None and _pid_is_running(pid)
        forced = False

        if pid is not None and was_running:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass

            if was_running and not _wait_for_arena_running(False, timeout_s=5.0):
                forced = True
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
                _wait_for_arena_running(False, timeout_s=2.0)

        if _arena_process is not None and _arena_process.poll() is None:
            _cleanup_arena_process()

        stopped = not _arena_running()
        if stopped:
            ARENA_LOCK_PATH.unlink(missing_ok=True)
            _arena_process = None

        return {
            "was_running": was_running,
            "forced": forced,
            "stopped": stopped,
        }


def _ensure_arena_running():
    global _arena_process

    if _arena_running():
        return True

    with _arena_lock:
        if _arena_running():
            return True

        run_arena = _script_path("run_arena.py")
        _arena_process = subprocess.Popen(
            [sys.executable, str(run_arena)],
            cwd=str(_scripts_dir().parent),
        )

    return _wait_for_arena_running(True, timeout_s=8.0)


def _reset_training_artifacts():
    global _play_model_mtime
    ensure_runtime_dirs()
    REPLAY_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    removed_files = 0
    for path in REPLAY_BUFFER_PATH.glob("*.npz"):
        path.unlink(missing_ok=True)
        removed_files += 1

    removed_archives = 0
    if MODEL_ARCHIVE_PATH.exists():
        for path in MODEL_ARCHIVE_PATH.glob("*.pth"):
            path.unlink(missing_ok=True)
            removed_archives += 1

    for path in (
        TRAINING_STATE_PATH,
        TRAINING_HISTORY_PATH,
        TRAINER_LOCK_PATH,
        ARENA_STATE_PATH,
        ARENA_HISTORY_PATH,
        ARENA_MATCHES_PATH,
        ARENA_LOCK_PATH,
    ):
        path.unlink(missing_ok=True)

    fresh_model = build_model()
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(fresh_model, BEST_MODEL_PATH)
    save_checkpoint(fresh_model, LATEST_MODEL_PATH)

    evaluator.model.load_state_dict(fresh_model.state_dict())
    evaluator.model.eval()
    evaluator.clear_cache()
    _play_model_mtime = None

    return {
        "ok": True,
        "removed_replay_files": removed_files,
        "removed_archive_models": removed_archives,
        "best_model_path": str(BEST_MODEL_PATH),
        "latest_model_path": str(LATEST_MODEL_PATH),
    }


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_in_use(port):
            return True
        time.sleep(0.1)
    return False


def _cleanup_actor_process():
    global _actor_process
    if _actor_process is not None and _actor_process.poll() is None:
        _actor_process.terminate()
        try:
            _actor_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _actor_process.kill()
    _actor_process = None


def _ensure_actor_cluster_running():
    global _actor_process

    if _port_in_use(5002):
        return True

    with _actor_lock:
        if _port_in_use(5002):
            return True

        if _actor_process is None or _actor_process.poll() is not None:
            run_actors = _script_path("run_actors.py")
            _actor_process = subprocess.Popen(
                [sys.executable, str(run_actors)],
                cwd=str(_scripts_dir().parent),
            )

    return _wait_for_port(5002, timeout_s=8.0)


atexit.register(_cleanup_actor_process)
atexit.register(_cleanup_arena_process)
atexit.register(_cleanup_trainer_process)


def _root_win_prob(root):
    if root is None or getattr(root, "total_n", 0.0) <= 0.0:
        return 50.0

    q = float(root.total_w / max(root.total_n, 1.0))
    return round(float((q + 1.0) * 50.0), 1)


def _maybe_reload_play_model():
    global _play_model_mtime, search_session
    model_path = LATEST_MODEL_PATH
    if not model_path.exists():
        return

    mtime = model_path.stat().st_mtime
    if _play_model_mtime is not None and mtime <= _play_model_mtime:
        return

    load_result = load_checkpoint(evaluator.model, model_path, map_location=evaluator.device, allow_partial=True)
    if load_result["loaded"]:
        evaluator.model.eval()
        evaluator.clear_cache()
        search_session.reset()
        _play_model_mtime = mtime

@app.route("/")
def index():
    return render_template("hub/home.html")

@app.route("/play")
def play():
    return render_template("gameplay/play.html")


@app.route("/training")
def training():
    return render_template("training_status/status.html")


@app.route("/arena")
def arena():
    _ensure_arena_running()
    return render_template("arena_status/status.html")


@app.route("/autotune")
def autotune():
    return render_template("autotune/status.html")


@app.route("/monitor")
def monitor():
    _ensure_actor_cluster_running()
    return redirect(_request_host_with_port(5002), code=302)


@app.route("/api/training_status")
def training_status():
    return jsonify(_enrich_runtime_payload(_load_json_payload(TRAINING_STATE_PATH, {"status": "idle"})))


@app.route("/api/training_history")
def training_history():
    return jsonify(_load_json_payload(TRAINING_HISTORY_PATH, []))


@app.route("/api/arena_status")
def arena_status():
    _ensure_arena_running()
    return jsonify(_enrich_runtime_payload(_load_json_payload(ARENA_STATE_PATH, {"status": "idle"})))


@app.route("/api/arena_history")
def arena_history():
    _ensure_arena_running()
    return jsonify(_load_json_payload(ARENA_HISTORY_PATH, []))


@app.route("/api/autotune_status")
def autotune_status():
    return jsonify(latest_autotune_run() or {"status": "idle", "trials": []})


@app.route("/api/autotune_runs")
def autotune_runs():
    return jsonify(list_autotune_runs())


@app.route("/api/autotune_recommendations")
def autotune_recommendations():
    return jsonify(load_recommendations())


@app.route("/api/training/reset", methods=["POST"])
def reset_training():
    arena_stop_result = _stop_arena_process()
    if not arena_stop_result["stopped"]:
        return jsonify({
            "ok": False,
            "error": "Failed to stop the arena process for reset.",
        }), 409

    stop_result = _stop_trainer_process()
    if not stop_result["stopped"]:
        return jsonify({
            "ok": False,
            "error": "Failed to stop the trainer process for reset.",
        }), 409

    payload = _reset_training_artifacts()
    trainer_restarted = _ensure_trainer_running()
    arena_restarted = _ensure_arena_running()
    payload.update(
        {
            "trainer_was_running": stop_result["was_running"],
            "trainer_forced_stop": stop_result["forced"],
            "trainer_restarted": trainer_restarted,
            "arena_was_running": arena_stop_result["was_running"],
            "arena_forced_stop": arena_stop_result["forced"],
            "arena_restarted": arena_restarted,
        }
    )
    if not trainer_restarted or not arena_restarted:
        payload["ok"] = False
        payload["error"] = "Training state was reset, but a background process did not restart automatically."
        return jsonify(payload), 500
    return jsonify(payload)

@app.route("/move", methods=["POST"])
def move():
    global board, search_session
    _maybe_reload_play_model()
    data = request.json
    uci = data.get("uci")
    
    try:
        move = move_from_uci(uci)
        if move in board.legal_moves:
            # 1. Human Move
            board.push(move)
            
            # 2. Engine Response
            if not board.is_game_over():
                engine_move, _, root = search_session.search(board)
                # Check legality before pushing to avoid AssertionError
                if engine_move in board.legal_moves:
                    board.push(engine_move)
                
                return jsonify({
                    "ok": True,
                    "fen": board.fen(),
                    "engine_move": engine_move.uci(),
                    "win_prob": _root_win_prob(root)
                })
            
            return jsonify({"ok": True, "fen": board.fen()})
        return jsonify({"ok": False, "error": "Illegal move"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/reset", methods=["POST"])
def reset():
    global board, search_session
    _maybe_reload_play_model()
    data = request.json or {}
    side = data.get("side", "w")
    
    # Strictly reset to a fresh board instance
    board = create_board()
    search_session.reset()
    
    # If playing as Black, Engine makes the first move for White
    if side == "b":
        engine_move, _, root = search_session.search(board)
        if engine_move in board.legal_moves:
            board.push(engine_move)
            return jsonify({
                "ok": True, 
                "fen": board.fen(), 
                "engine_move": engine_move.uci(),
                "win_prob": _root_win_prob(root)
            })
        
    return jsonify({"ok": True, "fen": board.fen(), "win_prob": 50.0})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
