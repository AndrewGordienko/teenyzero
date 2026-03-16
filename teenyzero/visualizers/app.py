import os
import sys
import time
import json
import atexit
import socket
import threading
import subprocess
from pathlib import Path
import numpy as np
import chess
import torch
from flask import Flask, request, jsonify, render_template, redirect
from teenyzero.mcts.search import MCTS
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.alphazero.model import AlphaNet

app = Flask(__name__)

# Fallback engine (will be overwritten if launched via run.py)
model = AlphaNet() 
evaluator = AlphaZeroEvaluator(model=model)
engine = MCTS(evaluator=evaluator, params={'SIMULATIONS': 400, 'PARALLEL_THREADS': 4})

board = chess.Board()
_actor_process = None
_actor_lock = threading.Lock()
_play_model_mtime = None
TRAINING_STATE_PATH = Path(__file__).resolve().parents[1] / "alphazero" / "data" / "training_state.json"
TRAINING_HISTORY_PATH = Path(__file__).resolve().parents[1] / "alphazero" / "data" / "training_history.json"


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts"


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
            run_actors = _scripts_dir() / "run_actors.py"
            _actor_process = subprocess.Popen(
                [sys.executable, str(run_actors)],
                cwd=str(run_actors.parent.parent),
            )

    return _wait_for_port(5002, timeout_s=8.0)


atexit.register(_cleanup_actor_process)


def _root_win_prob(root):
    if root is None or not getattr(root, "N", None):
        return 50.0

    weighted_q = 0.0
    total_visits = 0
    for move, visits in root.N.items():
        if visits <= 0:
            continue
        total_visits += visits
        weighted_q += root.W[move]

    if total_visits <= 0:
        return 50.0

    q = weighted_q / total_visits
    return round(float((q + 1.0) * 50.0), 1)


def _maybe_reload_play_model():
    global _play_model_mtime
    model_path = Path(__file__).resolve().parents[2] / "models" / "latest_model.pth"
    if not model_path.exists():
        return

    mtime = model_path.stat().st_mtime
    if _play_model_mtime is not None and mtime <= _play_model_mtime:
        return

    state_dict = torch.load(model_path, map_location=evaluator.device)
    evaluator.model.load_state_dict(state_dict)
    evaluator.model.eval()
    evaluator.clear_cache()
    _play_model_mtime = mtime

@app.route("/")
def index():
    return render_template("hub/home.html")

@app.route("/play")
def play():
    return render_template("engine_play/play.html")


@app.route("/training")
def training():
    return render_template("training/status.html")


@app.route("/monitor")
def monitor():
    _ensure_actor_cluster_running()
    return redirect("http://localhost:5002", code=302)


@app.route("/api/training_status")
def training_status():
    if not TRAINING_STATE_PATH.exists():
        return jsonify({"status": "idle"})

    with open(TRAINING_STATE_PATH, "r", encoding="utf-8") as handle:
        return jsonify(json.load(handle))


@app.route("/api/training_history")
def training_history():
    if not TRAINING_HISTORY_PATH.exists():
        return jsonify([])

    with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as handle:
        return jsonify(json.load(handle))

@app.route("/move", methods=["POST"])
def move():
    global board
    _maybe_reload_play_model()
    data = request.json
    uci = data.get("uci")
    
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            # 1. Human Move
            board.push(move)
            
            # 2. Engine Response
            if not board.is_game_over():
                engine_move, _, root = engine.search(board)
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
    global board
    _maybe_reload_play_model()
    data = request.json or {}
    side = data.get("side", "w")
    
    # Strictly reset to a fresh board instance
    board = chess.Board()
    
    # If playing as Black, Engine makes the first move for White
    if side == "b":
        engine_move, _, root = engine.search(board)
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
