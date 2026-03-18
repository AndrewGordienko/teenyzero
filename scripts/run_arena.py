import atexit
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

import chess
import chess.engine

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile, get_runtime_selection, runtime_profile_payload
from teenyzero.alphazero.search_session import SearchSession
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.paths import (
    ARENA_HISTORY_PATH,
    ARENA_LOCK_PATH,
    ARENA_MATCHES_PATH,
    ARENA_STATE_PATH,
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    MODEL_ARCHIVE_PATH,
    TRAINING_STATE_PATH,
    ensure_runtime_dirs,
    runtime_paths_payload,
)


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile
PROFILE_SETTINGS = runtime_profile_payload(PROFILE)

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, "").strip()
    if raw_value.isdigit():
        return max(minimum, int(raw_value))
    return int(default)

def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw_value = os.environ.get(name, "").strip()
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return float(default)

PROMOTION_GAMES = _env_int("TEENYZERO_ARENA_PROMOTION_GAMES", PROFILE.arena_promotion_games)
PROMOTION_THRESHOLD = PROFILE.arena_promotion_threshold
BASELINE_GAMES = _env_int("TEENYZERO_ARENA_BASELINE_GAMES", PROFILE.arena_baseline_games)
ARENA_MAX_PLIES = 180
ARENA_SIMULATIONS = _env_int("TEENYZERO_ARENA_SIMULATIONS", PROFILE.arena_simulations)
STOCKFISH_TIME_S = _env_float("TEENYZERO_STOCKFISH_TIME_MS", 50.0, minimum=1.0) / 1000.0
POLL_INTERVAL_S = 15.0
ELO_K = 24.0
DEFAULT_RATING = 1000.0
ARCHIVE_DIR = MODEL_ARCHIVE_PATH
MAX_CYCLE_ARCHIVES = max(1, int(os.environ.get("TEENYZERO_MAX_CYCLE_ARCHIVES", "2")))
MAX_CHAMPION_ARCHIVES = max(1, int(os.environ.get("TEENYZERO_MAX_CHAMPION_ARCHIVES", "3")))


OPENING_BOOK = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp_path, path)


def _default_arena_state():
    return {
        "status": "waiting",
        "heartbeat_at": None,
        "last_evaluated_cycle": 0,
        "last_latest_mtime": 0.0,
        "candidate_id": None,
        "candidate_rating": DEFAULT_RATING,
        "champion_id": None,
        "champion_rating": DEFAULT_RATING,
        "best_cycle": 0,
        "promotion_games": PROMOTION_GAMES,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "baseline_games": BASELINE_GAMES,
        "arena_simulations": ARENA_SIMULATIONS,
        "stockfish_time_s": STOCKFISH_TIME_S,
        "runtime_profile": PROFILE.name,
        "runtime_profile_settings": PROFILE_SETTINGS,
        "runtime_paths": runtime_paths_payload(),
        "stockfish_path": os.environ.get("TEENYZERO_STOCKFISH_PATH", ""),
        "stockfish_available": False,
        "latest_match": None,
        "recent_matches": [],
        "ratings": {},
        "external_results": [],
        "champion_archive_path": "",
        "last_error": None,
    }


def _acquire_lock():
    ARENA_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARENA_LOCK_PATH.exists():
        try:
            pid = int(ARENA_LOCK_PATH.read_text(encoding="utf-8").strip())
            if _pid_is_running(pid):
                return False
        except Exception:
            pass
        ARENA_LOCK_PATH.unlink(missing_ok=True)
    ARENA_LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_lock():
    ARENA_LOCK_PATH.unlink(missing_ok=True)


def _elo_expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


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


def _elo_update(ratings, player_a, player_b, score_a, k=ELO_K, fixed_b=False):
    rating_a = float(ratings.get(player_a, DEFAULT_RATING))
    rating_b = float(ratings.get(player_b, DEFAULT_RATING))
    expected_a = _elo_expected(rating_a, rating_b)
    delta = k * (score_a - expected_a)
    ratings[player_a] = rating_a + delta
    if not fixed_b:
        ratings[player_b] = rating_b - delta
    return ratings[player_a], rating_b if fixed_b else ratings[player_b]


def _load_training_cycle():
    state = _load_json(TRAINING_STATE_PATH, {})
    return int(state.get("training_cycles", 0)), state


def _archive_model(src: Path, cycle: int, label="cycle"):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE_DIR / f"{label}_{cycle:05d}.pth"
    shutil.copy2(src, dst)
    return dst


def _prune_archives(label: str, keep: int):
    if keep <= 0 or not ARCHIVE_DIR.exists():
        return []
    paths = sorted(ARCHIVE_DIR.glob(f"{label}_*.pth"))
    removed = []
    while len(paths) > keep:
        oldest = paths.pop(0)
        oldest.unlink(missing_ok=True)
        removed.append(str(oldest))
    return removed


def _recent_archived_champions(current_champion_archive_path, limit=3):
    if not ARCHIVE_DIR.exists():
        return []
    paths = sorted(ARCHIVE_DIR.glob("champion_*.pth"))
    selected = []
    current_path = Path(current_champion_archive_path) if current_champion_archive_path else None
    for path in reversed(paths):
        if current_path is not None and path == current_path:
            continue
        checkpoint_id = path.stem
        selected.append((checkpoint_id, path))
        if len(selected) >= limit:
            break
    selected.reverse()
    return selected


class TeenyZeroAgent:
    def __init__(self, model_path: Path, device: str, simulations=ARENA_SIMULATIONS):
        self.model = build_model()
        load_checkpoint(self.model, model_path, map_location=device, allow_partial=True)
        self.model.eval()
        evaluator = AlphaZeroEvaluator(model=self.model, device=device, use_cache=True)
        self.engine = MCTS(
            evaluator=evaluator,
            params={
                "SIMULATIONS": simulations,
                "C_PUCT": 1.35,
                "FPU_REDUCTION": 0.35,
                "LEAF_BATCH_SIZE": 8,
            },
        )
        self.session = SearchSession(self.engine)

    def choose_move(self, board: chess.Board):
        move, _, _ = self.session.search(board, is_training=False)
        return move

    def close(self):
        self.session.reset()
        return None


class UCIEngineAgent:
    def __init__(self, engine_path: str, options: dict, limit: chess.engine.Limit):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure(options)
        self.limit = limit

    def choose_move(self, board: chess.Board):
        result = self.engine.play(board, self.limit)
        return result.move

    def close(self):
        self.engine.quit()


def _apply_opening(board: chess.Board, opening_line):
    for uci in opening_line:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)


def _play_game(white_agent, black_agent, opening_line):
    board = chess.Board()
    _apply_opening(board, opening_line)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < ARENA_MAX_PLIES:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent.choose_move(board)
        if move is None or move not in board.legal_moves:
            return "0-1" if board.turn == chess.WHITE else "1-0", board, plies
        board.push(move)
        plies += 1
    return board.result(claim_draw=True), board, plies


def _result_score(result):
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def _opening_for_game(index):
    return OPENING_BOOK[index % len(OPENING_BOOK)]


def _match_summary(player_a_id, player_b_id, games):
    wins = sum(1 for game in games if game["score_a"] == 1.0)
    draws = sum(1 for game in games if game["score_a"] == 0.5)
    losses = sum(1 for game in games if game["score_a"] == 0.0)
    score = (wins + 0.5 * draws) / max(len(games), 1)
    return {
        "player_a": player_a_id,
        "player_b": player_b_id,
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "played_at": time.time(),
        "details": games,
    }


def _play_teeny_match(player_a_id, player_a_path, player_b_id, player_b_path, device, num_games):
    white_a = TeenyZeroAgent(player_a_path, device=device)
    white_b = TeenyZeroAgent(player_b_path, device=device)
    games = []
    try:
        for game_idx in range(num_games):
            opening = _opening_for_game(game_idx)
            if game_idx % 2 == 0:
                result, board, plies = _play_game(white_a, white_b, opening)
                score_a = _result_score(result)
                color = "white"
            else:
                result, board, plies = _play_game(white_b, white_a, opening)
                score_a = 1.0 - _result_score(result)
                color = "black"
            games.append(
                {
                    "color": color,
                    "result": result,
                    "score_a": score_a,
                    "plies": plies,
                    "final_fen": board.fen(),
                    "opening": opening,
                }
            )
    finally:
        white_a.close()
        white_b.close()
    return _match_summary(player_a_id, player_b_id, games)


def _stockfish_opponents(stockfish_path):
    if not stockfish_path or not Path(stockfish_path).exists():
        return []
    return [
        {
            "id": "stockfish_1320",
            "label": "Stockfish 1320",
            "rating": 1320.0,
            "options": {"UCI_LimitStrength": True, "UCI_Elo": 1320, "Threads": 1, "Hash": 32},
        },
        {
            "id": "stockfish_1600",
            "label": "Stockfish 1600",
            "rating": 1600.0,
            "options": {"UCI_LimitStrength": True, "UCI_Elo": 1600, "Threads": 1, "Hash": 32},
        },
        {
            "id": "stockfish_2000",
            "label": "Stockfish 2000",
            "rating": 2000.0,
            "options": {"UCI_LimitStrength": True, "UCI_Elo": 2000, "Threads": 1, "Hash": 32},
        },
    ]


def _play_external_match(player_id, player_path, opponent, device, num_games):
    teeny = TeenyZeroAgent(player_path, device=device)
    stockfish = UCIEngineAgent(
        opponent["path"],
        opponent["options"],
        chess.engine.Limit(time=STOCKFISH_TIME_S),
    )
    games = []
    try:
        for game_idx in range(num_games):
            opening = _opening_for_game(game_idx)
            if game_idx % 2 == 0:
                result, board, plies = _play_game(teeny, stockfish, opening)
                score_a = _result_score(result)
                color = "white"
            else:
                result, board, plies = _play_game(stockfish, teeny, opening)
                score_a = 1.0 - _result_score(result)
                color = "black"
            games.append(
                {
                    "color": color,
                    "result": result,
                    "score_a": score_a,
                    "plies": plies,
                    "final_fen": board.fen(),
                    "opening": opening,
                }
            )
    finally:
        teeny.close()
        stockfish.close()
    return _match_summary(player_id, opponent["id"], games)


def _append_history(history_path: Path, entry):
    history = _load_json(history_path, [])
    history.append(_json_safe(entry))
    history = history[-240:]
    _write_json(history_path, history)


def _append_matches(entry):
    matches = _load_json(ARENA_MATCHES_PATH, [])
    matches.append(_json_safe(entry))
    matches = matches[-300:]
    _write_json(ARENA_MATCHES_PATH, matches)


def _update_status(state, **extra):
    now = time.time()
    state["heartbeat_at"] = now
    for key, value in extra.items():
        state[key] = value
    _write_json(ARENA_STATE_PATH, state)


def main():
    if not _acquire_lock():
        print("[Arena] Another arena process is already running. Exiting.")
        return

    atexit.register(_release_lock)
    ensure_runtime_dirs()
    device = RUNTIME.device
    stockfish_path = os.environ.get("TEENYZERO_STOCKFISH_PATH", "").strip()
    state = _default_arena_state()
    state.update(_load_json(ARENA_STATE_PATH, {}))
    state["device"] = device
    state["runtime_profile"] = PROFILE.name
    state["runtime_profile_settings"] = PROFILE_SETTINGS
    state["runtime_paths"] = runtime_paths_payload()
    state["promotion_games"] = PROMOTION_GAMES
    state["baseline_games"] = BASELINE_GAMES
    state["arena_simulations"] = ARENA_SIMULATIONS
    state["stockfish_time_s"] = STOCKFISH_TIME_S
    state["stockfish_path"] = stockfish_path
    state["stockfish_available"] = bool(_stockfish_opponents(stockfish_path))
    _write_json(ARENA_STATE_PATH, state)

    print("[Arena] Online.")

    while True:
        try:
            cycle, training_state = _load_training_cycle()
            latest_exists = LATEST_MODEL_PATH.exists()
            latest_mtime = LATEST_MODEL_PATH.stat().st_mtime if latest_exists else 0.0

            _update_status(
                state,
                status="waiting",
                stockfish_available=bool(_stockfish_opponents(stockfish_path)),
                last_error=None,
            )

            if not latest_exists or cycle <= 0:
                time.sleep(POLL_INTERVAL_S)
                continue

            if cycle <= int(state.get("last_evaluated_cycle", 0)) and latest_mtime <= float(state.get("last_latest_mtime", 0.0)):
                time.sleep(POLL_INTERVAL_S)
                continue

            candidate_id = f"cycle_{cycle:05d}"
            _update_status(state, status="archiving_candidate", candidate_id=candidate_id)
            candidate_path = _archive_model(LATEST_MODEL_PATH, cycle, label="cycle")

            ratings = dict(state.get("ratings", {}))
            if candidate_id not in ratings:
                ratings[candidate_id] = float(state.get("champion_rating", DEFAULT_RATING))

            champion_id = state.get("champion_id")
            champion_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else None
            promotion_match = None
            promoted = False

            if champion_id and champion_path and champion_path.exists():
                _update_status(state, status="promotion_match", ratings=ratings)
                promotion_match = _play_teeny_match(
                    candidate_id,
                    candidate_path,
                    champion_id,
                    champion_path,
                    device,
                    PROMOTION_GAMES,
                )
                _append_matches({**promotion_match, "kind": "promotion"})
                candidate_rating, champion_rating = _elo_update(
                    ratings,
                    candidate_id,
                    champion_id,
                    promotion_match["score"],
                )
                promoted = promotion_match["score"] >= PROMOTION_THRESHOLD
            else:
                promoted = True
                promotion_match = {
                    "player_a": candidate_id,
                    "player_b": "bootstrap",
                    "games": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "score": 1.0,
                    "played_at": time.time(),
                    "details": [],
                }

            if promoted:
                shutil.copy2(candidate_path, BEST_MODEL_PATH)
                champion_id = candidate_id
                champion_archive_path = str(_archive_model(BEST_MODEL_PATH, cycle, label="champion"))
            else:
                champion_archive_path = str(state.get("champion_archive_path", ""))

            champion_rating = float(ratings.get(champion_id, DEFAULT_RATING))
            candidate_rating = float(ratings.get(candidate_id, champion_rating))

            external_results = []
            for opponent in _stockfish_opponents(stockfish_path):
                _update_status(state, status="external_eval", ratings=ratings, champion_id=champion_id)
                opponent = dict(opponent)
                opponent["path"] = stockfish_path
                match = _play_external_match(champion_id, BEST_MODEL_PATH, opponent, device, BASELINE_GAMES)
                _append_matches({**match, "kind": "external", "label": opponent["label"]})
                ratings.setdefault(opponent["id"], opponent["rating"])
                champion_rating, _ = _elo_update(
                    ratings,
                    champion_id,
                    opponent["id"],
                    match["score"],
                    fixed_b=True,
                )
                external_results.append(
                    {
                        "id": opponent["id"],
                        "label": opponent["label"],
                        "score": match["score"],
                        "wins": match["wins"],
                        "draws": match["draws"],
                        "losses": match["losses"],
                        "rating_anchor": opponent["rating"],
                    }
                )

            recent_champions = []
            for opponent_id, opponent_path in _recent_archived_champions(champion_archive_path):
                _update_status(state, status="checkpoint_eval", ratings=ratings, champion_id=champion_id)
                match = _play_teeny_match(champion_id, BEST_MODEL_PATH, opponent_id, opponent_path, device, BASELINE_GAMES)
                _append_matches({**match, "kind": "checkpoint"})
                champion_rating, _ = _elo_update(ratings, champion_id, opponent_id, match["score"])
                recent_champions.append(
                    {
                        "id": opponent_id,
                        "label": opponent_id.replace("_", " "),
                        "score": match["score"],
                        "wins": match["wins"],
                        "draws": match["draws"],
                        "losses": match["losses"],
                    }
                )

            removed_cycle_archives = _prune_archives("cycle", MAX_CYCLE_ARCHIVES)
            removed_champion_archives = _prune_archives("champion", MAX_CHAMPION_ARCHIVES)
            ratings[champion_id] = champion_rating
            state.update(
                {
                    "status": "waiting",
                    "heartbeat_at": time.time(),
                    "last_evaluated_cycle": cycle,
                    "last_latest_mtime": latest_mtime,
                    "candidate_id": candidate_id,
                    "candidate_rating": candidate_rating,
                    "champion_id": champion_id,
                    "champion_rating": champion_rating,
                    "best_cycle": cycle if promoted else int(state.get("best_cycle", 0)),
                    "latest_match": promotion_match,
                    "recent_matches": [promotion_match],
                    "ratings": ratings,
                    "external_results": external_results + recent_champions,
                    "stockfish_available": bool(_stockfish_opponents(stockfish_path)),
                    "champion_archive_path": champion_archive_path,
                    "removed_cycle_archives": len(removed_cycle_archives),
                    "removed_champion_archives": len(removed_champion_archives),
                    "last_error": None,
                }
            )
            _write_json(ARENA_STATE_PATH, state)
            _append_history(
                ARENA_HISTORY_PATH,
                {
                    "evaluated_at": time.time(),
                    "runtime_profile": PROFILE.name,
                    "cycle": cycle,
                    "candidate_id": candidate_id,
                    "candidate_rating": candidate_rating,
                    "champion_id": champion_id,
                    "champion_rating": champion_rating,
                    "promoted": promoted,
                    "promotion_score": promotion_match["score"],
                    "promotion_wins": promotion_match["wins"],
                    "promotion_draws": promotion_match["draws"],
                    "promotion_losses": promotion_match["losses"],
                    "external_results": external_results,
                    "checkpoint_results": recent_champions,
                },
            )
        except Exception as exc:
            state["last_error"] = str(exc)
            _update_status(state, status="error", last_error=str(exc))
            print(f"[Arena] Error: {exc}")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
