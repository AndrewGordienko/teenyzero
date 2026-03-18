import os
import errno
import json
import numpy as np
import chess
from collections import Counter, deque

from teenyzero.alphazero.backend import create_board
from teenyzero.alphazero.config import REPLAY_ENCODER_VERSION
from teenyzero.alphazero.runtime import get_runtime_profile
from teenyzero.paths import runtime_free_bytes, runtime_low_disk_watermark_bytes


PROFILE = get_runtime_profile()


class DataCollector:
    def __init__(self, evaluator, engine, buffer_path="data/replay_buffer"):
        self.evaluator = evaluator
        self.engine = engine

        self.buffer_path = os.path.abspath(buffer_path)
        os.makedirs(self.buffer_path, exist_ok=True)

        # Exploration / diversity controls
        self.EXPLORATION_GAMES_THRESHOLD = 2000
        self.FORCE_RANDOM_PLIES = 8
        self.MAX_GAME_LENGTH = 160
        self.TEMP_PLIES_MIN = 18
        self.TEMP_PLIES_MAX = 30
        self.OPENING_TEMPERATURE = 1.45
        self.MIDGAME_TEMPERATURE = 1.18
        self.RESIGN_AFTER_PLIES = 24
        self.RESIGN_VALUE_THRESHOLD = -0.92
        self.RESIGN_STREAK = 3
        self.CAPTURED_GAME_VALUE_THRESHOLD = 0.85
        self.CAPTURED_GAME_MATERIAL_THRESHOLD = 2.5
        self.AVOID_DRAW_REPETITION_PLIES = 120

        self.total_games = 0
        self.total_samples = 0
        self.total_positions_saved = 0
        self.total_game_time_ms = 0.0
        self.white_wins = 0
        self.draws = 0
        self.black_wins = 0

        self.opening_stats = Counter()
        self.hall_of_fame = deque(maxlen=8)
        self.profile_samples = 0
        self.profile_totals = {
            "total": 0.0,
            "root_eval": 0.0,
            "selection": 0.0,
            "leaf_eval": 0.0,
            "backprop": 0.0,
            "encode": 0.0,
            "policy_mask": 0.0,
            "inference_wait": 0.0,
            "inference_forward": 0.0,
            "simulations": 0.0,
            "positions_evaluated": 0.0,
            "cache_hits": 0.0,
            "cache_misses": 0.0,
        }
        self.last_profile = None
        self.replay_shard_format = self._resolve_replay_shard_format()

    def _low_disk_pressure(self):
        return runtime_free_bytes() <= runtime_low_disk_watermark_bytes()

    def _resolve_replay_shard_format(self):
        raw_value = os.environ.get("TEENYZERO_REPLAY_SHARD_FORMAT", "").strip().lower()
        if raw_value in {"npz", "raw"}:
            return raw_value
        return "raw" if not PROFILE.replay_compress else "npz"

    def _raw_shard_paths(self, filename):
        stem = filename[:-4] if filename.endswith(".npz") else filename
        base = os.path.join(self.buffer_path, stem)
        return {
            "stem": base,
            "states": base + ".states.npy",
            "pis": base + ".pis.npy",
            "zs": base + ".zs.npy",
            "meta": base + ".meta.json",
        }

    def collect_game(self, worker_id=0, stats_dict=None):
        import time
        game_start = time.perf_counter()
        board = create_board()
        game_history = []
        move_count = 0
        root = None
        resign_streak = {
            chess.WHITE: 0,
            chess.BLACK: 0,
        }
        forced_outcome = None

        # Early training exploration
        random_prob = max(0.0, 1.0 - (self.total_games / self.EXPLORATION_GAMES_THRESHOLD))
        is_forced_exploration = np.random.random() < random_prob

        # Keep temperature active longer so self-play does not collapse into
        # short symmetric lines that repeatedly get labeled as draws.
        temp_threshold = np.random.randint(self.TEMP_PLIES_MIN, self.TEMP_PLIES_MAX + 1)

        while not board.is_game_over(claim_draw=True) and move_count < self.MAX_GAME_LENGTH:
            best_move, pi_dist, root = self.engine.search(
                board,
                is_training=True,
                root=root,
            )
            search_profile = self.engine.last_search_stats
            self._record_search_profile(search_profile)
            temperature = self.OPENING_TEMPERATURE if move_count < self.FORCE_RANDOM_PLIES else self.MIDGAME_TEMPERATURE
            current_side = board.turn
            root_value = self._root_value(root)

            if self._should_resign(current_side, root_value, move_count, resign_streak):
                forced_outcome = -1.0 if current_side == chess.WHITE else 1.0
                break

            # Failsafe
            if best_move is None:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                best_move = np.random.choice(legal_moves)
                pi_dist = self._uniform_pi_dist(legal_moves)

            # Selection policy for self-play
            if is_forced_exploration and move_count < self.FORCE_RANDOM_PLIES:
                selected_move = np.random.choice(list(board.legal_moves))
            elif move_count < temp_threshold:
                selected_move = self._sample_from_pi(pi_dist, best_move, board, temperature=temperature)
            else:
                selected_move = best_move
            selected_move = self._avoid_draw_repetition(board, selected_move, pi_dist, best_move, move_count)

            # Save training example BEFORE pushing the move
            state = np.array(self.evaluator._encode_cached(board), copy=True)
            target_pi = self._apply_temperature(pi_dist, temperature) if move_count < temp_threshold else pi_dist
            pi_vector = self._dist_to_vector(target_pi, board)

            game_history.append({
                "state": state,
                "pi": pi_vector,
                "turn": board.turn,
            })

            if stats_dict is not None:
                self._update_shared_stats(stats_dict, worker_id, board, move_count, search_profile)

            if move_count == 0:
                try:
                    if hasattr(board, "san"):
                        self.opening_stats[board.san(selected_move)] += 1
                    else:
                        self.opening_stats[getattr(selected_move, "uci", lambda: str(selected_move))()] += 1
                except Exception:
                    pass

            legal_moves = list(board.legal_moves)
            if selected_move not in legal_moves:
                if not legal_moves:
                    break
                selected_move = best_move if best_move in legal_moves else legal_moves[0]

            board.push(selected_move)
            move_count += 1

            # Reuse subtree after played move
            root = self.engine.advance_root(root, selected_move)

        outcome = forced_outcome if forced_outcome is not None else self._get_game_outcome(board, root=root, move_count=move_count)
        game_time_ms = (time.perf_counter() - game_start) * 1000.0

        final_data = []
        for entry in game_history:
            z = outcome if entry["turn"] == chess.WHITE else -outcome
            final_data.append({
                "state": entry["state"],
                "pi": entry["pi"],
                "z": float(z)
            })

        self.total_games += 1
        self.total_samples += len(final_data)
        self.total_positions_saved += len(final_data)
        self.total_game_time_ms += game_time_ms
        if outcome > 0:
            self.white_wins += 1
        elif outcome < 0:
            self.black_wins += 1
        else:
            self.draws += 1

        if stats_dict is not None:
            self._publish_game_totals(stats_dict)

        return final_data

    def _sample_from_pi(self, pi_dist, fallback_move, board, temperature=1.0):
        """
        Sample a move from the MCTS visit distribution.
        Falls back safely if the distribution is malformed.
        """
        adjusted = self._apply_temperature(pi_dist, temperature)
        if not adjusted:
            return fallback_move

        moves = list(adjusted.keys())
        probs = np.array([adjusted[m] for m in moves], dtype=np.float64)

        try:
            return np.random.choice(moves, p=probs)
        except Exception:
            legal_moves = list(board.legal_moves)
            if fallback_move in legal_moves:
                return fallback_move
            return legal_moves[0] if legal_moves else None

    def _apply_temperature(self, pi_dist, temperature=1.0):
        if not pi_dist:
            return {}

        moves = list(pi_dist.keys())
        probs = np.array([pi_dist[m] for m in moves], dtype=np.float64)
        probs = np.clip(probs, 1e-12, None)

        if temperature <= 1e-6:
            greedy = np.zeros_like(probs)
            greedy[int(np.argmax(probs))] = 1.0
            return {move: float(prob) for move, prob in zip(moves, greedy)}

        scaled = np.power(probs, 1.0 / float(temperature))
        total = scaled.sum()
        if total <= 0.0:
            uniform = 1.0 / len(moves)
            return {move: uniform for move in moves}

        scaled /= total
        return {move: float(prob) for move, prob in zip(moves, scaled)}

    def _uniform_pi_dist(self, legal_moves):
        if not legal_moves:
            return {}
        p = 1.0 / len(legal_moves)
        return {move: p for move in legal_moves}

    def _get_pi_dist(self, root):
        """
        Extract normalized visit probabilities from the MCTS root.
        Kept here for compatibility/debugging, though search() already returns pi_dist.
        """
        if root is None or not getattr(root, "moves", ()):
            return {}

        total = float(np.sum(root.visits))
        if total <= 0:
            if len(root.priors) == 0:
                return {}
            prob_sum = float(np.sum(root.priors))
            if prob_sum <= 0:
                return self._uniform_pi_dist(list(root.moves))
            return {
                move: float(prob / prob_sum)
                for move, prob in zip(root.moves, root.priors)
            }

        return {
            move: float(count / total)
            for move, count in zip(root.moves, root.visits)
            if count > 0.0
        }

    def _dist_to_vector(self, pi_dist, board):
        """
        Maps move distribution to the 4672-sized AlphaZero policy vector.
        """
        vec = np.zeros(4672, dtype=np.float32)
        for move, prob in pi_dist.items():
            idx = self.evaluator.move_to_idx(move, board.turn)
            vec[idx] = float(prob)
        return vec

    def _root_value(self, root):
        if root is None or getattr(root, "total_n", 0.0) <= 0.0:
            return 0.0
        return float(root.total_w / max(root.total_n, 1.0))

    def _should_resign(self, side_to_move, root_value, move_count, resign_streak):
        if move_count < self.RESIGN_AFTER_PLIES:
            resign_streak[side_to_move] = 0
            return False

        if root_value <= self.RESIGN_VALUE_THRESHOLD:
            resign_streak[side_to_move] += 1
        else:
            resign_streak[side_to_move] = 0

        return resign_streak[side_to_move] >= self.RESIGN_STREAK

    def _move_creates_claimable_draw(self, board, move):
        probe = board.copy(stack=False)
        probe.push(move)
        return probe.can_claim_threefold_repetition() or probe.can_claim_fifty_moves()

    def _avoid_draw_repetition(self, board, selected_move, pi_dist, best_move, move_count):
        if selected_move is None or move_count >= self.AVOID_DRAW_REPETITION_PLIES:
            return selected_move
        if not self._move_creates_claimable_draw(board, selected_move):
            return selected_move

        for move, _ in sorted(pi_dist.items(), key=lambda item: item[1], reverse=True)[:6]:
            if move == selected_move:
                continue
            if move not in board.legal_moves:
                continue
            if not self._move_creates_claimable_draw(board, move):
                return move

        return best_move if best_move in board.legal_moves else selected_move

    def _material_score(self, board):
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        score = 0.0
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score

    def _adjudicate_capped_game(self, board, root):
        root_value = self._root_value(root)
        white_value = root_value if board.turn == chess.WHITE else -root_value
        material_score = self._material_score(board)

        if white_value >= self.CAPTURED_GAME_VALUE_THRESHOLD and material_score >= self.CAPTURED_GAME_MATERIAL_THRESHOLD:
            return 1.0
        if white_value <= -self.CAPTURED_GAME_VALUE_THRESHOLD and material_score <= -self.CAPTURED_GAME_MATERIAL_THRESHOLD:
            return -1.0
        return 0.0

    def _get_game_outcome(self, board, root=None, move_count=0):
        """
        Final scalar outcome from White's perspective.
        """
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        if result == "*" and move_count >= self.MAX_GAME_LENGTH:
            return self._adjudicate_capped_game(board, root)
        return 0.0

    def _record_search_profile(self, profile):
        if not profile:
            return

        timings = profile.get("timings_ms", {})
        evaluator = profile.get("evaluator", {})

        for key in (
            "total",
            "root_eval",
            "selection",
            "leaf_eval",
            "backprop",
            "encode",
            "policy_mask",
            "inference_wait",
            "inference_forward",
        ):
            self.profile_totals[key] += float(timings.get(key, 0.0))

        self.profile_totals["simulations"] += float(profile.get("simulations_completed", 0))
        self.profile_totals["positions_evaluated"] += float(evaluator.get("positions_evaluated", 0))
        self.profile_totals["cache_hits"] += float(evaluator.get("cache_hits", 0))
        self.profile_totals["cache_misses"] += float(evaluator.get("cache_misses", 0))
        self.profile_samples += 1
        self.last_profile = profile

    def _average_profile(self):
        if self.profile_samples <= 0:
            return {}

        avg = {
            key: value / self.profile_samples
            for key, value in self.profile_totals.items()
        }
        cache_total = avg["cache_hits"] + avg["cache_misses"]
        avg["cache_hit_rate"] = (avg["cache_hits"] / cache_total) if cache_total > 0 else 0.0
        return avg

    def _update_shared_stats(self, stats_dict, worker_id, board, move_count, search_profile):
        avg_profile = self._average_profile()
        last_timings = search_profile.get("timings_ms", {}) if search_profile else {}
        last_evaluator = search_profile.get("evaluator", {}) if search_profile else {}

        stats_dict[worker_id] = {
            "move_count": move_count,
            "turn_number": board.fullmove_number,
            "fen": board.fen(),
            "turn": "White" if board.turn == chess.WHITE else "Black",
            "total_games": self.total_games,
            "total_samples": self.total_samples,
            "total_positions_saved": self.total_positions_saved,
            "legal_moves": board.legal_moves.count(),
            "is_game_over": board.is_game_over(claim_draw=True),
            "search": {
                "simulations_requested": int(search_profile.get("simulations_requested", 0)) if search_profile else 0,
                "simulations_completed": int(search_profile.get("simulations_completed", 0)) if search_profile else 0,
                "leaf_batches": int(search_profile.get("leaf_batches", 0)) if search_profile else 0,
                "terminal_leaves": int(search_profile.get("terminal_leaves", 0)) if search_profile else 0,
                "last_ms": {
                    "total": float(last_timings.get("total", 0.0)),
                    "selection": float(last_timings.get("selection", 0.0)),
                    "leaf_eval": float(last_timings.get("leaf_eval", 0.0)),
                    "backprop": float(last_timings.get("backprop", 0.0)),
                    "encode": float(last_timings.get("encode", 0.0)),
                    "policy_mask": float(last_timings.get("policy_mask", 0.0)),
                    "inference_wait": float(last_timings.get("inference_wait", 0.0)),
                    "inference_forward": float(last_timings.get("inference_forward", 0.0)),
                },
                "avg_ms": {
                    "total": float(avg_profile.get("total", 0.0)),
                    "root_eval": float(avg_profile.get("root_eval", 0.0)),
                    "selection": float(avg_profile.get("selection", 0.0)),
                    "leaf_eval": float(avg_profile.get("leaf_eval", 0.0)),
                    "backprop": float(avg_profile.get("backprop", 0.0)),
                    "encode": float(avg_profile.get("encode", 0.0)),
                    "policy_mask": float(avg_profile.get("policy_mask", 0.0)),
                    "inference_wait": float(avg_profile.get("inference_wait", 0.0)),
                    "inference_forward": float(avg_profile.get("inference_forward", 0.0)),
                },
                "avg_simulations": float(avg_profile.get("simulations", 0.0)),
                "avg_positions_evaluated": float(avg_profile.get("positions_evaluated", 0.0)),
                "avg_cache_hit_rate": float(avg_profile.get("cache_hit_rate", 0.0)),
                "profile_samples": int(self.profile_samples),
                "last_cache_hits": int(last_evaluator.get("cache_hits", 0)),
                "last_cache_misses": int(last_evaluator.get("cache_misses", 0)),
            },
        }

    def _publish_game_totals(self, stats_dict):
        cluster_entry = dict(stats_dict.get("__cluster__", {}))
        totals = dict(cluster_entry.get("totals", {}))
        totals[str(id(self))] = {
            "games": int(self.total_games),
            "positions": int(self.total_positions_saved),
            "game_time_ms": float(self.total_game_time_ms),
            "white_wins": int(self.white_wins),
            "draws": int(self.draws),
            "black_wins": int(self.black_wins),
        }
        cluster_entry["totals"] = totals
        stats_dict["__cluster__"] = cluster_entry

    def save_batch(self, game_data, filename):
        if not game_data:
            return

        payload = {
            "states": np.array([g["state"] for g in game_data], dtype=np.float32),
            "pis": np.array([g["pi"] for g in game_data], dtype=np.float32),
            "zs": np.array([g["z"] for g in game_data], dtype=np.float32),
            "encoder_version": np.int32(REPLAY_ENCODER_VERSION),
            "runtime_profile": np.array(PROFILE.name),
        }
        if self._low_disk_pressure():
            self._prune_oldest_replay_files(max_remove=96, target_free_bytes=runtime_low_disk_watermark_bytes())
        try:
            self._write_replay_payload(filename, payload)
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            self._prune_oldest_replay_files(max_remove=256, target_free_bytes=runtime_low_disk_watermark_bytes())
            self._write_replay_payload(filename, payload)

    def _write_replay_payload(self, filename, payload):
        if self.replay_shard_format == "raw":
            self._write_raw_replay_payload(filename, payload)
            return
        path = os.path.join(self.buffer_path, filename)
        saver = np.savez_compressed if PROFILE.replay_compress else np.savez
        saver(path, **payload)

    def _write_raw_replay_payload(self, filename, payload):
        paths = self._raw_shard_paths(filename)
        tmp_paths = {key: value + ".tmp" for key, value in paths.items() if key != "stem"}

        with open(tmp_paths["states"], "wb") as handle:
            np.save(handle, payload["states"], allow_pickle=False)
        with open(tmp_paths["pis"], "wb") as handle:
            np.save(handle, payload["pis"], allow_pickle=False)
        with open(tmp_paths["zs"], "wb") as handle:
            np.save(handle, payload["zs"], allow_pickle=False)
        with open(tmp_paths["meta"], "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_count": int(payload["zs"].shape[0]),
                    "state_shape": list(payload["states"].shape[1:]),
                    "encoder_version": int(payload["encoder_version"]),
                    "runtime_profile": PROFILE.name,
                    "format": "raw",
                },
                handle,
                separators=(",", ":"),
                sort_keys=True,
            )

        os.replace(tmp_paths["states"], paths["states"])
        os.replace(tmp_paths["pis"], paths["pis"])
        os.replace(tmp_paths["zs"], paths["zs"])
        os.replace(tmp_paths["meta"], paths["meta"])

    def _prune_oldest_replay_files(self, max_remove=24, target_free_bytes=0):
        replay_files = []
        for entry in os.scandir(self.buffer_path):
            if entry.is_file() and entry.name.endswith(".npz"):
                stat = entry.stat()
                replay_files.append((stat.st_mtime, "npz", entry.path, int(stat.st_size)))
            elif entry.is_file() and entry.name.endswith(".meta.json"):
                stem = entry.path[: -len(".meta.json")]
                shard_paths = [
                    stem + ".states.npy",
                    stem + ".pis.npy",
                    stem + ".zs.npy",
                    entry.path,
                ]
                if not all(os.path.exists(path) for path in shard_paths[:-1]):
                    continue
                size_bytes = 0
                for path in shard_paths:
                    try:
                        size_bytes += int(os.path.getsize(path))
                    except OSError:
                        size_bytes += 0
                replay_files.append((entry.stat().st_mtime, "raw", stem, size_bytes))

        replay_files.sort()
        removed = 0
        free_bytes = runtime_free_bytes()
        for _, shard_format, path, size_bytes in replay_files:
            if removed >= max_remove:
                break
            if target_free_bytes > 0 and free_bytes >= target_free_bytes:
                break
            try:
                if shard_format == "raw":
                    for suffix in (".states.npy", ".pis.npy", ".zs.npy", ".meta.json"):
                        os.remove(path + suffix)
                else:
                    os.remove(path)
                removed += 1
                free_bytes += size_bytes
            except OSError:
                continue
