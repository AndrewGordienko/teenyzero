import os
import numpy as np
import chess
from collections import Counter, deque


class DataCollector:
    def __init__(self, evaluator, engine, buffer_path="data/replay_buffer"):
        self.evaluator = evaluator
        self.engine = engine

        self.buffer_path = os.path.abspath(buffer_path)
        os.makedirs(self.buffer_path, exist_ok=True)

        # Exploration / diversity controls
        self.EXPLORATION_GAMES_THRESHOLD = 500
        self.FORCE_RANDOM_PLIES = 2
        self.MAX_GAME_LENGTH = 100

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

    def collect_game(self, worker_id=0, stats_dict=None):
        import time
        game_start = time.perf_counter()
        board = chess.Board()
        game_history = []
        move_count = 0
        root = None

        # Early training exploration
        random_prob = max(0.0, 1.0 - (self.total_games / self.EXPLORATION_GAMES_THRESHOLD))
        is_forced_exploration = np.random.random() < random_prob

        # Shorter stochastic window for faster, cleaner games
        temp_threshold = np.random.randint(6, 12)

        while not board.is_game_over(claim_draw=True) and move_count < self.MAX_GAME_LENGTH:
            best_move, pi_dist, root = self.engine.search(
                board,
                is_training=True,
                root=root,
            )
            search_profile = self.engine.last_search_stats
            self._record_search_profile(search_profile)

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
                selected_move = self._sample_from_pi(pi_dist, best_move, board)
            else:
                selected_move = best_move

            # Save training example BEFORE pushing the move
            state = self.evaluator.encode_board(board)
            pi_vector = self._dist_to_vector(pi_dist, board)

            game_history.append({
                "state": state,
                "pi": pi_vector,
                "turn": board.turn,
            })

            if stats_dict is not None:
                self._update_shared_stats(stats_dict, worker_id, board, move_count, search_profile)

            if move_count == 0:
                try:
                    self.opening_stats[board.san(selected_move)] += 1
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

        outcome = self._get_game_outcome(board)
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

    def _sample_from_pi(self, pi_dist, fallback_move, board):
        """
        Sample a move from the MCTS visit distribution.
        Falls back safely if the distribution is malformed.
        """
        if not pi_dist:
            return fallback_move

        moves = list(pi_dist.keys())
        probs = np.array([pi_dist[m] for m in moves], dtype=np.float64)

        total = probs.sum()
        if total <= 0.0:
            return fallback_move

        probs /= total

        try:
            return np.random.choice(moves, p=probs)
        except Exception:
            legal_moves = list(board.legal_moves)
            if fallback_move in legal_moves:
                return fallback_move
            return legal_moves[0] if legal_moves else None

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
        if root is None or not root.N:
            return {}

        total = sum(root.N.values())
        if total <= 0:
            if not root.P:
                return {}
            prob_sum = sum(root.P.values())
            if prob_sum <= 0:
                return self._uniform_pi_dist(list(root.P.keys()))
            return {move: p / prob_sum for move, p in root.P.items()}

        return {move: count / total for move, count in root.N.items()}

    def _dist_to_vector(self, pi_dist, board):
        """
        Maps move distribution to the 4672-sized AlphaZero policy vector.
        """
        vec = np.zeros(4672, dtype=np.float32)
        for move, prob in pi_dist.items():
            idx = self.evaluator.move_to_idx(move, board.turn)
            vec[idx] = float(prob)
        return vec

    def _get_game_outcome(self, board):
        """
        Final scalar outcome from White's perspective.
        """
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
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

        path = os.path.join(self.buffer_path, filename)
        np.savez_compressed(
            path,
            states=np.array([g["state"] for g in game_data], dtype=np.float32),
            pis=np.array([g["pi"] for g in game_data], dtype=np.float32),
            zs=np.array([g["z"] for g in game_data], dtype=np.float32),
        )
