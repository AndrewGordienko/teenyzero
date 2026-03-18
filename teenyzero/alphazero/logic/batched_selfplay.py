from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import chess
import numpy as np

from teenyzero.alphazero.backend import create_board
from teenyzero.alphazero.checkpoints import load_checkpoint
from teenyzero.alphazero.logic.collector import DataCollector


@dataclass
class GameSlot:
    slot_id: int
    board: object = field(default_factory=create_board)
    root: object = None
    move_count: int = 0
    game_history: list = field(default_factory=list)
    resign_streak: dict = field(
        default_factory=lambda: {
            chess.WHITE: 0,
            chess.BLACK: 0,
        }
    )
    forced_outcome: float | None = None
    is_forced_exploration: bool = False
    temp_threshold: int = 0
    started_at: float = field(default_factory=time.perf_counter)


class BatchedSelfPlayRunner:
    def __init__(
        self,
        evaluator,
        engine,
        buffer_path,
        concurrent_games=8,
        reload_model_path=None,
        reload_interval_s=2.0,
    ):
        self.evaluator = evaluator
        self.engine = engine
        self.helper = DataCollector(evaluator, engine, buffer_path=buffer_path)
        self.concurrent_games = max(1, int(concurrent_games))
        self.simulations = int(engine.params.get("SIMULATIONS", 0))
        self.leaf_batch_size = max(1, int(engine.params.get("LEAF_BATCH_SIZE", 1)))
        self.reload_model_path = reload_model_path
        self.reload_interval_s = max(0.1, float(reload_interval_s))
        self._last_reload_check = 0.0
        self._last_loaded_mtime = 0.0

    def run_forever(self, worker_id=0, stats_dict=None):
        slots = [self._new_game(slot_id) for slot_id in range(self.concurrent_games)]

        while True:
            self._maybe_reload_model(stats_dict=stats_dict)
            self._play_batch_move(slots, worker_id=worker_id, stats_dict=stats_dict)

            finished_payloads = []
            for idx, slot in enumerate(slots):
                if not self._is_finished(slot):
                    continue
                finished_payloads.append((idx, self._finalize_game(slot, stats_dict=stats_dict)))
                slots[idx] = self._new_game(slot.slot_id)

            timestamp = int(time.time() * 1000)
            for game_idx, payload in finished_payloads:
                filename = f"game_{worker_id}_{timestamp}_{game_idx}.npz"
                self.helper.save_batch(payload, filename)
                print(f"[Runner {worker_id}] Saved {len(payload)} positions to {filename}")

    def _maybe_reload_model(self, stats_dict=None):
        if self.reload_model_path is None or getattr(self.evaluator, "model", None) is None:
            return

        now = time.time()
        if (now - self._last_reload_check) < self.reload_interval_s:
            return
        self._last_reload_check = now

        try:
            model_mtime = float(os.path.getmtime(self.reload_model_path))
        except OSError:
            return

        if model_mtime <= self._last_loaded_mtime:
            return

        load_result = load_checkpoint(
            self.evaluator.model,
            self.reload_model_path,
            map_location="cpu",
            allow_partial=True,
        )
        if not load_result.get("loaded", False):
            return

        self._last_loaded_mtime = model_mtime
        self.evaluator.clear_cache()
        if stats_dict is not None:
            cluster_entry = dict(stats_dict.get("__cluster__", {}))
            config = dict(cluster_entry.get("config", {}))
            config["model_reloaded_at"] = now
            config["model_path"] = self.reload_model_path
            cluster_entry["config"] = config
            stats_dict["__cluster__"] = cluster_entry
        print(f"[Runner] Reloaded self-play weights from {self.reload_model_path}")

    def _new_game(self, slot_id):
        random_prob = max(0.0, 1.0 - (self.helper.total_games / self.helper.EXPLORATION_GAMES_THRESHOLD))
        return GameSlot(
            slot_id=slot_id,
            is_forced_exploration=bool(np.random.random() < random_prob),
            temp_threshold=int(
                np.random.randint(self.helper.TEMP_PLIES_MIN, self.helper.TEMP_PLIES_MAX + 1)
            ),
        )

    def _is_finished(self, slot):
        return (
            slot.forced_outcome is not None
            or slot.board.is_game_over(claim_draw=True)
            or slot.move_count >= self.helper.MAX_GAME_LENGTH
        )

    def _play_batch_move(self, slots, worker_id=0, stats_dict=None):
        active_slots = [slot for slot in slots if not self._is_finished(slot)]
        if not active_slots:
            return
        active_count = len(active_slots)

        self.evaluator.reset_profile()
        search_started = time.perf_counter()
        aggregate = {
            "root_eval": 0.0,
            "selection": 0.0,
            "leaf_eval": 0.0,
            "backprop": 0.0,
            "simulations_completed": 0,
            "leaf_batches": 0,
            "terminal_leaves": 0,
        }

        for slot in active_slots:
            slot.root, root_eval_ms = self.engine.prepare_root(slot.board, root=slot.root, is_training=True)
            aggregate["root_eval"] += root_eval_ms

        remaining = {slot.slot_id: self.simulations for slot in active_slots}
        while any(value > 0 for value in remaining.values()):
            pending_all = []
            pending_by_slot = []
            total_pending = 0

            for slot in active_slots:
                budget = min(self.leaf_batch_size, remaining[slot.slot_id])
                if budget <= 0:
                    continue
                pending, selection_ms, terminal_leaves = self.engine.collect_leaf_batch(slot.board, slot.root, budget)
                aggregate["selection"] += selection_ms
                aggregate["terminal_leaves"] += terminal_leaves
                aggregate["leaf_batches"] += 1 if pending else 0
                aggregate["simulations_completed"] += len(pending)
                remaining[slot.slot_id] -= len(pending)
                if pending:
                    pending_all.extend(pending)
                    pending_by_slot.append((slot, pending))
                    total_pending += len(pending)

            if not pending_all:
                break

            leaf_eval_ms = self.engine.evaluate_pending(pending_all)
            backprop_ms = self.engine.apply_pending(pending_all)
            if total_pending > 0:
                for _, pending in pending_by_slot:
                    share = len(pending) / float(total_pending)
                    aggregate["leaf_eval"] += leaf_eval_ms * share
                    aggregate["backprop"] += backprop_ms * share

        total_ms = (time.perf_counter() - search_started) * 1000.0
        evaluator_profile = self.evaluator.snapshot_profile()
        per_game_evaluator = {}
        for key, value in evaluator_profile.items():
            if isinstance(value, (int, float)):
                per_game_evaluator[key] = float(value) / float(active_count)
            else:
                per_game_evaluator[key] = value

        aggregate_profile = {
            "batch_active_games": int(active_count),
            "simulations_requested": int(round((self.simulations * active_count) / float(active_count))),
            "simulations_completed": int(round(aggregate["simulations_completed"] / float(active_count))),
            "leaf_batches": int(aggregate["leaf_batches"]),
            "terminal_leaves": int(aggregate["terminal_leaves"]),
            "timings_ms": {
                "total": float(total_ms) / float(active_count),
                "root_eval": float(aggregate["root_eval"]) / float(active_count),
                "selection": float(aggregate["selection"]) / float(active_count),
                "leaf_eval": float(aggregate["leaf_eval"]) / float(active_count),
                "backprop": float(aggregate["backprop"]) / float(active_count),
                "encode": float(self.evaluator.profile["encode_ms"]) / float(active_count),
                "policy_mask": float(self.evaluator.profile["mask_ms"]) / float(active_count),
                "inference_wait": float(self.evaluator.profile["inference_wait_ms"]) / float(active_count),
                "inference_forward": float(self.evaluator.profile["inference_forward_ms"]) / float(active_count),
            },
            "evaluator": per_game_evaluator,
        }
        self.helper._record_search_profile(aggregate_profile)

        for slot in active_slots:
            best_move, pi_dist, slot.root = self.engine.finalize_root(slot.root)
            temperature = (
                self.helper.OPENING_TEMPERATURE
                if slot.move_count < self.helper.FORCE_RANDOM_PLIES
                else self.helper.MIDGAME_TEMPERATURE
            )
            current_side = slot.board.turn
            root_value = self.helper._root_value(slot.root)

            if self.helper._should_resign(current_side, root_value, slot.move_count, slot.resign_streak):
                slot.forced_outcome = -1.0 if current_side == chess.WHITE else 1.0
                continue

            if best_move is None:
                legal_moves = list(slot.board.legal_moves)
                if not legal_moves:
                    continue
                best_move = np.random.choice(legal_moves)
                pi_dist = self.helper._uniform_pi_dist(legal_moves)

            if slot.is_forced_exploration and slot.move_count < self.helper.FORCE_RANDOM_PLIES:
                selected_move = np.random.choice(list(slot.board.legal_moves))
            elif slot.move_count < slot.temp_threshold:
                selected_move = self.helper._sample_from_pi(
                    pi_dist,
                    best_move,
                    slot.board,
                    temperature=temperature,
                )
            else:
                selected_move = best_move

            selected_move = self.helper._avoid_draw_repetition(
                slot.board,
                selected_move,
                pi_dist,
                best_move,
                slot.move_count,
            )

            state = np.array(self.evaluator._encode_cached(slot.board), copy=True)
            target_pi = (
                self.helper._apply_temperature(pi_dist, temperature)
                if slot.move_count < slot.temp_threshold
                else pi_dist
            )
            pi_vector = self.helper._dist_to_vector(target_pi, slot.board)
            slot.game_history.append(
                {
                    "state": state,
                    "pi": pi_vector,
                    "turn": slot.board.turn,
                }
            )

            if stats_dict is not None:
                self.helper._update_shared_stats(
                    stats_dict,
                    worker_id=f"{worker_id}:{slot.slot_id}",
                    board=slot.board,
                    move_count=slot.move_count,
                    search_profile=aggregate_profile,
                )

            legal_moves = list(slot.board.legal_moves)
            if selected_move not in legal_moves:
                if not legal_moves:
                    continue
                selected_move = best_move if best_move in legal_moves else legal_moves[0]

            slot.board.push(selected_move)
            slot.move_count += 1
            slot.root = self.engine.advance_root(slot.root, selected_move)

    def _finalize_game(self, slot, stats_dict=None):
        outcome = (
            slot.forced_outcome
            if slot.forced_outcome is not None
            else self.helper._get_game_outcome(slot.board, root=slot.root, move_count=slot.move_count)
        )

        final_data = []
        for entry in slot.game_history:
            z = outcome if entry["turn"] == chess.WHITE else -outcome
            final_data.append(
                {
                    "state": entry["state"],
                    "pi": entry["pi"],
                    "z": float(z),
                }
            )

        self.helper.total_games += 1
        self.helper.total_samples += len(final_data)
        self.helper.total_positions_saved += len(final_data)
        self.helper.total_game_time_ms += (time.perf_counter() - slot.started_at) * 1000.0
        if outcome > 0:
            self.helper.white_wins += 1
        elif outcome < 0:
            self.helper.black_wins += 1
        else:
            self.helper.draws += 1

        if stats_dict is not None:
            self.helper._publish_game_totals(stats_dict)

        return final_data
