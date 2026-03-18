import math
import chess
import threading
import time
import numpy as np
from .node import MCTSNode


class MCTS:
    def __init__(self, evaluator, params=None):
        self.evaluator = evaluator
        self.params = {
            "SIMULATIONS": 64,
            "C_PUCT": 2.0,
            "ALPHA": 0.8,
            "EPS": 0.25,
            "FPU_REDUCTION": 0.2,
            "VIRTUAL_LOSS": 0.0,
            "PARALLEL_THREADS": 1,
            "LEAF_BATCH_SIZE": 8,
        }
        if params:
            self.params.update(params)

        self.tree_lock = threading.Lock()
        self.use_tree_lock = int(self.params.get("PARALLEL_THREADS", 1)) > 1
        self.c_puct = self.params["C_PUCT"]
        self.v_loss = self.params["VIRTUAL_LOSS"]
        self.leaf_batch_size = max(1, int(self.params["LEAF_BATCH_SIZE"]))
        self.last_search_stats = self._empty_search_stats()

    def search(self, board: chess.Board, is_training=False, num_simulations=None, root=None):
        """
        Runs MCTS from the given board and returns:
            best_move, pi_dist, root
        """
        self.evaluator.reset_profile()
        search_start = time.perf_counter()
        root_eval_ms = 0.0
        root, root_eval_ms = self.prepare_root(board, root=root, is_training=is_training)

        sim_count = int(num_simulations) if num_simulations is not None else int(self.params["SIMULATIONS"])
        if sim_count <= 0:
            self.last_search_stats = self._build_search_stats(
                sim_count=sim_count,
                sims_done=0,
                batch_count=0,
                terminal_leaves=0,
                root_eval_ms=root_eval_ms,
                selection_ms=0.0,
                leaf_eval_ms=0.0,
                backprop_ms=0.0,
                total_ms=(time.perf_counter() - search_start) * 1000.0,
            )
            return self._finalize_root(root)

        sims_done = 0
        batch_count = 0
        terminal_leaves = 0
        selection_ms = 0.0
        leaf_eval_ms = 0.0
        backprop_ms = 0.0
        while sims_done < sim_count:
            current_batch_size = min(self.leaf_batch_size, sim_count - sims_done)
            pending = []

            # 1. Collect a batch of leaves
            selection_start = time.perf_counter()
            for _ in range(current_batch_size):
                leaf = self._select_to_leaf(board, root)
                if leaf is not None:
                    pending.append(leaf)
            selection_ms += (time.perf_counter() - selection_start) * 1000.0

            if not pending:
                break

            # 2. Evaluate leaves in batch
            batch_count += 1
            terminal_leaves += sum(1 for item in pending if item["is_terminal"])
            leaf_eval_start = time.perf_counter()
            self._evaluate_leaf_batch(pending)
            leaf_eval_ms += (time.perf_counter() - leaf_eval_start) * 1000.0

            # 3. Expand + backprop
            backprop_start = time.perf_counter()
            for item in pending:
                self._apply_leaf_result(item)
            backprop_ms += (time.perf_counter() - backprop_start) * 1000.0

            sims_done += len(pending)

        self.last_search_stats = self._build_search_stats(
            sim_count=sim_count,
            sims_done=sims_done,
            batch_count=batch_count,
            terminal_leaves=terminal_leaves,
            root_eval_ms=root_eval_ms,
            selection_ms=selection_ms,
            leaf_eval_ms=leaf_eval_ms,
            backprop_ms=backprop_ms,
            total_ms=(time.perf_counter() - search_start) * 1000.0,
        )
        return self._finalize_root(root)

    def prepare_root(self, board: chess.Board, root=None, is_training=False):
        root_eval_ms = 0.0
        if root is None:
            root_eval_start = time.perf_counter()
            priors, _ = self.evaluator.evaluate(board)
            root = MCTSNode(priors)
            root_eval_ms = (time.perf_counter() - root_eval_start) * 1000.0

        if is_training:
            self._add_root_dirichlet_noise(root)
        return root, root_eval_ms

    def collect_leaf_batch(self, board: chess.Board, root: MCTSNode, count: int):
        pending = []
        selection_start = time.perf_counter()
        for _ in range(max(0, int(count))):
            leaf = self._select_to_leaf(board, root)
            if leaf is not None:
                pending.append(leaf)
        selection_ms = (time.perf_counter() - selection_start) * 1000.0
        terminal_leaves = sum(1 for item in pending if item["is_terminal"])
        return pending, selection_ms, terminal_leaves

    def evaluate_pending(self, pending):
        leaf_eval_start = time.perf_counter()
        self._evaluate_leaf_batch(pending)
        return (time.perf_counter() - leaf_eval_start) * 1000.0

    def apply_pending(self, pending):
        backprop_start = time.perf_counter()
        for item in pending:
            self._apply_leaf_result(item)
        return (time.perf_counter() - backprop_start) * 1000.0

    def finalize_root(self, root: MCTSNode):
        return self._finalize_root(root)

    def advance_root(self, root: MCTSNode, move):
        if root is None:
            return None
        return root.get_child(move)

    def _finalize_root(self, root: MCTSNode):
        total_visits = float(root.total_n)

        if total_visits <= 0:
            if not root.moves:
                return None, {}, root
            best_idx = int(np.argmax(root.priors))
            best_move = root.moves[best_idx]
            return best_move, {
                move: float(prob)
                for move, prob in zip(root.moves, root.priors)
            }, root

        pi_dist = {
            move: float(visits / total_visits)
            for move, visits in zip(root.moves, root.visits)
            if visits > 0.0
        }
        best_idx = int(np.argmax(root.visits))
        best_move = root.moves[best_idx]
        return best_move, pi_dist, root

    def _add_root_dirichlet_noise(self, root: MCTSNode):
        if not root.moves:
            return

        if getattr(root, "dirichlet_applied", False):
            return

        import numpy as np

        alpha = self.params["ALPHA"]
        eps = self.params["EPS"]
        noise = np.random.dirichlet([alpha] * len(root.moves)).astype(np.float32)
        root.priors = ((1.0 - eps) * root.priors + eps * noise).astype(np.float32, copy=False)

        root.dirichlet_applied = True

    def _select_to_leaf(self, root_board: chess.Board, root: MCTSNode):
        """
        Runs selection only, stopping at a leaf.
        Returns a dict with path/board/node info for later batched evaluation.
        """
        node = root
        path = []
        history_depth = max(0, int(getattr(self.evaluator, "history_length", 1)) - 1)
        board = root_board.copy(stack=history_depth if getattr(self.evaluator, "requires_history", False) else False)

        while True:
            if self.use_tree_lock:
                with self.tree_lock:
                    move_idx = self._select_child(node)
                    if move_idx is None:
                        return None

                    path.append((node, move_idx))

                    if self.v_loss != 0.0:
                        node.visits[move_idx] += self.v_loss
                        node.value_sums[move_idx] -= self.v_loss
                        node.total_n += self.v_loss
                        node.total_w -= self.v_loss

                    child = node.children[move_idx]
            else:
                move_idx = self._select_child(node)
                if move_idx is None:
                    return None

                path.append((node, move_idx))

                if self.v_loss != 0.0:
                    node.visits[move_idx] += self.v_loss
                    node.value_sums[move_idx] -= self.v_loss
                    node.total_n += self.v_loss
                    node.total_w -= self.v_loss

                child = node.children[move_idx]

            move = node.moves[move_idx]
            board.push(move)

            if child is None:
                break

            node = child

        item = {
            "path": path,
            "board": board,
            "parent_node": node,
            "leaf_index": move_idx,
            "is_terminal": board.is_game_over(claim_draw=True),
            "priors": None,
            "value": None,
        }
        return item

    def _evaluate_leaf_batch(self, pending):
        """
        Batched leaf evaluation.

        If evaluator has evaluate_many(), uses it.
        Otherwise falls back to repeated evaluate().
        """
        non_terminal_items = []
        non_terminal_boards = []

        for item in pending:
            if item["is_terminal"]:
                item["value"] = self._terminal_value(item["board"])
            else:
                non_terminal_items.append(item)
                non_terminal_boards.append(item["board"])

        if not non_terminal_items:
            return

        if hasattr(self.evaluator, "evaluate_many"):
            results = self.evaluator.evaluate_many(non_terminal_boards)
        else:
            results = [self.evaluator.evaluate(b) for b in non_terminal_boards]

        for item, (priors, value) in zip(non_terminal_items, results):
            item["priors"] = priors
            item["value"] = value

    def _apply_leaf_result(self, item):
        """
        Expands the leaf and backprops the value.
        """
        parent_node = item["parent_node"]
        leaf_index = item["leaf_index"]
        value = item["value"]

        if not item["is_terminal"]:
            priors = item["priors"]
            if self.use_tree_lock:
                with self.tree_lock:
                    if parent_node.children[leaf_index] is None:
                        parent_node.children[leaf_index] = MCTSNode(priors)
            else:
                if parent_node.children[leaf_index] is None:
                    parent_node.children[leaf_index] = MCTSNode(priors)

        if self.use_tree_lock:
            with self.tree_lock:
                for n, idx in reversed(item["path"]):
                    if self.v_loss != 0.0:
                        n.visits[idx] = n.visits[idx] - self.v_loss + 1
                        n.value_sums[idx] = n.value_sums[idx] + self.v_loss + value
                        n.total_n = n.total_n - self.v_loss + 1
                        n.total_w = n.total_w + self.v_loss + value
                    else:
                        n.visits[idx] += 1
                        n.value_sums[idx] += value
                        n.total_n += 1
                        n.total_w += value
                    value = -value
        else:
            for n, idx in reversed(item["path"]):
                if self.v_loss != 0.0:
                    n.visits[idx] = n.visits[idx] - self.v_loss + 1
                    n.value_sums[idx] = n.value_sums[idx] + self.v_loss + value
                    n.total_n = n.total_n - self.v_loss + 1
                    n.total_w = n.total_w + self.v_loss + value
                else:
                    n.visits[idx] += 1
                    n.value_sums[idx] += value
                    n.total_n += 1
                    n.total_w += value
                value = -value

    def _select_child(self, node: MCTSNode):
        """
        PUCT + FPU.
        """
        if not node.moves:
            return None

        total_n = node.total_n
        total_n_sqrt = math.sqrt(total_n + 1.0)
        # Weighted parent mean is cheaper to maintain than re-averaging explored edges.
        parent_q = (node.total_w / total_n) if total_n > 0 else 0.0

        fpu_value = parent_q - self.params["FPU_REDUCTION"]

        best_score = -float("inf")
        best_idx = None
        priors = node.priors
        visits = node.visits
        value_sums = node.value_sums

        for idx in range(len(priors)):
            n = float(visits[idx])
            q = (float(value_sums[idx]) / n) if n > 0 else fpu_value
            u = self.c_puct * float(priors[idx]) * total_n_sqrt / (1.0 + n)
            score = q + u

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _terminal_value(self, board: chess.Board):
        """
        Returns value from the perspective of the side to move at this terminal board.
        """
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _build_search_stats(
        self,
        sim_count,
        sims_done,
        batch_count,
        terminal_leaves,
        root_eval_ms,
        selection_ms,
        leaf_eval_ms,
        backprop_ms,
        total_ms,
    ):
        evaluator_profile = self.evaluator.snapshot_profile()
        return {
            "simulations_requested": int(sim_count),
            "simulations_completed": int(sims_done),
            "leaf_batches": int(batch_count),
            "terminal_leaves": int(terminal_leaves),
            "timings_ms": {
                "total": float(total_ms),
                "root_eval": float(root_eval_ms),
                "selection": float(selection_ms),
                "leaf_eval": float(leaf_eval_ms),
                "backprop": float(backprop_ms),
                "encode": float(evaluator_profile["encode_ms"]),
                "policy_mask": float(evaluator_profile["mask_ms"]),
                "inference_wait": float(evaluator_profile["inference_wait_ms"]),
                "inference_forward": float(evaluator_profile["inference_forward_ms"]),
            },
            "evaluator": evaluator_profile,
        }

    def _empty_search_stats(self):
        return {
            "simulations_requested": 0,
            "simulations_completed": 0,
            "leaf_batches": 0,
            "terminal_leaves": 0,
            "timings_ms": {
                "total": 0.0,
                "root_eval": 0.0,
                "selection": 0.0,
                "leaf_eval": 0.0,
                "backprop": 0.0,
                "encode": 0.0,
                "policy_mask": 0.0,
                "inference_wait": 0.0,
                "inference_forward": 0.0,
            },
            "evaluator": self.evaluator.snapshot_profile() if hasattr(self.evaluator, "snapshot_profile") else {},
        }
