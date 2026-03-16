import math
import chess
import threading
import time
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
        if root is None:
            root_eval_start = time.perf_counter()
            priors, _ = self.evaluator.evaluate(board)
            root = MCTSNode(priors)
            root_eval_ms = (time.perf_counter() - root_eval_start) * 1000.0

        if is_training:
            self._add_root_dirichlet_noise(root)

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

    def advance_root(self, root: MCTSNode, move):
        if root is None:
            return None
        return root.children.get(move)

    def _finalize_root(self, root: MCTSNode):
        total_visits = sum(root.N.values())

        if total_visits <= 0:
            if not root.P:
                return None, {}, root
            best_move = max(root.P, key=root.P.get)
            return best_move, dict(root.P), root

        pi_dist = {move: visits / total_visits for move, visits in root.N.items()}
        best_move = max(root.N, key=root.N.get)
        return best_move, pi_dist, root

    def _add_root_dirichlet_noise(self, root: MCTSNode):
        if not root.P:
            return

        if sum(root.N.values()) > 0:
            return

        import numpy as np

        moves = list(root.P.keys())
        alpha = self.params["ALPHA"]
        eps = self.params["EPS"]
        noise = np.random.dirichlet([alpha] * len(moves))

        for i, move in enumerate(moves):
            root.P[move] = (1.0 - eps) * root.P[move] + eps * float(noise[i])

    def _select_to_leaf(self, root_board: chess.Board, root: MCTSNode):
        """
        Runs selection only, stopping at a leaf.
        Returns a dict with path/board/node info for later batched evaluation.
        """
        node = root
        path = []
        board = root_board.copy(stack=False)

        while True:
            with self.tree_lock:
                move = self._select_child(node)
                if move is None:
                    return None

                path.append((node, move))

                if self.v_loss != 0.0:
                    node.N[move] += self.v_loss
                    node.W[move] -= self.v_loss

                child = node.children.get(move)

            board.push(move)

            if child is None:
                break

            node = child

        item = {
            "path": path,
            "board": board,
            "parent_node": node,
            "leaf_move": move,
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
        leaf_move = item["leaf_move"]
        value = item["value"]

        if not item["is_terminal"]:
            priors = item["priors"]
            with self.tree_lock:
                if leaf_move not in parent_node.children:
                    parent_node.children[leaf_move] = MCTSNode(priors)

        with self.tree_lock:
            for n, m in reversed(item["path"]):
                if self.v_loss != 0.0:
                    n.N[m] = n.N[m] - self.v_loss + 1
                    n.W[m] = n.W[m] + self.v_loss + value
                else:
                    n.N[m] += 1
                    n.W[m] += value
                value = -value

    def _select_child(self, node: MCTSNode):
        """
        PUCT + FPU.
        """
        if not node.P:
            return None

        total_n = sum(node.N.values())
        total_n_sqrt = math.sqrt(total_n + 1.0)

        parent_q = 0.0
        explored = 0
        for move in node.P:
            n = node.N[move]
            if n > 0:
                parent_q += node.W[move] / n
                explored += 1
        parent_q = (parent_q / explored) if explored > 0 else 0.0

        fpu_value = parent_q - self.params["FPU_REDUCTION"]

        best_score = -float("inf")
        best_move = None

        for move, prior in node.P.items():
            n = node.N[move]
            q = (node.W[move] / n) if n > 0 else fpu_value
            u = self.c_puct * prior * total_n_sqrt / (1.0 + n)
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

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
