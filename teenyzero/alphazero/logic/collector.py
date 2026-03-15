import os
import time
import numpy as np
import chess
import torch
from collections import Counter, deque

class DataCollector:
    def __init__(self, evaluator, engine, buffer_path="data/replay_buffer"):
        self.evaluator = evaluator
        self.engine = engine
        
        # Ensure absolute path for data storage
        self.buffer_path = os.path.abspath(buffer_path)
        os.makedirs(self.buffer_path, exist_ok=True)
        
        # --- Exploration & Diversity Stats ---
        self.EXPLORATION_GAMES_THRESHOLD = 500  
        self.FORCE_RANDOM_PLIES = 2             
        self.total_games = 0
        self.total_samples = 0
        
        # Track opening diversity
        self.opening_stats = Counter() 
        # For the dashboard "interesting games" feed
        self.hall_of_fame = deque(maxlen=8) 

    def collect_game(self, worker_id=0, stats_dict=None):
        board = chess.Board()
        game_history = []  # Stores (state, pi, turn)
        move_count = 0
        
        # Temperature/Exploration Strategy
        # Early in training, we force more randomness to see the whole board
        random_prob = max(0.0, 1.0 - (self.total_games / self.EXPLORATION_GAMES_THRESHOLD))
        is_forced_exploration = np.random.random() < random_prob
        
        # Use a random threshold for when to switch from "Stochastic" to "Deterministic" play
        # This helps the engine learn both exploration and "closing out" a game
        temp_threshold = np.random.randint(15, 30)

        while not board.is_game_over() and move_count < 250:
            # 1. Search
            best_move, root = self.engine.search(board)
            
            # 2. Extract Policy (Pi) from MCTS visit counts
            # We must map this to our 4672 move index
            pi_dist = self._get_pi_dist(root, board)
            
            # 3. Selection Logic (Stochastic vs Deterministic)
            if is_forced_exploration and move_count < self.FORCE_RANDOM_PLIES:
                selected_move = np.random.choice(list(board.legal_moves))
            elif move_count < temp_threshold:
                # Stochastic: Pick based on MCTS visit distribution
                moves = list(pi_dist.keys())
                probs = list(pi_dist.values())
                selected_move = np.random.choice(moves, p=probs)
            else:
                # Deterministic: Pick the absolute best move
                selected_move = max(pi_dist, key=pi_dist.get)

            # 4. Save State for Training
            # We encode the board BEFORE pushing the move
            state = self.evaluator.encode_board(board)
            pi_vector = self._dist_to_vector(pi_dist, board)
            game_history.append({
                "state": state,
                "pi": pi_vector,
                "turn": board.turn
            })

            # 5. Live Dashboard Updates
            if stats_dict is not None:
                self._update_shared_stats(stats_dict, worker_id, board, move_count)

            # 6. Execute
            if move_count == 0:
                self.opening_stats[board.san(selected_move)] += 1
                
            board.push(selected_move)
            move_count += 1

        # 7. Finalize Outcome (Z)
        # 1.0 for White win, -1.0 for Black win, 0.0 for Draw
        result = board.result()
        outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
        
        # Map the outcome to each state based on whose turn it was
        final_data = []
        for entry in game_history:
            # If it was White's turn, z = outcome
            # If it was Black's turn, z = -outcome
            z = outcome if entry["turn"] == chess.WHITE else -outcome
            final_data.append({
                "state": entry["state"],
                "pi": entry["pi"],
                "z": float(z)
            })

        self.total_games += 1
        self.total_samples += len(final_data)
        return final_data

    def _get_pi_dist(self, root, board):
        """Extracts normalized visit probabilities from the MCTS root."""
        counts = root.N
        total = np.sum(counts)
        return {move: count/total for move, count in zip(root.actions, counts)}

    def _dist_to_vector(self, pi_dist, board):
        """Maps chess moves to the 4672-sized policy vector."""
        vec = np.zeros(4672, dtype=np.float32)
        for move, prob in pi_dist.items():
            idx = self.evaluator.move_to_idx(move, board.turn)
            vec[idx] = prob
        return vec

    def _update_shared_stats(self, stats_dict, worker_id, board, move_count):
        """Pushes data to the multiprocessing manager for the dashboard."""
        stats_dict[worker_id] = {
            "move_count": move_count,
            "fen": board.fen(),
            "turn": "White" if board.turn == chess.WHITE else "Black",
            "total_games": self.total_games,
            "total_samples": self.total_samples
        }

    def save_batch(self, game_data, filename):
        """Saves game data to compressed NPZ format."""
        path = os.path.join(self.buffer_path, filename)
        np.savez_compressed(
            path,
            states=np.array([g["state"] for g in game_data]),
            pis=np.array([g["pi"] for g in game_data]),
            zs=np.array([g["z"] for g in game_data])
        )