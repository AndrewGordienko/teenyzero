from __future__ import annotations

from pathlib import Path

import chess

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.search_session import SearchSession
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS


OPENING_BOOK = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
]


def _opening_for_game(index: int) -> list[str]:
    return OPENING_BOOK[index % len(OPENING_BOOK)]


def _apply_opening(board: chess.Board, opening_line: list[str]) -> None:
    for uci in opening_line:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)


def _result_score(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


class ArenaAgent:
    def __init__(self, model_path: Path, device: str, simulations: int):
        self.model = build_model()
        load_checkpoint(self.model, model_path, map_location=device, allow_partial=True)
        self.model.eval()
        evaluator = AlphaZeroEvaluator(model=self.model, device=device, use_cache=True)
        self.engine = MCTS(
            evaluator=evaluator,
            params={
                "SIMULATIONS": max(1, int(simulations)),
                "C_PUCT": 1.35,
                "FPU_REDUCTION": 0.35,
                "LEAF_BATCH_SIZE": 8,
            },
        )
        self.session = SearchSession(self.engine)

    def choose_move(self, board: chess.Board):
        move, _, _ = self.session.search(board, is_training=False)
        return move

    def close(self) -> None:
        self.session.reset()


def _play_game(white_agent: ArenaAgent, black_agent: ArenaAgent, opening_line: list[str], max_plies: int = 160):
    board = chess.Board()
    _apply_opening(board, opening_line)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent.choose_move(board)
        if move is None or move not in board.legal_moves:
            return ("0-1" if board.turn == chess.WHITE else "1-0"), board, plies
        board.push(move)
        plies += 1
    return board.result(claim_draw=True), board, plies


def play_phase3_match(
    base_model_path: Path,
    candidate_model_path: Path,
    *,
    device: str,
    simulations: int,
    games: int,
) -> dict:
    base_agent = ArenaAgent(base_model_path, device=device, simulations=simulations)
    candidate_agent = ArenaAgent(candidate_model_path, device=device, simulations=simulations)
    game_rows = []
    try:
        for game_index in range(max(1, int(games))):
            opening = _opening_for_game(game_index)
            if game_index % 2 == 0:
                result, board, plies = _play_game(candidate_agent, base_agent, opening)
                score_candidate = _result_score(result)
                color = "white"
            else:
                result, board, plies = _play_game(base_agent, candidate_agent, opening)
                score_candidate = 1.0 - _result_score(result)
                color = "black"
            game_rows.append(
                {
                    "color": color,
                    "result": result,
                    "score_candidate": score_candidate,
                    "plies": plies,
                    "final_fen": board.fen(),
                    "opening": opening,
                }
            )
    finally:
        candidate_agent.close()
        base_agent.close()

    wins = sum(1 for row in game_rows if row["score_candidate"] == 1.0)
    draws = sum(1 for row in game_rows if row["score_candidate"] == 0.5)
    losses = sum(1 for row in game_rows if row["score_candidate"] == 0.0)
    score = (wins + 0.5 * draws) / max(len(game_rows), 1)
    return {
        "games": len(game_rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "details": game_rows,
    }
