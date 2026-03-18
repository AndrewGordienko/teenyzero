import argparse
import os

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

import chess

from teenyzero.alphazero.backend import create_board
from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.alphazero.search_session import SearchSession
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.paths import LATEST_MODEL_PATH, ensure_runtime_dirs


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile

def maybe_reload_model(model, evaluator, model_path, last_mtime, session=None):
    if not os.path.exists(model_path):
        return last_mtime

    current_mtime = os.path.getmtime(model_path)
    if last_mtime is not None and current_mtime <= last_mtime:
        return last_mtime

    load_result = load_checkpoint(model, model_path, map_location=evaluator.device, allow_partial=True)
    if load_result["loaded"]:
        model.eval()
        evaluator.clear_cache()
        if session is not None:
            session.reset()
    return current_mtime


def run_self_play(session, model, evaluator, model_path):
    """The terminal-based self-play loop."""
    board = create_board()
    last_mtime = None
    session.reset()
    print("\n" + "="*30)
    print("[*] Starting Self-Play Mode")
    print("="*30 + "\n")
    print(board)

    while not board.is_game_over():
        last_mtime = maybe_reload_model(model, evaluator, model_path, last_mtime, session=session)
        current_turn = "White" if board.turn == chess.WHITE else "Black"
        best_move, _, root_node = session.search(board)
        expected_value = (root_node.total_w / root_node.total_n) if root_node and root_node.total_n > 0 else 0.0
        
        print(f"\n[{current_turn}] Move: {best_move} | Value: {expected_value:.3f}")
        board.push(best_move)
        print(board)
        print("-" * 20)

    print(f"\n[*] Game Over. Result: {board.result()}")

def main():
    parser = argparse.ArgumentParser(description="teenyzero: AlphaZero Research Project")
    parser.add_argument("--mode", choices=["play", "visualize"], default="visualize", 
                        help="Choose between terminal self-play or the web dashboard.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Override the active profile simulation count for interactive play.",
    )
    args = parser.parse_args()

    device = RUNTIME.device
    print(f"[*] Initializing teenyzero on: {device}")
    print(f"[*] Active runtime profile: {PROFILE.name}")
    ensure_runtime_dirs()
    model_path = str(LATEST_MODEL_PATH)

    model = build_model()
    if os.path.exists(model_path):
        load_checkpoint(model, model_path, map_location=device, allow_partial=True)
    
    evaluator = AlphaZeroEvaluator(model=model, device=device)
    simulations = args.simulations if args.simulations is not None else max(160, PROFILE.arena_simulations)
    
    mcts_params = {
        'SIMULATIONS': simulations,
        'C_PUCT': 1.1,
        'VIRTUAL_LOSS': 0.0,
        'PARALLEL_THREADS': 1,
        'FPU_REDUCTION': 0.5,
        'LEAF_BATCH_SIZE': max(8, PROFILE.selfplay_leaf_batch_size // 2),
    }
    engine = MCTS(evaluator=evaluator, params=mcts_params)
    session = SearchSession(engine)

    if args.mode == "play":
        run_self_play(session, model, evaluator, model_path)
    
    elif args.mode == "visualize":
        print("[*] Injecting engine into Visualizer...")
        import teenyzero.visualizers.app as vis_app
        vis_app.model = model
        vis_app.evaluator = evaluator
        vis_app.engine = engine 
        vis_app.search_session = SearchSession(engine)
        
        print("[*] Launching Lab UI on http://localhost:5001")
        vis_app.app.run(debug=False, port=5001, host='0.0.0.0')

if __name__ == "__main__":
    main()
