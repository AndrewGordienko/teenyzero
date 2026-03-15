import os
import time
import torch
import multiprocessing as mp
import argparse
import sys

# Package imports
from teenyzero.alphazero.model import AlphaNet
from teenyzero.alphazero.servers.inference import inference_worker
from teenyzero.alphazero.logic.collector import DataCollector
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS

# Import the dashboard from your new folder structure
# teenyzero/visualizers/training_data_dashboard/dashboard.py
from teenyzero.visualizers.training_data_dashboard.dashboard import run_dashboard

def bootstrap_model(path):
    """Ensures a model exists so workers don't crash."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[*] Initializing fresh AlphaNet at {path}...")
        model = AlphaNet(num_res_blocks=10, channels=128)
        torch.save(model.state_dict(), path)

def worker_task(worker_id, task_queue, result_dict, shared_stats):
    """The actual worker process loop."""
    # Workers use BATCH mode to talk to the Inference Server
    evaluator = AlphaZeroEvaluator(task_queue=task_queue, result_dict=result_dict)
    
    # Standard MCTS search parameters
    mcts_params = {
        'SIMULATIONS': 400, 
        'C_PUCT': 1.2,
        'VIRTUAL_LOSS': 1.0,
        'PARALLEL_THREADS': 1, 
        'FPU_REDUCTION': 0.4
    }
    engine = MCTS(evaluator=evaluator, params=mcts_params)
    
    # The Collector manages game loops and saving to disk
    collector = DataCollector(evaluator, engine, buffer_path="teenyzero/alphazero/data/replay_buffer")
    
    print(f"[Worker {worker_id}] Online and searching...")
    
    while True:
        # 1. Play a full game
        game_data = collector.collect_game(worker_id=worker_id, stats_dict=shared_stats)
        
        # 2. Save the game data as an NPZ file
        timestamp = int(time.time() * 1000)
        filename = f"game_{worker_id}_{timestamp}.npz"
        collector.save_batch(game_data, filename)
        
        # 3. Log progress to terminal
        print(f"[Worker {worker_id}] Saved {len(game_data)} positions to {filename}")

if __name__ == "__main__":
    # Essential for MacOS/MPS and Multiprocessing consistency
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of self-play processes")
    args = parser.parse_args()

    # Path Setup
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    bootstrap_model(MODEL_PATH)
    
    with mp.Manager() as manager:
        # Shared memory for Inter-Process Communication (IPC)
        result_dict = manager.dict() 
        shared_stats = manager.dict()
        task_queue = mp.Queue(maxsize=2048) 
        processes = []

        # 1. Start the Inference Server (The GPU Master)
        inf_p = mp.Process(target=inference_worker, args=(MODEL_PATH, DEVICE, task_queue, result_dict))
        inf_p.daemon = True
        inf_p.start()
        processes.append(inf_p)

        # 2. Start Training Dashboard (Port 5002)
        # It reads from the shared_stats to show live worker activity
        dash_p = mp.Process(target=run_dashboard, args=(shared_stats,))
        dash_p.daemon = True
        dash_p.start()
        processes.append(dash_p)

        # 3. Start Self-Play Workers (The Actors)
        for i in range(args.workers):
            p = mp.Process(target=worker_task, args=(i, task_queue, result_dict, shared_stats))
            p.daemon = True
            p.start()
            processes.append(p)

        print(f"\n" + "="*50)
        print(f"  TEENYZERO FACTORY ONLINE")
        print(f"="*50)
        print(f"[*] Workers:   {args.workers}")
        print(f"[*] Device:    {DEVICE}")
        print(f"[*] Dashboard: http://localhost:5002")
        print(f"[*] Data:      teenyzero/alphazero/data/replay_buffer/")
        print(f"Press Ctrl+C to stop cluster.")
        print(f"="*50 + "\n")

        try:
            # Keep the main process alive
            for p in processes: 
                p.join()
        except KeyboardInterrupt:
            print("\n[!] Shutting down clusters...")
            for p in processes: 
                p.terminate()