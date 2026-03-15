import torch
import numpy as np
import time

def inference_worker(model_path, device, task_queue, result_dict):
    """
    The GPU Master process. 
    Optimized for dynamic batching to maximize throughput across multiple workers.
    """
    from teenyzero.alphazero.model import AlphaNet
    
    print(f"[Inference] Initializing AlphaNet on {device}...")
    model = AlphaNet(num_res_blocks=10, channels=128).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Inference] Weights loaded successfully from {model_path}")
    except Exception as e:
        print(f"[Inference] Warning: Starting with fresh weights ({e})")
    
    model.eval()

    BATCH_SIZE = 64
    WAIT_TIMEOUT = 0.001 # 1ms window to wait for batch to fill
    
    while True:
        batch = []
        ids = []
        
        # 1. Block for at least one task
        try:
            task_id, state = task_queue.get(timeout=1.0)
            batch.append(state)
            ids.append(task_id)
        except:
            continue

        # 2. Dynamic Batching
        start_wait = time.time()
        while len(batch) < BATCH_SIZE:
            try:
                task_id, state = task_queue.get_nowait()
                batch.append(state)
                ids.append(task_id)
            except:
                if time.time() - start_wait < WAIT_TIMEOUT:
                    time.sleep(0.0001) 
                    continue
                else:
                    break 

        # 3. Batch Inference
        if batch:
            with torch.no_grad():
                tensors = torch.from_numpy(np.stack(batch)).to(device)
                logits, values = model(tensors)
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                vals = values.cpu().numpy().flatten()
                
                # Use a local dict to batch the shared dictionary updates
                updates = {tid: (probs[i], float(vals[i])) for i, tid in enumerate(ids)}
                result_dict.update(updates)