import time
import queue
import os
import torch
import numpy as np


def inference_worker(model_path, device, task_queue, response_queues, shared_stats=None):
    """
    Dynamic batched inference worker.

    Request format from workers:
        (task_id, encoded_state_or_batch, worker_id, is_batch)

    Single response format:
        (task_id, logits, value, False)

    Batched response format:
        (task_id, logits_batch, values_batch, True)
    """
    from teenyzero.alphazero.model import AlphaNet

    print(f"[Inference] Initializing AlphaNet on {device}...")

    model = AlphaNet(num_res_blocks=10, channels=128)

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[Inference] Weights loaded successfully from {model_path}")
    except Exception as e:
        print(f"[Inference] Warning: Starting with fresh weights ({e})")

    try:
        model_mtime = os.path.getmtime(model_path)
    except OSError:
        model_mtime = None

    model = model.to(device)
    model.eval()

    BATCH_SIZE = 64
    WAIT_TIMEOUT = 0.0001
    IDLE_GET_TIMEOUT = 1.0
    last_stats_push = 0.0
    cluster_stats = {
        "device": device,
        "total_requests": 0,
        "total_positions": 0,
        "single_requests": 0,
        "explicit_batch_requests": 0,
        "dynamic_batches": 0,
        "merged_explicit_batches": 0,
        "server_forwards": 0,
        "avg_dynamic_batch_size": 0.0,
        "avg_explicit_batch_size": 0.0,
        "avg_merged_batch_size": 0.0,
        "avg_gather_wait_ms": 0.0,
        "avg_forward_ms": 0.0,
        "queue_depth": 0,
    }

    def maybe_reload_model():
        nonlocal model_mtime
        try:
            current_mtime = os.path.getmtime(model_path)
        except OSError:
            return

        if model_mtime is not None and current_mtime <= model_mtime:
            return

        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            model_mtime = current_mtime
            print(f"[Inference] Reloaded weights from {model_path}")
        except Exception as exc:
            print(f"[Inference] Failed to reload weights: {exc}")

    while True:
        maybe_reload_model()
        singles = []
        singles_meta = []
        batch_requests = []

        try:
            task = task_queue.get(timeout=IDLE_GET_TIMEOUT)
        except queue.Empty:
            continue
        except Exception:
            continue

        if len(task) == 4:
            task_id, payload, worker_id, is_batch = task
        else:
            task_id, payload, worker_id = task
            is_batch = False

        if is_batch:
            batch_requests.append((task_id, payload, worker_id))
        else:
            singles.append(payload)
            singles_meta.append((task_id, worker_id))

        gather_start = time.perf_counter()
        if not is_batch:
            while len(singles) < BATCH_SIZE:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    if (time.perf_counter() - gather_start) >= WAIT_TIMEOUT:
                        break
                    continue
                except Exception:
                    break

                if len(task) == 4:
                    task_id, payload, worker_id, is_batch = task
                else:
                    task_id, payload, worker_id = task
                    is_batch = False

                if is_batch:
                    batch_requests.append((task_id, payload, worker_id))
                else:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id))
        else:
            while True:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break

                if len(task) == 4:
                    task_id, payload, worker_id, nested_is_batch = task
                else:
                    task_id, payload, worker_id = task
                    nested_is_batch = False

                if nested_is_batch:
                    batch_requests.append((task_id, payload, worker_id))
                elif len(singles) < BATCH_SIZE:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id))

        gather_wait_ms = (time.perf_counter() - gather_start) * 1000.0

        single_batch = np.asarray(singles, dtype=np.float32) if singles else None
        explicit_arrays = [np.asarray(payload, dtype=np.float32) for _, payload, _ in batch_requests]
        forward_inputs = []

        if single_batch is not None and len(single_batch) > 0:
            forward_inputs.append(single_batch)
        if explicit_arrays:
            forward_inputs.extend(explicit_arrays)

        if forward_inputs:
            merged_batch = np.concatenate(forward_inputs, axis=0)
            tensor = torch.from_numpy(merged_batch).to(device, non_blocking=True)

            forward_start = time.perf_counter()
            with torch.inference_mode():
                logits, values = model(tensor)
                logits = logits.detach().cpu().numpy().astype(np.float16, copy=False)
                vals = values.detach().cpu().numpy().reshape(-1)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            cluster_stats["server_forwards"] += 1
            forward_events = cluster_stats["server_forwards"]
            cluster_stats["avg_forward_ms"] += (
                (forward_ms - cluster_stats["avg_forward_ms"]) / forward_events
            )

            offset = 0
            if single_batch is not None and len(single_batch) > 0:
                meta = {
                    "forward_ms": float(forward_ms),
                    "gather_wait_ms": float(gather_wait_ms),
                    "batch_size": int(len(single_batch)),
                    "merged_batch_size": int(len(merged_batch)),
                }
                for i, (task_id, worker_id) in enumerate(singles_meta):
                    response_queues[worker_id].put(
                        (task_id, logits[offset + i], float(vals[offset + i]), False, meta)
                    )

                offset += len(single_batch)
                cluster_stats["dynamic_batches"] += 1
                cluster_stats["single_requests"] += len(singles_meta)
                cluster_stats["total_requests"] += len(singles_meta)
                cluster_stats["total_positions"] += len(single_batch)
                dynamic_batches = cluster_stats["dynamic_batches"]
                cluster_stats["avg_dynamic_batch_size"] += (
                    (len(single_batch) - cluster_stats["avg_dynamic_batch_size"]) / dynamic_batches
                )
                cluster_stats["avg_gather_wait_ms"] += (
                    (gather_wait_ms - cluster_stats["avg_gather_wait_ms"]) / dynamic_batches
                )

            if explicit_arrays:
                meta = {
                    "forward_ms": float(forward_ms),
                    "gather_wait_ms": float(gather_wait_ms),
                    "merged_batch_size": int(len(merged_batch)),
                    "request_count": int(len(batch_requests)),
                }
                for (task_id, payload, worker_id), np_batch in zip(batch_requests, explicit_arrays):
                    batch_len = len(np_batch)
                    response_queues[worker_id].put(
                        (task_id, logits[offset:offset + batch_len], vals[offset:offset + batch_len], True, meta)
                    )
                    offset += batch_len

                cluster_stats["explicit_batch_requests"] += len(batch_requests)
                cluster_stats["merged_explicit_batches"] += 1
                cluster_stats["total_requests"] += len(batch_requests)
                cluster_stats["total_positions"] += sum(len(batch) for batch in explicit_arrays)
                explicit_requests = cluster_stats["explicit_batch_requests"]
                merged_explicit_batches = cluster_stats["merged_explicit_batches"]
                cluster_stats["avg_explicit_batch_size"] += (
                    ((sum(len(batch) for batch in explicit_arrays) / max(1, len(batch_requests))) - cluster_stats["avg_explicit_batch_size"]) / explicit_requests
                )
                cluster_stats["avg_merged_batch_size"] += (
                    (sum(len(batch) for batch in explicit_arrays) - cluster_stats["avg_merged_batch_size"]) / merged_explicit_batches
                )

        if shared_stats is not None and (time.perf_counter() - last_stats_push) >= 0.5:
            try:
                queue_depth = task_queue.qsize()
            except (AttributeError, NotImplementedError):
                queue_depth = -1

            cluster_stats["queue_depth"] = queue_depth
            current_cluster = dict(shared_stats.get("__cluster__", {}))
            current_cluster["inference"] = {
                "device": cluster_stats["device"],
                "total_requests": int(cluster_stats["total_requests"]),
                "total_positions": int(cluster_stats["total_positions"]),
                "single_requests": int(cluster_stats["single_requests"]),
                "explicit_batch_requests": int(cluster_stats["explicit_batch_requests"]),
                "dynamic_batches": int(cluster_stats["dynamic_batches"]),
                "merged_explicit_batches": int(cluster_stats["merged_explicit_batches"]),
                "avg_dynamic_batch_size": float(cluster_stats["avg_dynamic_batch_size"]),
                "avg_explicit_batch_size": float(cluster_stats["avg_explicit_batch_size"]),
                "avg_merged_batch_size": float(cluster_stats["avg_merged_batch_size"]),
                "avg_gather_wait_ms": float(cluster_stats["avg_gather_wait_ms"]),
                "avg_forward_ms": float(cluster_stats["avg_forward_ms"]),
                "queue_depth": int(cluster_stats["queue_depth"]),
                "max_batch_size": int(BATCH_SIZE),
            }
            shared_stats["__cluster__"] = current_cluster
            last_stats_push = time.perf_counter()
