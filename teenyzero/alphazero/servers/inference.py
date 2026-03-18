import time
import queue
import os
from collections import deque
import torch
import numpy as np

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


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
    print(f"[Inference] Initializing AlphaNet on {device}...")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    model = build_model()
    if device == "cuda" and PROFILE.inference_precision == "bf16":
        inference_dtype = torch.bfloat16
    else:
        inference_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    use_channels_last = device in {"mps", "cuda"}

    if device == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    try:
        load_result = load_checkpoint(model, model_path, map_location="cpu", allow_partial=True)
        if load_result["loaded"]:
            print(f"[Inference] Weights loaded successfully from {model_path}")
        else:
            print(f"[Inference] Warning: Starting with fresh weights ({load_result['reason']})")
    except Exception as e:
        print(f"[Inference] Warning: Starting with fresh weights ({e})")

    try:
        model_mtime = os.path.getmtime(model_path)
    except OSError:
        model_mtime = None

    model = model.to(device=device, dtype=inference_dtype)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device == "cuda" and PROFILE.inference_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()

    MAX_SINGLE_BATCH = PROFILE.inference_single_batch if device in {"cuda", "mps"} else 64
    MAX_MERGED_POSITIONS = PROFILE.inference_merged_batch if device in {"cuda", "mps"} else 64
    WAIT_TIMEOUT = PROFILE.inference_wait_timeout
    IDLE_GET_TIMEOUT = 1.0
    response_prob_dtype = np.float16 if device in {"cuda", "mps"} else np.float32
    batch_pad_buckets = ()
    if device == "cuda" and PROFILE.inference_compile:
        batch_pad_buckets = tuple(
            candidate
            for candidate in (32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)
            if candidate <= max(MAX_MERGED_POSITIONS, MAX_SINGLE_BATCH)
        )
        if MAX_MERGED_POSITIONS not in batch_pad_buckets:
            batch_pad_buckets = tuple(sorted((*batch_pad_buckets, MAX_MERGED_POSITIONS)))
    last_stats_push = 0.0
    pending_tasks = deque()
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
        "avg_padded_batch_size": 0.0,
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
            load_checkpoint(model, model_path, map_location="cpu", allow_partial=True)
            model.to(device=device, dtype=inference_dtype)
            if use_channels_last:
                model.to(memory_format=torch.channels_last)
            model.eval()
            model_mtime = current_mtime
            print(f"[Inference] Reloaded weights from {model_path}")
        except Exception as exc:
            print(f"[Inference] Failed to reload weights: {exc}")

    def next_task():
        if pending_tasks:
            return pending_tasks.popleft()
        return task_queue.get(timeout=IDLE_GET_TIMEOUT)

    def unpack_task(task):
        if len(task) == 5:
            task_id, payload, worker_id, is_batch, legal_meta = task
            return task_id, payload, worker_id, is_batch, legal_meta
        if len(task) == 4:
            task_id, payload, worker_id, is_batch = task
            return task_id, payload, worker_id, is_batch, None
        task_id, payload, worker_id = task
        return task_id, payload, worker_id, False, None

    def pad_batch_size(batch_size):
        if not batch_pad_buckets:
            return batch_size
        for candidate in batch_pad_buckets:
            if batch_size <= candidate:
                return candidate
        return batch_size

    def sparse_policy_from_row(logit_row, legal_indices):
        if legal_indices is None:
            return logit_row.detach().cpu().numpy().astype(response_prob_dtype, copy=False)
        if len(legal_indices) == 0:
            return np.empty((0,), dtype=response_prob_dtype)
        indices = torch.as_tensor(legal_indices, device=logit_row.device, dtype=torch.long)
        legal_logits = logit_row.float().index_select(0, indices)
        probs = torch.softmax(legal_logits, dim=0)
        target_dtype = torch.float16 if response_prob_dtype == np.float16 else torch.float32
        return probs.to(dtype=target_dtype).cpu().numpy()

    while True:
        maybe_reload_model()
        singles = []
        singles_meta = []
        batch_requests = []
        merged_positions = 0

        try:
            task = next_task()
        except queue.Empty:
            continue
        except Exception:
            continue

        task_id, payload, worker_id, is_batch, legal_meta = unpack_task(task)

        if is_batch:
            batch_requests.append((task_id, payload, worker_id, legal_meta))
            merged_positions += int(len(payload))
        else:
            singles.append(payload)
            singles_meta.append((task_id, worker_id, legal_meta))
            merged_positions += 1

        gather_start = time.perf_counter()
        if not is_batch:
            while len(singles) < MAX_SINGLE_BATCH and merged_positions < MAX_MERGED_POSITIONS:
                try:
                    task = pending_tasks.popleft() if pending_tasks else task_queue.get_nowait()
                except queue.Empty:
                    if (time.perf_counter() - gather_start) >= WAIT_TIMEOUT:
                        break
                    continue
                except Exception:
                    break

                task_id, payload, worker_id, is_batch, legal_meta = unpack_task(task)

                if is_batch:
                    payload_len = int(len(payload))
                    if merged_positions + payload_len > MAX_MERGED_POSITIONS and merged_positions > 0:
                        pending_tasks.appendleft((task_id, payload, worker_id, True, legal_meta))
                        break
                    batch_requests.append((task_id, payload, worker_id, legal_meta))
                    merged_positions += payload_len
                else:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id, legal_meta))
                    merged_positions += 1
        else:
            while merged_positions < MAX_MERGED_POSITIONS:
                try:
                    task = pending_tasks.popleft() if pending_tasks else task_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break

                task_id, payload, worker_id, nested_is_batch, legal_meta = unpack_task(task)

                if nested_is_batch:
                    payload_len = int(len(payload))
                    if merged_positions + payload_len > MAX_MERGED_POSITIONS and merged_positions > 0:
                        pending_tasks.appendleft((task_id, payload, worker_id, True, legal_meta))
                        break
                    batch_requests.append((task_id, payload, worker_id, legal_meta))
                    merged_positions += payload_len
                elif len(singles) < MAX_SINGLE_BATCH and merged_positions < MAX_MERGED_POSITIONS:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id, legal_meta))
                    merged_positions += 1
                else:
                    pending_tasks.appendleft((task_id, payload, worker_id, False, legal_meta))
                    break

        gather_wait_ms = (time.perf_counter() - gather_start) * 1000.0

        single_batch = np.asarray(singles) if singles else None
        explicit_arrays = [np.asarray(payload) for _, payload, _, _ in batch_requests]
        forward_inputs = []

        if single_batch is not None and len(single_batch) > 0:
            forward_inputs.append(single_batch)
        if explicit_arrays:
            forward_inputs.extend(explicit_arrays)

        if forward_inputs:
            merged_batch = np.concatenate(forward_inputs, axis=0)
            effective_batch_size = int(len(merged_batch))
            padded_batch_size = pad_batch_size(effective_batch_size)
            if padded_batch_size > effective_batch_size:
                run_batch = np.zeros((padded_batch_size, *merged_batch.shape[1:]), dtype=merged_batch.dtype)
                run_batch[:effective_batch_size] = merged_batch
            else:
                run_batch = merged_batch

            tensor = torch.from_numpy(run_batch).to(device=device, dtype=inference_dtype, non_blocking=True)
            if use_channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)

            forward_start = time.perf_counter()
            with torch.inference_mode():
                logits, values = model(tensor)
                logits = logits.detach()[:effective_batch_size]
                values = values.detach().reshape(-1)[:effective_batch_size]
                if logits.dtype == torch.bfloat16:
                    logits = logits.to(dtype=torch.float32)
                if values.dtype == torch.bfloat16:
                    values = values.to(dtype=torch.float32)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            vals = values.cpu().numpy().reshape(-1).astype(np.float32, copy=False)

            cluster_stats["server_forwards"] += 1
            forward_events = cluster_stats["server_forwards"]
            cluster_stats["avg_forward_ms"] += (
                (forward_ms - cluster_stats["avg_forward_ms"]) / forward_events
            )
            cluster_stats["avg_padded_batch_size"] += (
                (padded_batch_size - cluster_stats["avg_padded_batch_size"]) / forward_events
            )

            offset = 0
            if single_batch is not None and len(single_batch) > 0:
                meta = {
                    "forward_ms": float(forward_ms),
                    "gather_wait_ms": float(gather_wait_ms),
                    "batch_size": int(len(single_batch)),
                    "merged_batch_size": int(effective_batch_size),
                    "padded_batch_size": int(padded_batch_size),
                }
                for i, (task_id, worker_id, legal_meta) in enumerate(singles_meta):
                    payload = sparse_policy_from_row(logits[offset + i], legal_meta)
                    meta["sparse_policy"] = bool(legal_meta is not None)
                    response_queues[worker_id].put(
                        (task_id, payload, float(vals[offset + i]), False, meta)
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
                    "merged_batch_size": int(effective_batch_size),
                    "padded_batch_size": int(padded_batch_size),
                    "request_count": int(len(batch_requests)),
                }
                for (task_id, payload, worker_id, legal_meta), np_batch in zip(batch_requests, explicit_arrays):
                    batch_len = len(np_batch)
                    batch_logits = logits[offset:offset + batch_len]
                    if legal_meta is not None:
                        response_payload = [
                            sparse_policy_from_row(batch_logits[row_idx], legal_indices)
                            for row_idx, legal_indices in enumerate(legal_meta)
                        ]
                    else:
                        response_payload = batch_logits.detach().cpu().numpy().astype(response_prob_dtype, copy=False)
                    meta["sparse_policy"] = bool(legal_meta is not None)
                    response_queues[worker_id].put(
                        (
                            task_id,
                            response_payload,
                            vals[offset:offset + batch_len],
                            True,
                            meta,
                        )
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

            cluster_stats["queue_depth"] = queue_depth + len(pending_tasks) if queue_depth >= 0 else len(pending_tasks)
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
                "avg_padded_batch_size": float(cluster_stats["avg_padded_batch_size"]),
                "queue_depth": int(cluster_stats["queue_depth"]),
                "max_batch_size": int(MAX_MERGED_POSITIONS),
            }
            shared_stats["__cluster__"] = current_cluster
            last_stats_push = time.perf_counter()
