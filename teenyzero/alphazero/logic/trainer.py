import contextlib
import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from teenyzero.alphazero.config import INPUT_SHAPE
from teenyzero.alphazero.checkpoints import save_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


@dataclass
class ReplayFileInfo:
    path: str
    sample_count: int
    mtime: float
    state_shape: tuple[int, ...]
    shard_format: str = "npz"
    states_path: str | None = None
    pis_path: str | None = None
    zs_path: str | None = None
    meta_path: str | None = None


class ReplayWindowDataset(Dataset):
    def __init__(self, replay_files, sample_size=None, rng_seed=None, progress_callback=None, raw_cache_size=6):
        self.files = replay_files
        self.sample_size = sample_size
        self.rng_seed = rng_seed
        self.progress_callback = progress_callback
        self.raw_cache_size = max(1, int(raw_cache_size))
        self.states = np.empty((0, *INPUT_SHAPE), dtype=np.float32)
        self.pis = np.empty((0, 4672), dtype=np.float32)
        self.zs = np.empty((0,), dtype=np.float32)
        self.file_index = {info.path: info for info in self.files}
        self.sample_refs = []
        self._raw_cache = {}
        self._raw_cache_order = []
        self.lazy_raw_mode = bool(self.files) and all(info.shard_format == "raw" for info in self.files)
        if self.lazy_raw_mode:
            self._build_sample_refs()
        else:
            self._load_files()

    def _load_files(self):
        total_files = len(self.files)
        total_window_samples = sum(info.sample_count for info in self.files)
        target_samples = total_window_samples
        if self.sample_size is not None:
            target_samples = min(int(self.sample_size), total_window_samples)

        if target_samples <= 0:
            return

        if target_samples >= total_window_samples:
            selected_global = np.arange(total_window_samples, dtype=np.int64)
        else:
            rng = np.random.default_rng(self.rng_seed)
            selected_global = np.sort(
                rng.choice(total_window_samples, size=target_samples, replace=False)
            )

        state_chunks = []
        pi_chunks = []
        z_chunks = []
        load_started = time.perf_counter()
        global_offset = 0
        for file_idx, info in enumerate(self.files, start=1):
            file_start = global_offset
            file_end = global_offset + info.sample_count
            local_start = int(np.searchsorted(selected_global, file_start, side="left"))
            local_end = int(np.searchsorted(selected_global, file_end, side="left"))
            chosen = selected_global[local_start:local_end] - file_start

            if len(chosen) > 0:
                with np.load(info.path) as data:
                    state_chunks.append(np.asarray(data["states"][chosen], dtype=np.float32))
                    pi_chunks.append(np.asarray(data["pis"][chosen], dtype=np.float32))
                    z_chunks.append(np.asarray(data["zs"][chosen], dtype=np.float32))

            global_offset = file_end
            if self.progress_callback is not None and (file_idx == total_files or file_idx % 25 == 0):
                elapsed_s = time.perf_counter() - load_started
                safe_elapsed = max(elapsed_s, 1e-9)
                loaded_samples = int(sum(chunk.shape[0] for chunk in z_chunks))
                self.progress_callback(
                    {
                        "stage": "loading_replay_window",
                        "loaded_files": file_idx,
                        "total_files": total_files,
                        "loaded_samples": loaded_samples,
                        "selected_samples": target_samples,
                        "window_samples": total_window_samples,
                        "elapsed_s": elapsed_s,
                        "files_per_s": file_idx / safe_elapsed,
                        "samples_per_s": loaded_samples / safe_elapsed,
                    }
                )

        if z_chunks:
            self.states = np.ascontiguousarray(np.concatenate(state_chunks, axis=0))
            self.pis = np.ascontiguousarray(np.concatenate(pi_chunks, axis=0))
            self.zs = np.ascontiguousarray(np.concatenate(z_chunks, axis=0))

    def _build_sample_refs(self):
        total_files = len(self.files)
        total_window_samples = sum(info.sample_count for info in self.files)
        target_samples = total_window_samples
        if self.sample_size is not None:
            target_samples = min(int(self.sample_size), total_window_samples)

        if target_samples <= 0:
            return

        if target_samples >= total_window_samples:
            selected_global = np.arange(total_window_samples, dtype=np.int64)
        else:
            rng = np.random.default_rng(self.rng_seed)
            selected_global = np.sort(
                rng.choice(total_window_samples, size=target_samples, replace=False)
            )

        build_started = time.perf_counter()
        global_offset = 0
        sample_refs = []
        for file_idx, info in enumerate(self.files, start=1):
            file_start = global_offset
            file_end = global_offset + info.sample_count
            local_start = int(np.searchsorted(selected_global, file_start, side="left"))
            local_end = int(np.searchsorted(selected_global, file_end, side="left"))
            chosen = selected_global[local_start:local_end] - file_start
            if len(chosen) > 0:
                sample_refs.extend((info.path, int(local_idx)) for local_idx in chosen.tolist())
            global_offset = file_end
            if self.progress_callback is not None and (file_idx == total_files or file_idx % 25 == 0):
                elapsed_s = time.perf_counter() - build_started
                safe_elapsed = max(elapsed_s, 1e-9)
                self.progress_callback(
                    {
                        "stage": "indexing_replay_window",
                        "loaded_files": file_idx,
                        "total_files": total_files,
                        "loaded_samples": len(sample_refs),
                        "selected_samples": target_samples,
                        "window_samples": total_window_samples,
                        "elapsed_s": elapsed_s,
                        "files_per_s": file_idx / safe_elapsed,
                        "samples_per_s": len(sample_refs) / safe_elapsed,
                    }
                )
        self.sample_refs = sample_refs

    def _load_raw_arrays(self, info: ReplayFileInfo):
        cached = self._raw_cache.get(info.path)
        if cached is not None:
            if info.path in self._raw_cache_order:
                self._raw_cache_order.remove(info.path)
            self._raw_cache_order.append(info.path)
            return cached

        arrays = (
            np.load(info.states_path, mmap_mode="r"),
            np.load(info.pis_path, mmap_mode="r"),
            np.load(info.zs_path, mmap_mode="r"),
        )
        self._raw_cache[info.path] = arrays
        self._raw_cache_order.append(info.path)

        while len(self._raw_cache_order) > self.raw_cache_size:
            oldest_path = self._raw_cache_order.pop(0)
            self._raw_cache.pop(oldest_path, None)

        return arrays

    def __len__(self):
        if self.lazy_raw_mode:
            return len(self.sample_refs)
        return int(self.zs.shape[0])

    def __getitem__(self, idx):
        if self.lazy_raw_mode:
            file_path, local_idx = self.sample_refs[idx]
            info = self.file_index[file_path]
            states, pis, zs = self._load_raw_arrays(info)
            state = np.array(states[local_idx], dtype=np.float32, copy=True)
            pi = np.array(pis[local_idx], dtype=np.float32, copy=True)
            z = float(zs[local_idx])
            return (
                torch.from_numpy(state).float(),
                torch.from_numpy(pi).float(),
                torch.tensor([z], dtype=torch.float32),
            )
        state = self.states[idx]
        pi = self.pis[idx]
        z = self.zs[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(pi).float(),
            torch.tensor([z], dtype=torch.float32),
        )


def replay_file_infos(data_dir):
    infos = []
    for path in glob.glob(os.path.join(data_dir, "*.npz")):
        try:
            with np.load(path) as data:
                sample_count = int(len(data["zs"]))
                state_shape = tuple(int(dim) for dim in data["states"].shape[1:])
        except Exception:
            continue
        if state_shape != INPUT_SHAPE:
            continue
        infos.append(
            ReplayFileInfo(
                path=path,
                sample_count=sample_count,
                mtime=os.path.getmtime(path),
                state_shape=state_shape,
                shard_format="npz",
            )
        )
    for meta_path in glob.glob(os.path.join(data_dir, "*.meta.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            sample_count = int(meta["sample_count"])
            state_shape = tuple(int(dim) for dim in meta["state_shape"])
            stem = meta_path[: -len(".meta.json")]
            states_path = stem + ".states.npy"
            pis_path = stem + ".pis.npy"
            zs_path = stem + ".zs.npy"
            if not (os.path.exists(states_path) and os.path.exists(pis_path) and os.path.exists(zs_path)):
                continue
        except Exception:
            continue
        if state_shape != INPUT_SHAPE:
            continue
        infos.append(
            ReplayFileInfo(
                path=stem,
                sample_count=sample_count,
                mtime=os.path.getmtime(meta_path),
                state_shape=state_shape,
                shard_format="raw",
                states_path=states_path,
                pis_path=pis_path,
                zs_path=zs_path,
                meta_path=meta_path,
            )
        )
    infos.sort(key=lambda item: item.mtime)
    return infos


def replay_buffer_summary(data_dir):
    infos = replay_file_infos(data_dir)
    total_samples = sum(info.sample_count for info in infos)
    return {
        "files": infos,
        "file_count": len(infos),
        "sample_count": total_samples,
    }


def latest_replay_window(data_dir, max_samples):
    infos = replay_file_infos(data_dir)
    selected = []
    running_total = 0

    for info in reversed(infos):
        selected.append(info)
        running_total += info.sample_count
        if running_total >= max_samples:
            break

    selected.reverse()
    return selected, running_total


def prune_replay_buffer(data_dir, max_samples_to_keep):
    infos = replay_file_infos(data_dir)
    running_total = sum(info.sample_count for info in infos)
    removed = []

    while infos and running_total > max_samples_to_keep:
        oldest = infos.pop(0)
        try:
            if oldest.shard_format == "raw":
                for path in (oldest.states_path, oldest.pis_path, oldest.zs_path, oldest.meta_path):
                    if path:
                        os.remove(path)
            else:
                os.remove(oldest.path)
            removed.append(oldest.path)
            running_total -= oldest.sample_count
        except OSError:
            break

    return {
        "removed_files": removed,
        "remaining_samples": running_total,
        "remaining_files": len(infos),
    }


class AlphaTrainer:
    def __init__(
        self,
        model,
        device="cpu",
        lr=None,
        optimizer_name=None,
        weight_decay=None,
        momentum=None,
        grad_accum_steps=None,
        precision=None,
        use_compile=None,
        max_grad_norm=None,
    ):
        lr = PROFILE.train_lr if lr is None else lr
        optimizer_name = PROFILE.train_optimizer if optimizer_name is None else optimizer_name
        weight_decay = PROFILE.train_weight_decay if weight_decay is None else weight_decay
        momentum = PROFILE.train_momentum if momentum is None else momentum
        grad_accum_steps = PROFILE.train_grad_accum_steps if grad_accum_steps is None else grad_accum_steps
        precision = PROFILE.train_precision if precision is None else precision
        use_compile = PROFILE.train_compile if use_compile is None else use_compile
        max_grad_norm = PROFILE.max_grad_norm if max_grad_norm is None else max_grad_norm

        if device == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)
        self.device = device
        self.use_channels_last = self.device == "cuda"
        self.optimizer_name = optimizer_name.lower()
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.precision = precision.lower()
        self.use_compile = bool(use_compile and hasattr(torch, "compile"))
        self.max_grad_norm = float(max_grad_norm or 0.0)
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.use_autocast = self.device == "cuda" and self.precision in {"bf16", "fp16"}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device == "cuda" and self.precision == "fp16")
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        if self.use_compile:
            self.model = torch.compile(self.model)

    def _autocast_context(self):
        if not self.use_autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def train_epoch(self, dataloader, progress_callback=None):
        self.model.train()
        total_loss = 0.0
        policy_losses = 0.0
        value_losses = 0.0
        batch_count = 0
        skipped_batches = 0
        samples_seen = 0
        total_batches = len(dataloader)
        epoch_started = time.perf_counter()
        optimizer_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (states, target_pis, target_zs) in enumerate(dataloader, start=1):
            states = states.to(self.device, non_blocking=self.device == "cuda")
            if self.use_channels_last:
                states = states.contiguous(memory_format=torch.channels_last)
            target_pis = target_pis.to(self.device, non_blocking=self.device == "cuda")
            target_zs = torch.nan_to_num(
                target_zs.to(self.device, non_blocking=self.device == "cuda"),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp_(-1.0, 1.0)
            target_pis = torch.nan_to_num(target_pis, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)
            pi_sums = target_pis.sum(dim=1, keepdim=True)
            valid_rows = pi_sums.squeeze(1) > 0
            if valid_rows.any():
                target_pis[valid_rows] = target_pis[valid_rows] / pi_sums[valid_rows]
            if (~valid_rows).any():
                target_pis[~valid_rows] = 0.0
            samples_seen += int(states.size(0))

            with self._autocast_context():
                out_pi_logits, out_v = self.model(states)
                log_probs = F.log_softmax(out_pi_logits, dim=1)
                loss_pi = -torch.sum(target_pis * log_probs, dim=1).mean()
                loss_v = F.mse_loss(out_v, target_zs)
                loss = loss_pi + loss_v

            if not torch.isfinite(loss):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / float(self.grad_accum_steps)
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (batch_idx % self.grad_accum_steps == 0) or (batch_idx == total_batches)
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            total_loss += float(loss.item())
            policy_losses += float(loss_pi.item())
            value_losses += float(loss_v.item())
            batch_count += 1

            if progress_callback is not None and (batch_idx == total_batches or batch_idx % 10 == 0):
                elapsed_s = time.perf_counter() - epoch_started
                safe_elapsed = max(elapsed_s, 1e-9)
                safe_batch_count = max(batch_count, 1)
                progress_callback(
                    {
                        "stage": "training_batches",
                        "completed_batches": batch_idx,
                        "total_batches": total_batches,
                        "running_loss": total_loss / safe_batch_count,
                        "running_policy_loss": policy_losses / safe_batch_count,
                        "running_value_loss": value_losses / safe_batch_count,
                        "elapsed_s": elapsed_s,
                        "avg_batch_time_ms": (elapsed_s / safe_batch_count) * 1000.0,
                        "batches_per_s": batch_count / safe_elapsed,
                        "samples_per_s": samples_seen / safe_elapsed,
                        "samples_seen": samples_seen,
                        "skipped_batches": skipped_batches,
                        "optimizer_steps": optimizer_steps,
                    }
                )

        if batch_count == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "batches": 0,
                "samples_seen": 0,
                "duration_s": 0.0,
                "batches_per_s": 0.0,
                "samples_per_s": 0.0,
                "avg_batch_time_ms": 0.0,
                "skipped_batches": skipped_batches,
                "optimizer_steps": optimizer_steps,
            }

        elapsed_s = time.perf_counter() - epoch_started
        safe_elapsed = max(elapsed_s, 1e-9)
        metrics = {
            "loss": total_loss / batch_count,
            "policy_loss": policy_losses / batch_count,
            "value_loss": value_losses / batch_count,
            "batches": batch_count,
            "samples_seen": samples_seen,
            "duration_s": elapsed_s,
            "batches_per_s": batch_count / safe_elapsed,
            "samples_per_s": samples_seen / safe_elapsed,
            "avg_batch_time_ms": (elapsed_s / batch_count) * 1000.0,
            "skipped_batches": skipped_batches,
            "optimizer_steps": optimizer_steps,
        }
        print(
            "[*] Training Complete: "
            f"Loss {metrics['loss']:.4f} "
            f"(Pol: {metrics['policy_loss']:.4f}, Val: {metrics['value_loss']:.4f})"
        )
        if skipped_batches:
            print(f"[*] Skipped {skipped_batches} non-finite training batches.")
        return metrics

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_checkpoint(self.model, path)
        print(f"[*] Saved checkpoint to {path}")


def dataloader_for_replay_window(
    data_dir,
    max_samples,
    sample_size,
    batch_size,
    shuffle=True,
    progress_callback=None,
    rng_seed=None,
    num_workers=None,
    pin_memory=None,
    prefetch_factor=None,
):
    files, sample_count = latest_replay_window(data_dir, max_samples=max_samples)
    dataset = ReplayWindowDataset(
        files,
        sample_size=sample_size,
        rng_seed=rng_seed,
        progress_callback=progress_callback,
    )
    effective_num_workers = PROFILE.train_num_workers if num_workers is None else num_workers
    effective_pin_memory = PROFILE.train_pin_memory if pin_memory is None else pin_memory
    effective_prefetch = PROFILE.train_prefetch_factor if prefetch_factor is None else prefetch_factor
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": effective_num_workers,
        "pin_memory": effective_pin_memory,
        "persistent_workers": effective_num_workers > 0,
    }
    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = effective_prefetch
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sample_count, files
