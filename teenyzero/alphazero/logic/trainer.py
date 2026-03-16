import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class ReplayFileInfo:
    path: str
    sample_count: int
    mtime: float


class ReplayWindowDataset(Dataset):
    def __init__(self, replay_files, progress_callback=None):
        self.samples = []
        self.files = replay_files
        self.progress_callback = progress_callback
        self._load_files()

    def _load_files(self):
        total_files = len(self.files)
        for file_idx, info in enumerate(self.files, start=1):
            with np.load(info.path) as data:
                states = data["states"]
                pis = data["pis"]
                zs = data["zs"]
                for idx in range(len(states)):
                    self.samples.append((states[idx], pis[idx], zs[idx]))
            if self.progress_callback is not None and (file_idx == total_files or file_idx % 25 == 0):
                self.progress_callback(
                    {
                        "stage": "loading_replay_window",
                        "loaded_files": file_idx,
                        "total_files": total_files,
                        "loaded_samples": len(self.samples),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, pi, z = self.samples[idx]
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
        except Exception:
            continue
        infos.append(
            ReplayFileInfo(
                path=path,
                sample_count=sample_count,
                mtime=os.path.getmtime(path),
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
    def __init__(self, model, device="cpu", lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

    def train_epoch(self, dataloader, progress_callback=None):
        self.model.train()
        total_loss = 0.0
        policy_losses = 0.0
        value_losses = 0.0
        batch_count = 0
        total_batches = len(dataloader)

        for batch_idx, (states, target_pis, target_zs) in enumerate(dataloader, start=1):
            states = states.to(self.device)
            target_pis = target_pis.to(self.device)
            target_zs = target_zs.to(self.device)

            self.optimizer.zero_grad()
            out_pi_logits, out_v = self.model(states)

            log_probs = F.log_softmax(out_pi_logits, dim=1)
            loss_pi = -torch.sum(target_pis * log_probs, dim=1).mean()
            loss_v = F.mse_loss(out_v, target_zs)
            loss = loss_pi + loss_v

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            policy_losses += float(loss_pi.item())
            value_losses += float(loss_v.item())
            batch_count += 1

            if progress_callback is not None and (batch_idx == total_batches or batch_idx % 10 == 0):
                progress_callback(
                    {
                        "stage": "training_batches",
                        "completed_batches": batch_idx,
                        "total_batches": total_batches,
                        "running_loss": total_loss / batch_count,
                        "running_policy_loss": policy_losses / batch_count,
                        "running_value_loss": value_losses / batch_count,
                    }
                )

        if batch_count == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "batches": 0,
            }

        metrics = {
            "loss": total_loss / batch_count,
            "policy_loss": policy_losses / batch_count,
            "value_loss": value_losses / batch_count,
            "batches": batch_count,
        }
        print(
            "[*] Training Complete: "
            f"Loss {metrics['loss']:.4f} "
            f"(Pol: {metrics['policy_loss']:.4f}, Val: {metrics['value_loss']:.4f})"
        )
        return metrics

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[*] Saved checkpoint to {path}")


def dataloader_for_replay_window(data_dir, max_samples, batch_size, shuffle=True, progress_callback=None):
    files, sample_count = latest_replay_window(data_dir, max_samples=max_samples)
    dataset = ReplayWindowDataset(files, progress_callback=progress_callback)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, sample_count, files
