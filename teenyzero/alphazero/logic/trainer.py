import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.samples = []
        self._load_all_data()

    def _load_all_data(self):
        """Loads all .npz files into memory for fast training."""
        for f in self.data_files:
            with np.load(f) as data:
                states = data['states']
                pis = data['pis']
                zs = data['zs']
                for i in range(len(states)):
                    self.samples.append((states[i], pis[i], zs[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, pi, z = self.samples[idx]
        return (
            torch.from_numpy(state).float(), 
            torch.from_numpy(pi).float(), 
            torch.tensor([z]).float()
        )

class AlphaTrainer:
    def __init__(self, model, device="cpu", lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        policy_losses = 0
        value_losses = 0

        for batch_idx, (states, target_pis, target_zs) in enumerate(dataloader):
            states, target_pis, target_zs = (
                states.to(self.device), 
                target_pis.to(self.device), 
                target_zs.to(self.device)
            )

            self.optimizer.zero_grad()

            # 1. Forward Pass
            out_pi_logits, out_v = self.model(states)

            # 2. Dual Loss Calculation
            # Policy Loss: Cross Entropy (using log_softmax for numerical stability)
            log_probs = F.log_softmax(out_pi_logits, dim=1)
            loss_pi = -torch.sum(target_pis * log_probs, dim=1).mean()

            # Value Loss: Mean Squared Error
            loss_v = F.mse_loss(out_v, target_zs)

            # Total Loss
            loss = loss_pi + loss_v
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            policy_losses += loss_pi.item()
            value_losses += loss_v.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[*] Training Complete: Loss {avg_loss:.4f} (Pol: {policy_losses/len(dataloader):.4f}, Val: {value_losses/len(dataloader):.4f})")
        return avg_loss

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[*] Saved new best model to {path}")