import os
import torch
from torch.utils.data import DataLoader
from teenyzero.alphazero.model import AlphaNet
from teenyzero.alphazero.logic.trainer import AlphaTrainer, ChessDataset

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_dir = "teenyzero/alphazero/data/replay_buffer"
    model_path = "models/best_model.pth"

    # 1. Load Model
    model = AlphaNet(num_res_blocks=10, channels=128)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    # 2. Load Data
    dataset = ChessDataset(data_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Train
    trainer = AlphaTrainer(model, device=device)
    trainer.train_epoch(loader)
    trainer.save_checkpoint(model_path)

if __name__ == "__main__":
    main()