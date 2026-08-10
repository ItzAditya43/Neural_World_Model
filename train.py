import numpy as np
import torch
import torch.nn as nn
from model import WorldModel

EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3


def load_data():
    d = np.load("data/transitions.npz")
    frames = torch.from_numpy(d["frames"]).permute(0, 3, 1, 2)       # N,3,H,W
    next_frames = torch.from_numpy(d["next_frames"]).permute(0, 3, 1, 2)
    actions = torch.from_numpy(d["actions"])
    return frames, actions, next_frames


def main():
    assert torch.cuda.is_available(), "CUDA GPU required for training"
    device = "cuda"
    frames, actions, next_frames = load_data()
    n = frames.shape[0]
    print(f"{n} transitions loaded")

    model = WorldModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    idx = np.arange(n)
    for epoch in range(EPOCHS):
        np.random.shuffle(idx)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            f = frames[b].to(device)
            a = actions[b].to(device)
            nf = next_frames[b].to(device)

            pred_next, z, z_next = model(f, a)
            loss = loss_fn(pred_next, nf)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch+1}/{EPOCHS}  loss={total_loss/n_batches:.5f}")

    torch.save(model.state_dict(), "checkpoints/world_model.pt")
    print("saved checkpoints/world_model.pt")


if __name__ == "__main__":
    main()
