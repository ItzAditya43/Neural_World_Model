"""
Neural World Model — expanded study, run on Kaggle GPU.

Bigger dataset + longer training + real quantitative evaluation:
  - train/val reconstruction loss curves
  - compounding-error curve: open-loop imagined position error vs horizon
  - planner (random-shooting MPC) vs random-action and heuristic baselines
  - ablation over number of imagined candidates

The planning/ablation evaluation is fully batched (all episodes and all
imagined candidates run as one tensor per timestep) instead of looping over
them one at a time — this cuts GPU calls from ~1.4M down to a few thousand.

Everything is self-contained in this one file (env + model + train + eval)
so it can run as a single Kaggle script kernel with no repo dependency.
All results are written to /kaggle/working/ as JSON + PNG plots.
"""
import json
import os
import time
import subprocess
import sys
import numpy as np

# Kaggle buffers stdout when it's not a TTY, so `kaggle kernels logs` shows
# nothing until the process exits. Force line buffering so progress is
# actually visible while the kernel is still running.
sys.stdout.reconfigure(line_buffering=True)

# Kaggle sometimes assigns an older Tesla P100 (Pascal, sm_60). Recent
# stable PyTorch releases (2.5+) dropped sm_60 kernels entirely, so
# reinstalling "latest" doesn't help — pin an older cu118 build that
# still ships Pascal kernels (also works fine on T4).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.3.1", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
assert torch.cuda.is_available(), "CUDA GPU required — no CPU fallback"
device = "cuda"
print("device:", device, "-", torch.cuda.get_device_name(0), flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                       "--format=csv"], capture_output=True, text=True).stdout, flush=True)


def save_results_snapshot(results):
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

_sanity = torch.nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8).cuda())
print("CUDA sanity check passed")

# ----------------------------------------------------------------------
# Environment (single + batched/vectorized versions)
# ----------------------------------------------------------------------
IMG_SIZE = 48
GRAVITY = 0.0018
THRUST = 0.010
DAMPING = 0.995
RADIUS = 0.045
DT = 1.0
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "noop", 1: "left", 2: "right", 3: "up"}
GOAL_Y = 0.12


class BallWorld:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.x = self.rng.uniform(0.2, 0.8)
        self.y = self.rng.uniform(0.2, 0.5)
        self.vx = self.rng.uniform(-0.01, 0.01)
        self.vy = 0.0
        return self.state()

    def state(self):
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.vx -= THRUST
        elif action == 2:
            self.vx += THRUST
        elif action == 3:
            self.vy -= THRUST * 1.6
        self.vy += GRAVITY
        self.vx *= DAMPING
        self.vy *= DAMPING
        self.x += self.vx * DT
        self.y += self.vy * DT
        if self.x - RADIUS < 0:
            self.x = RADIUS; self.vx *= -0.85
        if self.x + RADIUS > 1:
            self.x = 1 - RADIUS; self.vx *= -0.85
        if self.y - RADIUS < 0:
            self.y = RADIUS; self.vy *= -0.85
        if self.y + RADIUS > 1:
            self.y = 1 - RADIUS; self.vy *= -0.7
        return self.state()

    def render(self, size=IMG_SIZE):
        img = np.zeros((size, size, 3), dtype=np.float32)
        for row in range(size):
            t = row / size
            img[row, :, 2] = 0.05 + 0.10 * t
            img[row, :, 0] = 0.02 + 0.02 * t
        cx, cy = int(self.x * size), int(self.y * size)
        r = max(2, int(RADIUS * size))
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = [1.0, 0.55, 0.15]
        return img


class BatchedBallWorld:
    """Vectorized version of BallWorld: steps B independent episodes in lockstep."""

    def __init__(self, seeds):
        self.B = len(seeds)
        rngs = [np.random.default_rng(s) for s in seeds]
        self.x = np.array([r.uniform(0.2, 0.8) for r in rngs], dtype=np.float32)
        self.y = np.array([r.uniform(0.2, 0.5) for r in rngs], dtype=np.float32)
        self.vx = np.array([r.uniform(-0.01, 0.01) for r in rngs], dtype=np.float32)
        self.vy = np.zeros(self.B, dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        self.vx = self.vx + np.where(actions == 1, -THRUST, 0.0).astype(np.float32)
        self.vx = self.vx + np.where(actions == 2, THRUST, 0.0).astype(np.float32)
        self.vy = self.vy + np.where(actions == 3, -THRUST * 1.6, 0.0).astype(np.float32)
        self.vy = self.vy + GRAVITY
        self.vx = self.vx * DAMPING
        self.vy = self.vy * DAMPING
        self.x = self.x + self.vx * DT
        self.y = self.y + self.vy * DT

        lo = self.x - RADIUS < 0
        self.x = np.where(lo, RADIUS, self.x); self.vx = np.where(lo, self.vx * -0.85, self.vx)
        hi = self.x + RADIUS > 1
        self.x = np.where(hi, 1 - RADIUS, self.x); self.vx = np.where(hi, self.vx * -0.85, self.vx)
        lo = self.y - RADIUS < 0
        self.y = np.where(lo, RADIUS, self.y); self.vy = np.where(lo, self.vy * -0.85, self.vy)
        hi = self.y + RADIUS > 1
        self.y = np.where(hi, 1 - RADIUS, self.y); self.vy = np.where(hi, self.vy * -0.7, self.vy)

    def render(self, size=IMG_SIZE):
        B = self.B
        img = np.zeros((B, size, size, 3), dtype=np.float32)
        row_t = np.arange(size, dtype=np.float32) / size
        img[:, :, :, 2] = (0.05 + 0.10 * row_t)[None, :, None]
        img[:, :, :, 0] = (0.02 + 0.02 * row_t)[None, :, None]
        cx = (self.x * size).astype(int)
        cy = (self.y * size).astype(int)
        r = max(2, int(RADIUS * size))
        yy, xx = np.ogrid[:size, :size]
        for i in range(B):
            mask = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2 <= r * r
            img[i][mask] = [1.0, 0.55, 0.15]
        return img


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
LATENT_DIM = 32
N_ACTIONS = 4


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
        )
        self.fc = nn.Linear(64 * 6 * 6, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 6 * 6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 6, 6))


class Transition(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z, action):
        a = self.action_emb(action)
        return z + self.net(torch.cat([z, a], dim=-1))


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.transition = Transition()

    def forward(self, frame, action):
        z = self.encoder(frame)
        z_next = self.transition(z, action)
        return self.decoder(z_next), z, z_next

    def imagine_step(self, z, action):
        z_next = self.transition(z, action)
        return z_next, self.decoder(z_next)


# ----------------------------------------------------------------------
# 1. Generate a much bigger dataset than the local run (180k vs 24k)
# ----------------------------------------------------------------------
N_EPISODES = 3000
STEPS_PER_EP = 60

print(f"\n== Generating dataset: {N_EPISODES} episodes x {STEPS_PER_EP} steps ==")
t0 = time.time()
rng = np.random.default_rng(0)
frames, actions, next_frames = [], [], []
for ep in range(N_EPISODES):
    world = BallWorld(seed=ep)
    world.reset()
    for t in range(STEPS_PER_EP):
        f = world.render()
        a = int(rng.choice(ACTIONS, p=[0.4, 0.2, 0.2, 0.2]))
        world.step(a)
        nf = world.render()
        frames.append(f); actions.append(a); next_frames.append(nf)
# Store as uint8 (0-255) instead of float32 -- cuts dataset RAM ~4x (this
# dataset would otherwise be ~10GB as float32). Converted to float on GPU
# per-batch instead.
frames = (np.stack(frames) * 255).astype(np.uint8)
next_frames = (np.stack(next_frames) * 255).astype(np.uint8)
actions = np.array(actions, dtype=np.int64)
print(f"dataset: {frames.shape}, {frames.nbytes / 1e9:.2f}GB, generated in {time.time()-t0:.1f}s", flush=True)

n = frames.shape[0]
idx = np.random.default_rng(1).permutation(n)
n_val = int(n * 0.1)
val_idx, train_idx = idx[:n_val], idx[n_val:]

frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)
next_frames_t = torch.from_numpy(next_frames).permute(0, 3, 1, 2)
actions_t = torch.from_numpy(actions)


def to_float_batch(t):
    return t.to(device).float() / 255.0

# ----------------------------------------------------------------------
# 2. Train, with train/val loss tracked every epoch
# ----------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 256
LR = 3e-4
GRAD_CLIP = 1.0

model = WorldModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

history = {"epoch": [], "train_loss": [], "val_loss": []}

print(f"\n== Training: {EPOCHS} epochs, {len(train_idx)} train / {len(val_idx)} val ==")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(train_idx)
    total, nb = 0.0, 0
    for start in range(0, len(perm), BATCH_SIZE):
        b = perm[start:start + BATCH_SIZE]
        f = to_float_batch(frames_t[b]); a = actions_t[b].to(device); nf = to_float_batch(next_frames_t[b])
        pred, z, z_next = model(f, a)
        loss = loss_fn(pred, nf)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        total += loss.item(); nb += 1
    train_loss = total / nb

    model.eval()
    with torch.no_grad():
        total, nb = 0.0, 0
        for start in range(0, len(val_idx), BATCH_SIZE):
            b = val_idx[start:start + BATCH_SIZE]
            f = to_float_batch(frames_t[b]); a = actions_t[b].to(device); nf = to_float_batch(next_frames_t[b])
            pred, _, _ = model(f, a)
            total += loss_fn(pred, nf).item(); nb += 1
        val_loss = total / nb

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    print(f"epoch {epoch+1:2d}/{EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}")
print(f"training took {time.time()-t0:.1f}s")

torch.save(model.state_dict(), f"{OUT}/world_model_kaggle.pt")

plt.figure(figsize=(6, 4))
plt.plot(history["epoch"], history["train_loss"], label="train")
plt.plot(history["epoch"], history["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("reconstruction MSE"); plt.yscale("log")
plt.title("World model training (180k transitions)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/loss_curves.png", dpi=130)
plt.close()

results = {
    "device": torch.cuda.get_device_name(0),
    "dataset": {"n_episodes": N_EPISODES, "steps_per_episode": STEPS_PER_EP, "n_transitions": int(n)},
    "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "final_train_loss": history["train_loss"][-1],
                 "final_val_loss": history["val_loss"][-1], "history": history},
}
save_results_snapshot(results)

model.eval()


def estimate_ball_pos(frame):
    """CPU/numpy version used only by the (cheap) compounding-error study."""
    b = frame.sum(axis=-1)
    thresh = b > (b.max() * 0.75)
    if thresh.sum() == 0:
        return (0.5, 0.5)
    ys_idx, xs_idx = np.nonzero(thresh)
    h, w = frame.shape[:2]
    return (float(xs_idx.mean()) / w, float(ys_idx.mean()) / h)


def estimate_ball_pos_batched(frame):
    """GPU/tensor version: frame is (N,3,H,W) in [0,1]. Returns (N,2) xy in [0,1]."""
    N, C, H, W = frame.shape
    brightness = frame.sum(dim=1)  # (N,H,W)
    maxval = brightness.amax(dim=(1, 2), keepdim=True)
    thresh = (brightness > 0.75 * maxval).float()
    total = thresh.sum(dim=(1, 2)).clamp(min=1.0)
    ys_idx = torch.arange(H, device=frame.device, dtype=torch.float32).view(1, H, 1)
    xs_idx = torch.arange(W, device=frame.device, dtype=torch.float32).view(1, 1, W)
    cx = (thresh * xs_idx).sum(dim=(1, 2)) / total / W
    cy = (thresh * ys_idx).sum(dim=(1, 2)) / total / H
    return torch.stack([cx, cy], dim=1)


# ----------------------------------------------------------------------
# 3. Compounding-error study: open-loop imagined position error vs horizon
#    (small enough to leave as a simple per-trajectory loop)
# ----------------------------------------------------------------------
print("\n== Compounding-error study (open-loop imagination vs ground truth) ==")
t0 = time.time()
HORIZONS = list(range(1, 15))
N_TRAJ = 60
errors_by_h = {h: [] for h in HORIZONS}

with torch.no_grad():
    for traj in range(N_TRAJ):
        world = BallWorld(seed=10_000 + traj)
        world.reset()
        real_positions = [(world.x, world.y)]
        real_frame0 = world.render()
        action_seq = [int(np.random.default_rng(traj).choice(ACTIONS, p=[0.35, 0.2, 0.2, 0.25])) for _ in range(max(HORIZONS))]

        for a in action_seq:
            world.step(a)
            real_positions.append((world.x, world.y))

        f0 = torch.from_numpy(real_frame0).permute(2, 0, 1).unsqueeze(0).to(device)
        z = model.encoder(f0)
        for t, a in enumerate(action_seq, start=1):
            a_t = torch.tensor([a], device=device, dtype=torch.long)
            z, frame_pred = model.imagine_step(z, a_t)
            if t in errors_by_h:
                pred_pos = estimate_ball_pos(frame_pred[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                true_pos = real_positions[t]
                err = float(np.hypot(pred_pos[0] - true_pos[0], pred_pos[1] - true_pos[1]))
                errors_by_h[t].append(err)

mean_err = [float(np.mean(errors_by_h[h])) for h in HORIZONS]
std_err = [float(np.std(errors_by_h[h])) for h in HORIZONS]
print(f"compounding-error study took {time.time()-t0:.1f}s")

plt.figure(figsize=(6, 4))
plt.plot(HORIZONS, mean_err, marker="o", color="#c9591c")
plt.fill_between(HORIZONS,
                  [m - s for m, s in zip(mean_err, std_err)],
                  [m + s for m, s in zip(mean_err, std_err)],
                  alpha=0.15, color="#c9591c")
plt.xlabel("imagined steps ahead (open-loop, no re-encoding)")
plt.ylabel("position error (normalized units)")
plt.title(f"Compounding error over {N_TRAJ} random trajectories")
plt.tight_layout()
plt.savefig(f"{OUT}/compounding_error.png", dpi=130)
plt.close()
print("mean position error @1 step:", round(mean_err[0], 4), " @14 steps:", round(mean_err[-1], 4))

results["compounding_error"] = {"horizons": HORIZONS, "mean_error": mean_err, "std_error": std_err}
save_results_snapshot(results)


# ----------------------------------------------------------------------
# 4 & 5. Planner vs baselines, and ablation over candidate count
#    Fully batched: all episodes x all candidates run as ONE tensor per
#    horizon step, instead of a python loop per candidate per episode.
# ----------------------------------------------------------------------
def score_batched(xs, ys):
    """xs, ys: (B, C, T) tensors of imagined positions -> (B, C) scores."""
    height_reward = -(ys - GOAL_Y).clamp(min=0).mean(dim=2) * 3.0
    overshoot_penalty = -(GOAL_Y - ys).clamp(min=0).mean(dim=2) * 1.5
    center_penalty = -(xs - 0.5).abs().mean(dim=2) * 0.3
    return height_reward + overshoot_penalty + center_penalty


@torch.no_grad()
def run_planner_batch(model, seeds, horizon=14, n_candidates=22, real_steps=40, decode_every=2, action_rng_seed=123):
    """Runs len(seeds) episodes in lockstep. At every real step, imagines
    n_candidates random action sequences PER episode, all batched into one
    (B*C) tensor per horizon step, scores them, and acts on the best."""
    B = len(seeds)
    world = BatchedBallWorld(seeds)
    rng = np.random.default_rng(action_rng_seed)
    ys = np.zeros((B, real_steps), dtype=np.float32)

    for step in range(real_steps):
        frames = world.render()
        f_t = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
        z0 = model.encoder(f_t)  # (B, latent)

        action_seqs = rng.choice(ACTIONS, size=(B, n_candidates, horizon), p=[0.35, 0.2, 0.2, 0.25])
        z = z0.unsqueeze(1).expand(B, n_candidates, -1).reshape(B * n_candidates, -1).contiguous()
        actions_seq_t = torch.from_numpy(action_seqs).long().to(device)  # (B,C,horizon)

        decoded = []
        for t in range(horizon):
            a_t = actions_seq_t[:, :, t].reshape(B * n_candidates)
            z, frame = model.imagine_step(z, a_t)
            if t % decode_every == 0 or t == horizon - 1:
                pos = estimate_ball_pos_batched(frame)  # (B*C, 2)
                decoded.append(pos)

        positions = torch.stack(decoded, dim=1)  # (B*C, T, 2)
        positions = positions.view(B, n_candidates, len(decoded), 2)
        xs, ysd = positions[..., 0], positions[..., 1]
        scores = score_batched(xs, ysd)  # (B, C)
        best_idx = scores.argmax(dim=1).cpu().numpy()

        best_actions = action_seqs[np.arange(B), best_idx, 0]
        world.step(best_actions)
        ys[:, step] = world.y

    return ys


def run_random_batch(seeds, real_steps=40, action_rng_seed=999):
    B = len(seeds)
    world = BatchedBallWorld(seeds)
    rng = np.random.default_rng(action_rng_seed)
    ys = np.zeros((B, real_steps), dtype=np.float32)
    for step in range(real_steps):
        a = rng.choice(ACTIONS, size=B)
        world.step(a)
        ys[:, step] = world.y
    return ys


def run_heuristic_batch(seeds, real_steps=40):
    B = len(seeds)
    world = BatchedBallWorld(seeds)
    ys = np.zeros((B, real_steps), dtype=np.float32)
    for step in range(real_steps):
        world.step(np.full(B, 3))
        ys[:, step] = world.y
    return ys


def summarize(all_ys):
    all_ys = np.array(all_ys)
    frac_in_goal = float((all_ys <= GOAL_Y + 0.05).mean())
    min_y_mean = float(all_ys.min(axis=1).mean())
    final_y_mean = float(all_ys[:, -1].mean())
    return {"frac_steps_near_goal": frac_in_goal, "mean_min_y_reached": min_y_mean, "mean_final_y": final_y_mean}


print("\n== Planner vs baselines (20 episodes x 40 steps each, batched) ==")
t0 = time.time()
N_EVAL_EPISODES = 20
eval_seeds = [5000 + i for i in range(N_EVAL_EPISODES)]
planner_ys = run_planner_batch(model, eval_seeds)
random_ys = run_random_batch(eval_seeds)
heuristic_ys = run_heuristic_batch(eval_seeds)
print(f"planner-vs-baselines eval took {time.time()-t0:.1f}s")

comparison = {
    "planner (world-model MPC)": summarize(planner_ys),
    "random actions": summarize(random_ys),
    "always-thrust-up heuristic": summarize(heuristic_ys),
}
for name, stats in comparison.items():
    print(f"{name:30s}  near-goal={stats['frac_steps_near_goal']:.2%}  "
          f"min_y={stats['mean_min_y_reached']:.3f}  final_y={stats['mean_final_y']:.3f}")

labels = list(comparison.keys())
near_goal_vals = [comparison[k]["frac_steps_near_goal"] for k in labels]
plt.figure(figsize=(6.5, 4))
plt.bar(labels, near_goal_vals, color=["#c9591c", "#4fb4e8", "#8a8578"])
plt.ylabel("fraction of steps within goal band")
plt.title(f"Planner vs baselines ({N_EVAL_EPISODES} episodes)")
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/planner_comparison.png", dpi=130)
plt.close()

results["planner_vs_baselines"] = comparison
save_results_snapshot(results)

print("\n== Ablation: number of candidates (10 episodes each, batched) ==")
t0 = time.time()
CAND_COUNTS = [4, 8, 16, 32, 64]
N_ABL_EPISODES = 10
abl_seeds = [7000 + i for i in range(N_ABL_EPISODES)]
ablation = {}
for nc in CAND_COUNTS:
    ys = run_planner_batch(model, abl_seeds, n_candidates=nc, action_rng_seed=321 + nc)
    stats = summarize(ys)
    ablation[nc] = stats
    print(f"candidates={nc:3d}  near-goal={stats['frac_steps_near_goal']:.2%}  min_y={stats['mean_min_y_reached']:.3f}")
print(f"ablation took {time.time()-t0:.1f}s")

plt.figure(figsize=(6, 4))
plt.plot(CAND_COUNTS, [ablation[nc]["frac_steps_near_goal"] for nc in CAND_COUNTS], marker="o", color="#3a7d4f")
plt.xlabel("number of imagined candidates per step")
plt.ylabel("fraction of steps within goal band")
plt.title("Planning quality vs imagination budget")
plt.tight_layout()
plt.savefig(f"{OUT}/ablation_candidates.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# Save everything
# ----------------------------------------------------------------------
results["ablation_candidates"] = {str(k): v for k, v in ablation.items()}
save_results_snapshot(results)

print("\n== Done. Saved to /kaggle/working: world_model_kaggle.pt, loss_curves.png,")
print("   compounding_error.png, planner_comparison.png, ablation_candidates.png, results.json ==")
