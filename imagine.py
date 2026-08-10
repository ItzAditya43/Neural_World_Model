"""Use the trained world model to IMAGINE multiple candidate futures purely
in latent space (no real environment steps), score them, pick the best
first action, act for real, and repeat. Exports everything to JSON for the
web visualization.
"""
import json
import numpy as np
import torch
from env import BallWorld, ACTIONS, ACTION_NAMES, IMG_SIZE
from model import WorldModel

HORIZON = 14          # how many steps ahead the model imagines
N_CANDIDATES = 22      # random action sequences to imagine per real step
REAL_STEPS = 40        # how many real steps the agent takes
GOAL_Y = 0.12          # objective: get the ball up near the top and keep it there

device = "cuda" if torch.cuda.is_available() else "cpu"


def score_rollout(ys, xs):
    ys = np.array(ys)
    xs = np.array(xs)
    height_reward = -(ys - GOAL_Y).clip(min=0).mean() * 3.0
    overshoot_penalty = -(GOAL_Y - ys).clip(min=0).mean() * 1.5
    center_penalty = -np.abs(xs - 0.5).mean() * 0.3
    return float(height_reward + overshoot_penalty + center_penalty)


def sample_action_sequence(rng, horizon):
    return rng.choice(ACTIONS, size=horizon, p=[0.35, 0.2, 0.2, 0.25])


@torch.no_grad()
def imagine_from(model, z0, action_seq, decode_every=1):
    """Roll the world model forward in latent space, returns list of
    (x, y) decoded-ish positions approximated via a tiny probe decode,
    plus decoded frames at intervals."""
    z = z0
    frames_b64 = []
    positions = []
    for t, a in enumerate(action_seq):
        a_t = torch.tensor([a], device=device, dtype=torch.long)
        z, frame = model.imagine_step(z, a_t)
        if t % decode_every == 0 or t == len(action_seq) - 1:
            f = frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            positions.append(estimate_ball_pos(f))
    return z, positions


def estimate_ball_pos(frame):
    """Cheap centroid estimate of the bright ball from a decoded frame."""
    brightness = frame.sum(axis=-1)
    thresh = brightness > (brightness.max() * 0.75)
    if thresh.sum() == 0:
        return (0.5, 0.5)
    ys_idx, xs_idx = np.nonzero(thresh)
    h, w = frame.shape[:2]
    return (float(xs_idx.mean()) / w, float(ys_idx.mean()) / h)


def main():
    model = WorldModel().to(device)
    model.load_state_dict(torch.load("checkpoints/world_model.pt", map_location=device))
    model.eval()

    rng = np.random.default_rng(42)
    world = BallWorld(seed=123)
    world.reset()

    timeline = []

    for step in range(REAL_STEPS):
        real_frame = world.render()
        with torch.no_grad():
            f_t = torch.from_numpy(real_frame).permute(2, 0, 1).unsqueeze(0).to(device)
            z0 = model.encoder(f_t)

        candidates = []
        best_idx, best_score = -1, -1e9
        for c in range(N_CANDIDATES):
            action_seq = sample_action_sequence(rng, HORIZON)
            _, positions = imagine_from(model, z0, action_seq, decode_every=2)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            score = score_rollout(ys, xs)
            candidates.append({
                "actions": [int(a) for a in action_seq],
                "positions": [[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)],
                "score": round(score, 4),
            })
            if score > best_score:
                best_score = score
                best_idx = c

        chosen_action = int(candidates[best_idx]["actions"][0])

        timeline.append({
            "step": step,
            "real_pos": [round(world.x, 4), round(world.y, 4)],
            "chosen_action": chosen_action,
            "chosen_action_name": ACTION_NAMES[chosen_action],
            "best_candidate_idx": best_idx,
            "candidates": candidates,
        })

        world.step(chosen_action)

        if step % 5 == 0:
            print(f"step {step:3d}  pos=({world.x:.2f},{world.y:.2f})  action={ACTION_NAMES[chosen_action]:5s}  best_score={best_score:.3f}")

    with open("exports/rollout.json", "w") as f:
        json.dump({
            "horizon": HORIZON,
            "n_candidates": N_CANDIDATES,
            "goal_y": GOAL_Y,
            "action_names": ACTION_NAMES,
            "timeline": timeline,
        }, f)
    print("saved exports/rollout.json")


if __name__ == "__main__":
    main()
