"""Collect (frame, action, next_frame) transitions from BallWorld using a
random policy, save as a single .npz dataset."""
import numpy as np
from env import BallWorld, ACTIONS, IMG_SIZE

N_EPISODES = 400
STEPS_PER_EP = 60


def collect():
    frames, actions, next_frames = [], [], []
    rng = np.random.default_rng(0)
    for ep in range(N_EPISODES):
        world = BallWorld(seed=ep)
        world.reset()
        for t in range(STEPS_PER_EP):
            frame = world.render()
            a = int(rng.choice(ACTIONS, p=[0.4, 0.2, 0.2, 0.2]))
            world.step(a)
            next_frame = world.render()
            frames.append(frame)
            actions.append(a)
            next_frames.append(next_frame)
    frames = np.stack(frames).astype(np.float32)
    next_frames = np.stack(next_frames).astype(np.float32)
    actions = np.array(actions, dtype=np.int64)
    print("dataset:", frames.shape, actions.shape)
    np.savez_compressed("data/transitions.npz", frames=frames, actions=actions, next_frames=next_frames)


if __name__ == "__main__":
    collect()
