"""A tiny 2D physics sandbox: a ball with gravity, walls, and a paddle-like
side-thruster the agent can fire. Renders to small RGB frames.

State: ball position (x, y) and velocity (vx, vy) in a unit box [0,1]x[0,1].
Actions: 0=noop, 1=push left, 2=push right, 3=push up (thrust)
"""
import numpy as np

IMG_SIZE = 48
GRAVITY = 0.0018
THRUST = 0.010
DAMPING = 0.995
RADIUS = 0.045
DT = 1.0

ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "noop", 1: "left", 2: "right", 3: "up"}


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

    def set_state(self, s):
        self.x, self.y, self.vx, self.vy = [float(v) for v in s]

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

        # walls: bounce
        if self.x - RADIUS < 0:
            self.x = RADIUS
            self.vx *= -0.85
        if self.x + RADIUS > 1:
            self.x = 1 - RADIUS
            self.vx *= -0.85
        if self.y - RADIUS < 0:
            self.y = RADIUS
            self.vy *= -0.85
        if self.y + RADIUS > 1:
            self.y = 1 - RADIUS
            self.vy *= -0.7

        return self.state()

    def render(self, size=IMG_SIZE):
        img = np.zeros((size, size, 3), dtype=np.float32)
        # background gradient
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

    def clone(self):
        w = BallWorld.__new__(BallWorld)
        w.rng = self.rng
        w.x, w.y, w.vx, w.vy = self.x, self.y, self.vx, self.vy
        return w
