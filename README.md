# Neural World Model — Imagination Engine

A small, from-scratch implementation of the core idea behind model-based
reinforcement learning: instead of training a network to answer *"what
action should I take?"*, train it to answer *"what happens if I do this?"*
— then use that learned simulator to imagine several possible futures
before acting, and pick the one that looks best.

```
Current State  +  Action
        ↓
  ┌──────────────┐
  │  WORLD MODEL  │
  └──────┬────────┘
         ↓
  Predicted Future
```

## Why

Most introductory RL projects train a policy directly: state in, action
out, learned end-to-end from reward. That works, but it collapses two
separate problems into one — *understanding how the world behaves* and
*deciding what to do about it*. This project keeps them separate on
purpose:

1. Learn a **world model**: a neural network that predicts the next
   state of an environment given the current state and an action,
   trained purely on transitions, with no notion of reward at all.
2. Use that world model as a **mental simulator**: at decision time,
   roll it forward under many different candidate action sequences
   *without touching the real environment*, score each imagined
   outcome, and act on the first step of whichever imagined future
   scored highest.

This is a miniature version of the planning loop inside model-based RL
systems (e.g. Dreamer, PlaNet, MuZero's learned dynamics function): learn
a cheap, differentiable simulator, then plan inside it instead of the
real, expensive environment.

## Environment

A small 2D physics sandbox (`env.py`): a ball subject to gravity and wall
collisions, rendered as a 48×48 RGB frame. Four discrete actions are
available: `noop`, `push left`, `push right`, `thrust up`. This was
chosen over a pixel-perfect real game because:

- dynamics are continuous and non-trivial (gravity, damped bounces off
  four walls) — enough to make prediction a real learning problem
- it renders fast (pure NumPy, no external renderer), so tens of
  thousands of transitions can be generated in seconds
- the resulting frames are small enough that a lightweight CNN world
  model trains in minutes, even on a laptop GPU

## Model

`model.py` implements the world model as three small networks trained
jointly:

- **Encoder** — a 3-layer CNN that compresses a 48×48×3 frame into a
  32-dimensional latent vector.
- **Transition** — an MLP that takes the current latent vector plus an
  embedded action and predicts the *change* in latent state (a residual
  update: `z_next = z + f(z, action)`). This is the actual "world
  model" — everything else exists to train and inspect it.
- **Decoder** — a transposed-CNN mirror of the encoder that reconstructs
  a predicted next frame from the predicted next latent vector, purely
  so predictions can be visualized and scored in pixel/position space.

Training (`train.py`) is supervised: given `(frame, action, next_frame)`
triples collected under a random policy (`collect_data.py`, 24,000
transitions), minimize MSE between the decoded prediction and the true
next frame. No reward, no policy gradient — the model only ever learns
environment dynamics.

Trained on an NVIDIA RTX 3050 (CUDA), 15 epochs over 24k transitions:
reconstruction MSE loss dropped from **0.0185 → 0.0015**, i.e. the model
converges to visually accurate one-step predictions of ball position and
motion within a few epochs.

## Imagination and planning

`imagine.py` is the part that turns a trained world model into a
decision-maker, using a **random-shooting Model Predictive Control**
scheme:

1. Encode the real current frame into a latent vector.
2. Sample 22 random 14-step action sequences.
3. For each sequence, roll the *world model itself* forward
   latent-to-latent (no real environment involved), decoding frames
   periodically to estimate the ball's imagined trajectory.
4. Score each imagined trajectory against a simple objective — reach and
   hold a target height near the top of the frame, stay centered.
5. Take only the **first action** of the highest-scoring imagined
   sequence, step the real environment once, and repeat the whole
   process from the new real state (standard receding-horizon control).

This loop runs for 40 real steps per episode. The agent never sees a
reward signal from the real environment during planning — every
candidate future it compares is purely a hallucination produced by the
learned dynamics model.

## What I found

- A world model trained only on random-policy transitions is enough to
  support meaningful planning — the agent reliably steers the ball
  toward the target band near the top of the frame despite having no
  policy training at all, purely by imagining and comparing futures.
- Prediction quality degrades visibly the further the model is rolled
  forward without re-encoding from a real frame (compounding error,
  a well-known failure mode of learned dynamics models) — this is
  visible in the visualization as candidate trajectories that fan out
  and occasionally become physically implausible late in the 14-step
  horizon. Re-encoding from the real frame every real step (rather than
  ever trusting the model's own rollout as ground truth) keeps this in
  check.
- Random-shooting planning is a strong, almost embarrassingly simple
  baseline: with a good enough world model, you don't need a learned
  policy at all — sampling and re-scoring candidate action sequences
  every step is enough to produce goal-directed behavior.

## Large-scale quantitative study

The findings above come from the small demo run. A separate, much larger study — 180,000
transitions, 30 training epochs, run on a Kaggle GPU (Tesla P100) — adds real quantitative
evidence on top of it: a measured compounding-error curve, a planner-vs-baselines comparison,
and an ablation over imagination budget. Full writeup, plots, and raw numbers:
**[`kaggle_study/FINDINGS.md`](kaggle_study/FINDINGS.md)**.

Headline results:

- Reconstruction loss converges cleanly (0.0143 → 0.00147 over 30 epochs, train ≈ val, no overfitting) — after fixing a dead-ReLU training collapse the first attempt hit, documented in the report.
- Compounding error is small through ~4 imagined steps, rises steeply through step 8, and saturates by step ~11 — the expected shape for a learned dynamics model rolled forward without correction.
- The planner beats random action selection (30.8% vs 23.9% of steps within the goal band) but **loses to a trivial hand-coded heuristic** (77.1%) — a genuine negative result showing a correctly-trained world model isn't sufficient on its own; random-shooting MPC with no value function or policy prior is a weak planner when a strong simple prior exists that it can't represent.

## Visualization

`exports/viewer.html` is a self-contained interactive page (open it
directly in a browser, no server needed) that replays a full imagined
episode: the ball's real trajectory, every imagined candidate future as
a translucent branch, and the chosen plan highlighted, stepped or played
back with live score readouts.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or, for CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

python collect_data.py   # generates data/transitions.npz
python train.py          # trains checkpoints/world_model.pt (requires a CUDA GPU)
python imagine.py        # runs the imagination/planning loop, writes exports/rollout.json
```

Then open `exports/viewer.html` in a browser. It already ships with a
pre-computed rollout embedded, so it works standalone without running
the pipeline first.

## Project layout

```
env.py            2D physics sandbox: ball + gravity + walls, renders 48x48 frames
collect_data.py    generates random-policy transition dataset
model.py           encoder / latent transition / decoder — the world model itself
train.py           supervised training of the world model (GPU required)
imagine.py         random-shooting planning loop: imagine futures, pick the best, act
exports/viewer.html   interactive visualization of a full imagined episode
```

## Limitations / future work

- The world model is only ever trained on random-policy data; it has
  never seen states the *planned* agent visits, so distribution shift
  between training and deployment is unaddressed (the standard fix is
  iterative data aggregation — retrain on states the planner actually
  visits, à la DAgger).
- Planning is random-shooting, not gradient-based or learned (no value
  function, no policy network) — effective here because the action
  space is tiny (4 actions) and the horizon is short.
- The reward/objective used to score imagined futures is hand-written,
  not learned — replacing it with a learned reward model would make this
  closer to a full model-based RL system.
