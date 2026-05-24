# 3D End-Effector Tracking

**RL Based Robotic Controller**

A reinforcement learning system that trains a robotic arm to follow a trajectory target through 3D space, matching both position and orientation.

A Truncated Quantile Critics (TQC) agent learns to control the arm directly in joint space. Designed for Linux hosts.

---

## How it works

A TQC agent observes the noisy state of the arm and the (clean) state of the target, and outputs small changes to the joint angles (Δq). MuJoCo's position actuators execute them. Over a few million steps the agent learns the inverse kinematics implicitly.

- **Simulator** — MuJoCo 3.x, Franka Emika Panda 7-DOF arm, wrapped as a Gymnasium environment.
- **Control** — joint position actuators driven by delta-angle commands.
- **Agent** — TQC (sb3-contrib): off-policy, distributional actor-critic, with a 400k-transition replay buffer.
- **Training targets** — randomized Lissajous curves, re-sampled every episode (random per-axis amplitude, frequency and phase).
- **Evaluation target** — a single fixed ("canonical") Lissajous curve, so the eval score is a stable, comparable benchmark across runs.
- **Uncertainty source** — Gaussian noise on the observed end-effector pose and joint angles.

---

## Results

![input](examples/training.png)

The agent learns stable sub-centimetre position tracking on the canonical Lissajous benchmark. Run `agent/evaluate.py` after training for the full position/orientation error breakdown (it reports a "settled" error that skips the startup ramp).

![input](examples/evaluation.png)

---

## Design choices

### Joint-space delta control, with implicit inverse kinematics

The agent outputs **delta joint angles** (Δq) — small per-joint changes, capped at `max_delta_q` each step. Two alternatives were rejected. *Cartesian deltas plus an IK solver* would make the RL problem trivial: the inverse-kinematics solver already does the hard geometric work, so the agent would behave only as a filter. *Direct joint-torque control* would force the agent to also learn low-level dynamics, which is outside the scope here.

The agent must discover the arm's kinematics purely from experience, while MuJoCo's built-in PD controller handles the dynamics of reaching each commanded angle. The trained policy is the task-specific inverse-kinematics controller.

### Truncated Quantile Critics (TQC)

The learner is TQC, an off-policy actor-critic algorithm. Like SAC it stores every transition in a replay buffer and reuses it for many gradient updates, which is sample-efficient — the right property when the simulator runs on CPU with 8 parallel environments. Unlike SAC, TQC represents each critic as a *distribution* over returns and truncates the top quantiles when forming the target, which controls the value-overestimation bias.

Plain SAC was tried first; on this task its critics overestimated and diverged, destabilizing training. Switching to TQC removed that failure mode and produced the stable runs.

### Lissajous trajectories, randomized for training

**Training** re-samples a fresh **randomized Lissajous curve** every episode — random per-axis amplitude, frequency and phase. Re-randomizing every episode forces the agent to learn tracking as a genuine skill rather than memorize one path.

**Evaluation** uses a single fixed **canonical Lissajous curve**. Because training already covers the same family of motion (and the same speeds), the eval curve is in-distribution, and the eval score is a stable, comparable benchmark across runs rather than a noisy out-of-distribution probe.

Both train and eval curves are placed relative to the end-effector's reset position and shifted by `curve_center_offset` (see *Workspace feasibility* below). A legacy random-waypoint mode with minimum-jerk interpolation is still available via `train_type: "waypoint"`.

### Startup ramps

At the start of each episode both the **position and orientation targets ramp smoothly onto the curve** over `ori_ramp_duration` seconds — position from the EE's reset location, orientation from its reset pose to the downward pose. Snapping the target straight onto a relocated, downward-pointing curve would otherwise force a large unavoidable error for the first ~1.5 s; the min-jerk ramps remove that startup transient.

### Workspace feasibility

The downward end-effector orientation cannot be achieved everywhere in the arm's reach: pointing straight down costs horizontal reach, so the far parts of a large curve become infeasible in 6-DOF pose even though the positions alone are reachable. `agent/reachability_check.py` (inverse-kinematics feasibility) and `agent/manipulability_check.py` (Yoshikawa manipulability) characterize this. `curve_center_offset` and `eval_amp_scale` relocate and scale the curve so it stays inside the region where the EE can point down along its whole length.

### Orientation: 6D representation and geodesic error

Orientations are encoded with the continuous 6-dimensional representation rather than quaternions or Euler angles. Quaternions have a double-cover discontinuity and Euler angles have gimbal-lock singularities; both create points where a tiny rotation causes a large jump in the encoding. The 6D form is continuous everywhere.

The orientation error in the reward is the **geodesic distance** on SO(3).

### Uncertainty model

The source of uncertainty is **Gaussian observation noise** on the measured end-effector pose and joint angles. The target trajectory is given noise-free. The observation noise forces the agent to be robust.

### Proprioceptive observations

The observation also includes the arm's own **joint velocities**. A position-controlled arm carries momentum, so position alone does not fully describe its state.

---

## Setup

The Docker image carries all dependencies. The robot model files are downloaded separately.

```bash
# 1. Clone the repository
git clone https://github.com/JacoboT27/3D-End-Effector-Tracking.git
cd 3D-End-Effector-Tracking

# 2. Run script to download robot model files (XML + meshes) into assets/
bash scripts/download_assets.sh

# 3. pull the image and train
docker compose pull
docker compose up train

# alternatively, you can build the image and train
docker compose up --build train

# 4. Obtain training curves
docker compose run --rm evaluate python -m agent.plot_training

# 5. evaluate once training is done
docker compose up evaluate

# 6. watch the trained policy in the MuJoCo viewer
xhost +local:                # once per session, grants the container the display
docker compose up viewer
```

**Warm-starting.** `train.py` accepts `--resume <model>` to continue training from an existing checkpoint instead of starting fresh — useful after a config change (e.g. a new curve placement), since the tracking skill transfers and only needs to adapt:

```bash
docker compose run --rm train python agent/train.py --config configs/default.yaml --resume models/best/best_model
```

**Workspace analysis.** The kinematic feasibility of the eval curve can be checked directly:

```bash
docker compose run --rm evaluate python agent/reachability_check.py
docker compose run --rm evaluate python agent/manipulability_check.py
```

## Local setup (no Docker)

```bash
git clone https://github.com/JacoboT27/3D-End-Effector-Tracking.git
cd 3D-End-Effector-Tracking
bash scripts/download_assets.sh

# install the CPU build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m agent.train --config configs/default.yaml
python -m agent.evaluate --config configs/default.yaml --model models/best/best_model
```
---

## Observation space

A flat vector. `n` is the number of arm joints — 7 for Franka, 6 for UR5e — so the total is **45** for the Franka.

| Component | Dim | Description |
|---|---|---|
| End-effector position (noisy) | 3 | Current hand XYZ |
| End-effector orientation (noisy) | 6 | 6D rotation representation |
| Target position | 3 | Current target XYZ |
| Target orientation | 6 | 6D rotation representation |
| Target linear velocity | 3 | Predictive trajectory info |
| Target angular velocity | 3 | Predictive trajectory info |
| Joint positions (noisy) | n | Current joint angles |
| Joint velocities | n | Current joint angular velocities |
| Previous action | n | Last Δq, for the smoothness penalty |

## Action space

`Δq` — a delta joint-angle vector of size `n`, output in [−1, 1] and scaled to ±`max_delta_q` radians. The environment adds it to the current joint angles and sends the result to MuJoCo's position actuators.

---

## Reward function

At every step the agent receives a scalar reward built from two exponential "closeness" terms — one for position, one for orientation — **multiplied together**, minus a smoothness penalty:

```
r = exp(−‖p_ee − p_target‖ / pos_scale)        position closeness    ∈ (0, 1]
  × exp(−geodesic(R_ee, R_target) / ori_scale)  orientation closeness ∈ (0, 1]
  − β · ‖aₜ − aₜ₋₁‖                             smoothness penalty
```

**Pure product, not a weighted sum.** Each closeness term is 1 when its error is zero and decays toward 0 as the error grows. Because they are *multiplied*, the agent must get **both** position and orientation right — letting either error grow drives the whole reward toward zero. This removes the corner solution a weighted sum of penalties allows, where the agent nails position and quietly ignores orientation.

**`pos_scale` / `ori_scale`** — the exponential decay scales of the two terms (metres and radians). Smaller values make the reward sharper, demanding tighter tracking before it pays out.

**Smoothness penalty (`β`)** — the magnitude of the change in action between consecutive steps, discouraging jittery commands.

**Scale.** A policy that tracks well sits near +1 per step (both closeness terms near 1, minus a small smoothness penalty); a poor one sits near 0. Watching the mean episode reward climb is the clearest sign that training is working.

The three knobs (`beta`, `pos_scale`, `ori_scale`) are set under `reward` in `configs/default.yaml`.

---

## Project structure

```
3D-End-Effector-Tracking/
├── assets/                       # created by download_assets.sh
│   ├── franka/                   # Franka Panda XML + meshes
│   └── ur5e/                     # UR5e XML + meshes
├── env/
│   ├── tracking_env.py           # Gymnasium environment — delta-q control, noisy obs, product reward
│   ├── trajectory.py             # randomized Lissajous (train) + fixed Lissajous (eval), startup ramps
│   ├── noise.py                  # Gaussian observation noise
│   └── utils.py                  # SO(3) utilities, 6D rotation representation
├── agent/
│   ├── train.py                  # TQC training (supports --resume warm-start)
│   ├── evaluate.py               # evaluation + trajectory plots
│   ├── plot_training.py          # training-curve plots from progress.csv
│   ├── visualize.py              # interactive MuJoCo viewer for a trained policy
│   ├── reachability_check.py     # IK feasibility analysis of the eval curve
│   └── manipulability_check.py   # Yoshikawa manipulability along the curve
├── scripts/
│   └── download_assets.sh        # downloads robot model files
├── configs/
│   └── default.yaml              # all hyperparameters
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Robots supported

| Key | Model | DOF |
|---|---|---|
| `franka` | Franka Emika Panda | 7 |
| `ur5` | Universal Robots UR5e | 6 |

Set the robot in `configs/default.yaml` under `env.robot`.

---

## Configuration

Key parameters in `configs/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `env.robot` | `franka` | Robot model |
| `env.control_freq` | `20` Hz | Agent control rate |
| `env.episode_steps` | `500` | Hard cap; Lissajous episodes end earlier (see `lissajous_duration`) |
| `env.max_delta_q` | `0.08` rad | Max joint-angle change per step |
| `env.track_orientation` | `true` | Enable 6-DOF tracking |
| `noise.ee_pos_std` | `0.005` m | End-effector position noise |
| `trajectory.lissajous_duration` | `10.0` s | Duration of one curve / episode |
| `trajectory.ori_ramp_duration` | `1.5` s | Startup ramp (position and orientation) onto the curve |
| `trajectory.curve_center_offset` | `[-0.10, 0.0, -0.10]` m | Shifts every curve into the down-feasible workspace |
| `trajectory.eval_amp_scale` | `0.90` | Scales the canonical eval curve |
| `reward.pos_scale` | `0.10` m | Position-reward exponential decay scale |
| `reward.ori_scale` | `0.5` rad | Orientation-reward exponential decay scale |
| `reward.beta` | `0.01` | Smoothness penalty weight |
| `training.n_envs` | `8` | Parallel environments |
| `training.total_timesteps` | `2_000_000` | Total training steps per run |

The TQC hyperparameters (learning rate `1e-4`, replay buffer `400k`, `gamma 0.98`, batch size `256`, etc.) are set directly in `agent/train.py` — they are **not** read from the config file.