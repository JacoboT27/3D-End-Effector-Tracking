# 3D End-Effector Tracking

**RL Based Robotic Controller**

A reinforcement learning system that trains a robotic arm to follow a trajectory target through 3D space, matching both position and orientation.

A Soft Actor-Critic agent learns to control the arm directly in joint space. Designed for Linux hosts

---

## How it works

A SAC agent observes the noisy state of the arm and the (clean) state of the target, and outputs small changes to the joint angles (Δq). MuJoCo's position actuators execute them. Over ~1M steps the agent learns the inverse kinematics implicitly.

- **Simulator** — MuJoCo 3.x, Franka Emika Panda 7-DOF arm, wrapped as a Gymnasium environment.
- **Control** — joint position actuators driven by delta-angle commands.
- **Agent** — SAC (stable-baselines3): off-policy, actor-critic, with a 1M-transition replay buffer.
- **Training targets** — random waypoint trajectories with minimum-jerk interpolation, re-sampled every episode.
- **Evaluation target** — a fixed Lissajous curve the agent never trains on, so the eval score measures generalization.
- **Uncertainty source** — Gaussian noise on the observed end-effector pose and joint angles.

---

## Design choices

### Joint-space delta control, with implicit inverse kinematics

The agent outputs **delta joint angles** (Δq) — small per-joint changes, capped at `max_delta_q` each step. Two alternatives were rejected. *Cartesian deltas plus an IK solver* would make the RL problem trivial: the inverse-kinematics solver already does the hard geometric work, so the agent would behave only as a filter. *Direct joint-torque control* would force the agent to also learn low-level dynamics, which is outside the scope here.

Delta joint angles sit in between. The agent must discover the arm's kinematics purely from experience, while MuJoCo's built-in PD controller handles the dynamics of reaching each commanded angle. The trained policy is the task-specific inverse-kinematics controller.

### Orientation: 6D representation and geodesic error

Orientations are encoded with the continuous 6-dimensional representation rather than quaternions or Euler angles. Quaternions have a double-cover discontinuity and Euler angles have gimbal-lock singularities, both create points where a tiny rotation causes a large jump in the encoding. The 6D form is continuous everywhere.

The orientation error in the reward is the **geodesic distance** on SO(3)

### Training vs evaluation trajectories

**Training** re-samples a fresh trajectory every episode: random waypoints inside the workspace sphere, joined by **minimum-jerk interpolation**. Re-randomizing every episode forces the agent to learn tracking as a skill rather than memorize a path. 

**Evaluation** uses a single fixed **Lissajous curve** the agent never trains on; because it is out-of-distribution, the evaluation score measures genuine generalization rather than recall.

### Soft Actor-Critic

The learner is SAC, an off-policy actor-critic algorithm. Off-policy means every transition is stored in a replay buffer and reused for many gradient updates, making it sample-efficient. I considered this the right choice because the simulator runs on CPU with 4 parallel environments. SAC also maximizes policy entropy alongside reward, keeping exploration alive and training stable.

### Uncertainty model

The source of uncertainty is **Gaussian observation noise** on the measured end-effector pose and joint angles. The target trajectory is given noise-free. The observation noise force the agent to be robust. 

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

# 4. evaluate once training is done
docker compose up evaluate

# 4. watch the trained policy in the MuJoCo viewer
xhost +local:                # once per session, grants the container the display
docker compose up viewer
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

At every step the agent receives a scalar reward built from four terms:

```
r = − α · ‖p_ee − p_target‖          position tracking error
  − β · ‖aₜ − aₜ₋₁‖                  smoothness penalty
  − γ · geodesic(R_ee, R_target)      orientation error
  + bonus   if ‖position error‖ < threshold
```

**Position error** (`α = 1.0`) — the Euclidean distance, in metres, between the hand and the target. This is the primary objective and the dominant term. It is used as a plain distance rather than a squared distance on purpose: squaring over-weights large mistakes and flattens out near zero, leaving the agent with almost no gradient once it is roughly close. A linear distance keeps a steady pull toward the target at every scale.

**Smoothness penalty** (`β = 0.1`) — the magnitude of the change in action between consecutive steps. Without it, a policy can chase the target with rapid, oscillating joint commands that look fine on a tracking plot but would be jerky and hard on real hardware. The small weight lets this shape the *style* of motion without overriding the tracking objective.

**Orientation error** (`γ = 0.5`) — the geodesic distance between the hand's rotation and the target rotation, in radians: zero when aligned, up to π when fully opposed. This is what makes the task full 6-DOF rather than position-only. It is weighted below the position term, so the agent prioritizes getting the hand to the right place while still aligning it.

**Close-to-target bonus** (`+0.5` within `threshold = 0.02 m`) — a small positive reward whenever the hand is within 2 cm of the target. Every other term is a penalty, so without this the best achievable score is zero. The bonus gives the agent an explicit positive signal — a target worth committing to — and pushes the policy toward *precise* tracking instead of settling for "roughly close."

**Scale.** A policy that tracks well sits near +0.4 per step (the bonus firing, minus small penalties); a poor one is strongly negative. Watching the mean episode reward climb toward zero — and the bonus begin to fire — is the clearest sign that training is working.

All five knobs (`α`, `β`, `γ`, `bonus_threshold`, `bonus_value`) are set under `reward` in `configs/default.yaml`.

---

## Project structure

```
3D-End-Effector-Tracking/
├── assets/                     # created by download_assets.sh
│   ├── franka/                 # Franka Panda XML + meshes
│   └── ur5e/                   # UR5e XML + meshes
├── env/
│   ├── tracking_env.py         # Gymnasium environment — delta-q control, noisy obs, 6-DOF reward
│   ├── trajectory.py           # Waypoint (train) + Lissajous (eval) generators
│   ├── noise.py                # Gaussian observation noise
│   └── utils.py                # SO(3) utilities, 6D rotation representation
├── agent/
│   ├── train.py                # SAC training
│   ├── evaluate.py             # Evaluation + trajectory plots
│   └── visualize.py            # Interactive MuJoCo viewer for a trained policy
├── scripts/
│   └── download_assets.sh      # downloads robot model files
├── configs/
│   └── default.yaml            # all hyperparameters
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
| `env.episode_steps` | `500` | Max steps per episode |
| `env.max_delta_q` | `0.05` rad | Max joint-angle change per step |
| `env.track_orientation` | `true` | Enable 6-DOF tracking |
| `noise.ee_pos_std` | `0.005` m | End-effector position noise |
| `trajectory.n_waypoints` | `6` | Waypoints per training trajectory |
| `training.n_envs` | `8` | Parallel environments |
| `training.gradient_steps` | `8` | SAC gradient updates per environment step |
| `training.total_timesteps` | `1_000_000` | Total training steps |
