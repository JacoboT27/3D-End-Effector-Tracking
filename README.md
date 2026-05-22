# EE Tracking — RL-based End-Effector Trajectory Tracking

A reinforcement learning system that trains a robotic arm to follow a moving
end-effector target through 3D space — matching both position and orientation.
A Soft Actor-Critic agent learns to control the arm directly in joint space;
no inverse-kinematics solver is used.

## How it works

A SAC agent observes the noisy state of the arm and the (clean) state of the
target, and outputs small changes to the joint angles (Δq). MuJoCo's position
actuators execute them. Over ~1M steps the agent learns the inverse kinematics
implicitly — it discovers which joint motions move the hand where it needs to go.

- **Simulator** — MuJoCo 3.x, Franka Emika Panda 7-DOF arm, wrapped as a Gymnasium environment.
- **Control** — joint position actuators. Each agent step advances physics by one full
  control period (`1 / control_freq`), so the physics clock and the trajectory clock stay in sync.
- **Agent** — SAC (stable-baselines3): off-policy, actor-critic, with a 1M-transition replay buffer.
- **Training targets** — random waypoint trajectories with minimum-jerk interpolation, re-sampled every episode.
- **Evaluation target** — a fixed Lissajous curve the agent never trains on, so the eval score measures generalization.
- **Uncertainty** — Gaussian noise on the observed end-effector pose and joint angles.

## Setup

The Docker image carries all dependencies. The robot model files are downloaded
separately.

```bash
git clone <your-repo-url>
cd 3D-End-Effector-Tracking

# 1. download robot model files (XML + meshes) into assets/
bash scripts/download_assets.sh

# 2. build the image and train
docker compose up --build train

# 3. evaluate once training is done
docker compose up evaluate
```

Training runs headless by default (`MUJOCO_GL=osmesa`). For live rendering on
Linux, see the commented X11 section in `docker-compose.yml`.

## Local setup (no Docker)

```bash
git clone <your-repo-url>
cd 3D-End-Effector-Tracking
bash scripts/download_assets.sh

# install the CPU build of PyTorch first —
# the default PyPI wheel is a multi-GB CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m agent.train --config configs/default.yaml
python -m agent.evaluate --config configs/default.yaml --model models/best/best_model
```

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
│   ├── train.py                # SAC training with parallel environments
│   └── evaluate.py             # Evaluation + trajectory plots
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
| `training.total_timesteps` | `1_000_000` | Total training steps |

## Observation space

A flat vector. `n` is the number of arm joints — 7 for Franka, 6 for UR5e — so the
total is **38** for the Franka.

| Component | Dim | Description |
|---|---|---|
| End-effector position (noisy) | 3 | Current hand XYZ |
| End-effector orientation (noisy) | 6 | 6D rotation representation |
| Target position | 3 | Current target XYZ |
| Target orientation | 6 | 6D rotation representation |
| Target linear velocity | 3 | Predictive trajectory info |
| Target angular velocity | 3 | Predictive trajectory info |
| Joint positions (noisy) | n | Current joint angles |
| Previous action | n | Last Δq, for the smoothness penalty |

## Action space

`Δq` — a delta joint-angle vector of size `n`, output in [−1, 1] and scaled to
±`max_delta_q` radians. The environment adds it to the current joint angles and
sends the result to MuJoCo's position actuators.

## Reward

Per step:

```
r = − α · ‖p_ee − p_target‖          position tracking error
  − β · ‖aₜ − aₜ₋₁‖                  smoothness penalty
  − γ · geodesic(R_ee, R_target)      orientation error
  + bonus   if ‖position error‖ < threshold
```

Weights (`α`, `β`, `γ`), the bonus threshold, and the bonus value are set under
`reward` in `configs/default.yaml`. A well-tracking policy scores near zero or
slightly positive; a poor one is strongly negative.