# EE Tracking — RL-based End-Effector Trajectory Tracking

A reinforcement learning system for controlling a robotic arm to track a desired end-effector trajectory in 3D space, including orientation.

## Approach

- **Agent:** SAC (Soft Actor-Critic) — outputs delta joint angles (Δq)
- **Control:** MuJoCo position actuators execute the delta commands
- **Learning:** The agent implicitly learns inverse kinematics through experience
- **Uncertainty:** Gaussian noise on EE position, orientation, and joint observations
- **Training:** Random waypoint trajectories (minimum-jerk interpolation)
- **Evaluation:** Lissajous curve — tests generalization to unseen trajectory

## Setup

### Local

```bash
# 1. pull robot models
git submodule update --init

# 2. install dependencies
pip install -r requirements.txt

# 3. train
python agent/train.py --config configs/default.yaml

# 4. evaluate
python agent/evaluate.py --model models/best/best_model
```

### Docker

```bash
# pull robot models first (required before building image)
git submodule update --init

# train
docker-compose up train

# evaluate
docker-compose up evaluate
```

> **Note on rendering:** Training runs headless by default (`MUJOCO_GL=osmesa`). For live rendering on Linux, see the commented section in `docker-compose.yml` for X11 forwarding.

## Project Structure

```
ee-tracking/
├── mujoco_menagerie/           # git submodule — robot MJCF models
├── env/
│   ├── tracking_env.py         # Gymnasium environment
│   ├── trajectory.py           # Waypoint + Lissajous generators
│   ├── noise.py                # Gaussian observation noise
│   └── utils.py                # SO(3) utilities, 6D rotation repr
├── agent/
│   ├── train.py                # SAC training with parallel envs
│   └── evaluate.py             # Evaluation + trajectory plots
├── configs/
│   └── default.yaml            # All hyperparameters
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Robots Supported

| Key | Model |
|---|---|
| `franka` | Franka Emika Panda (7-DOF) |
| `ur5` | Universal Robots UR5e (6-DOF) |

Switch robot in `configs/default.yaml` under `env.robot`.

## Configuration

All parameters are in `configs/default.yaml`. Key options:

| Parameter | Default | Description |
|---|---|---|
| `env.robot` | `franka` | Robot model |
| `env.track_orientation` | `true` | Enable 6-DOF tracking |
| `env.max_delta_q` | `0.05` rad | Max joint angle change per step |
| `noise.ee_pos_std` | `0.005` m | EE position noise |
| `training.n_envs` | `8` | Parallel training environments |
| `training.total_timesteps` | `1,000,000` | Total training steps |

## Observation Space

| Component | Dim | Description |
|---|---|---|
| EE position (noisy) | 3 | Current end-effector XYZ |
| EE orientation 6D (noisy) | 6 | Zhou et al. 2019 representation |
| Target position | 3 | Current trajectory target XYZ |
| Target orientation 6D | 6 | Target rotation |
| Target linear velocity | 3 | Predictive trajectory info |
| Target angular velocity | 3 | Predictive trajectory info |
| Joint positions (noisy) | n | Current joint angles |
| Previous action | n | Last delta-q for smoothness |

## Reward

```
r = -α‖p_ee - p_target‖            position tracking error
  - β‖aₜ - aₜ₋₁‖                   smoothness penalty
  - γ · geodesic(R_ee, R_target)    orientation error
  + bonus  if ‖error‖ < threshold
```