"""
Manipulability check — tests whether tracking-error spikes coincide with
near-singular arm configurations.

Rolls out one evaluation episode and, at every step, computes the Franka's
end-effector manipulability (Yoshikawa index, sqrt(det(J J^T))) alongside the
position and orientation tracking error. If manipulability dips toward zero
exactly where the error spikes, the eval trajectory is passing through
kinematically hard / near-singular configurations -- i.e. the residual error
is a workspace-geometry limit, not a controller deficiency.

Drop this file into agent/ and run:
    python agent/manipulability_check.py
    python agent/manipulability_check.py --model models/best/best_model
"""
import argparse
import yaml
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")  # headless: write the plot to file
import matplotlib.pyplot as plt
from sb3_contrib import TQC

from env.tracking_env import EETrackingEnv


def ee_jacobian(env):
    """6 x n_joints end-effector Jacobian at the current sim state.
    Rows 0-2 are translational, rows 3-5 are rotational."""
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jac(env.model, env.data, jacp, jacr,
                  env.data.xpos[env._ee_body_id], env._ee_body_id)
    return np.vstack([jacp, jacr])[:, :env.n_joints]


def manipulability(J):
    """Yoshikawa manipulability index: sqrt(det(J J^T)).
    Approaches zero as the configuration approaches a singularity.
    max(...,0) guards against tiny negative determinants from round-off."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def run(config_path, model_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = EETrackingEnv(config, eval_mode=True)
    model = TQC.load(model_path, env=env)
    print(f"Loaded {model_path}")

    pos_err, ori_err = [], []
    manip_full, manip_rot = [], []

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        J = ee_jacobian(env)                          # 6 x n_joints
        manip_full.append(manipulability(J))          # full 6-DOF
        manip_rot.append(manipulability(J[3:, :]))    # rotation rows only

        pos_err.append(info["pos_error"])
        ori_err.append(info["ori_error"])

    env.close()

    steps = np.arange(len(pos_err))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    top.plot(steps, pos_err, "r-", label="position error (m)")
    top.plot(steps, ori_err, "b-", label="orientation error (rad)")
    top.set_ylabel("Tracking error")
    top.set_title("Tracking error vs. end-effector manipulability")
    top.legend(loc="upper left")
    top.grid(alpha=0.3)

    bot.plot(steps, manip_full, "k-", label="full 6-DOF manipulability")
    bot.plot(steps, manip_rot, "g-", label="rotational manipulability")
    bot.set_xlabel("Step")
    bot.set_ylabel("Manipulability")
    bot.legend(loc="upper left")
    bot.grid(alpha=0.3)

    fig.tight_layout()
    out = "logs/manipulability_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved {out}")
    print(f"  full manipulability : min={min(manip_full):.5f}  max={max(manip_full):.5f}")
    print(f"  rot. manipulability : min={min(manip_rot):.5f}  max={max(manip_rot):.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="models/best/best_model")
    args = p.parse_args()
    run(args.config, args.model)