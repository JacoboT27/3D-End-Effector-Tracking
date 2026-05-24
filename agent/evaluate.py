import os
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write plots to file, no GUI window
import matplotlib.pyplot as plt
from sb3_contrib import TQC

from env.tracking_env import EETrackingEnv


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(config_path: str, model_path: str, n_episodes: int = 3):
    config = load_config(config_path)

    env = EETrackingEnv(config, eval_mode=True)

    # NOTE: the agent is TQC (sb3-contrib), so it must be loaded with TQC.
    model = TQC.load(model_path, env=env)
    print(f"Loaded model from {model_path}")

    # Steps to skip when reporting the "settled" error: the orientation
    # target ramps in over ori_ramp_duration, so the first chunk of every
    # episode is an intentional transient and should not dominate the metric.
    control_freq = config["env"]["control_freq"]
    ramp = config["trajectory"].get("ori_ramp_duration", 1.5)
    settle = int(ramp * control_freq) + 5

    all_pos_errors, all_ori_errors = [], []
    all_pos_settled, all_ori_settled = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_pos_errors, ep_ori_errors = [], []

        ee_positions, target_positions = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_pos_errors.append(info["pos_error"])
            ep_ori_errors.append(info["ori_error"])

            # log positions for plotting
            ee_pos, _ = env._get_ee_pose()
            target_pos = env._current_target[0]
            ee_positions.append(ee_pos.copy())
            target_positions.append(target_pos.copy())

        all_pos_errors.extend(ep_pos_errors)
        all_ori_errors.extend(ep_ori_errors)
        all_pos_settled.extend(ep_pos_errors[settle:])
        all_ori_settled.extend(ep_ori_errors[settle:])

        print(f"Episode {ep+1}: mean pos error={np.mean(ep_pos_errors):.4f}m  "
              f"mean ori error={np.mean(ep_ori_errors):.4f}rad")

        # plot trajectory for last episode
        if ep == n_episodes - 1:
            _plot_trajectory(
                np.array(ee_positions),
                np.array(target_positions),
                ep_pos_errors,
                ep_ori_errors,
            )

    print(f"\nOverall mean pos error  : {np.mean(all_pos_errors):.4f} m")
    print(f"Overall mean ori error  : {np.mean(all_ori_errors):.4f} rad")
    print(f"Settled mean pos error  : {np.mean(all_pos_settled):.4f} m   "
          f"(excludes first {settle} steps)")
    print(f"Settled mean ori error  : {np.mean(all_ori_settled):.4f} rad "
          f"(excludes first {settle} steps)")
    env.close()


def _plot_trajectory(ee_pos, target_pos, pos_errors, ori_errors):
    fig = plt.figure(figsize=(14, 5))

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
             "b--", linewidth=1.5, label="Target (Lissajous)")
    ax1.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2],
             "r-", linewidth=1.5, label="EE actual")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # position error over time
    ax2 = fig.add_subplot(132)
    ax2.plot(pos_errors, "r-", linewidth=1)
    ax2.axhline(np.mean(pos_errors), color="k", linestyle="--",
                label=f"Mean={np.mean(pos_errors):.4f}m")
    ax2.set_title("Position Tracking Error")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Error (m)")
    ax2.legend()

    # orientation error over time
    ax3 = fig.add_subplot(133)
    ax3.plot(ori_errors, "b-", linewidth=1)
    ax3.axhline(np.mean(ori_errors), color="k", linestyle="--",
                label=f"Mean={np.mean(ori_errors):.4f}rad")
    ax3.set_title("Orientation Tracking Error")
    ax3.set_xlabel("Step"); ax3.set_ylabel("Error (rad)")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/eval_trajectory.png", dpi=150)
    print("Saved trajectory plot to logs/eval_trajectory.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="models/best/best_model")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    evaluate(args.config, args.model, args.episodes)