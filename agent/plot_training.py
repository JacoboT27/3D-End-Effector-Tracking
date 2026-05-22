import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: write a PNG, never open a window
import matplotlib.pyplot as plt


def read_eval_npz(log_dir):
    """Eval reward over timesteps, from EvalCallback's evaluations.npz."""
    path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]              # shape (n_evals, n_eval_episodes)
    return timesteps, results.mean(axis=1), results.std(axis=1)


def read_progress_csv(log_dir):
    """All rows of SB3's progress.csv as dicts (or None if absent)."""
    path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def column(rows, x_key, y_key):
    """Extract (x, y) arrays for rows where y_key is present and numeric."""
    xs, ys = [], []
    for row in rows:
        xv, yv = row.get(x_key, ""), row.get(y_key, "")
        if xv == "" or yv == "":
            continue
        try:
            xs.append(float(xv))
            ys.append(float(yv))
        except ValueError:
            continue
    return np.array(xs), np.array(ys)


def plot_training(log_dir):
    evals = read_eval_npz(log_dir)
    rows = read_progress_csv(log_dir)

    panels = (["eval"] if evals is not None else []) + \
             (["train_reward", "losses"] if rows is not None else [])
    if not panels:
        print(f"No logs found in {log_dir} (need evaluations.npz or progress.csv).")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "eval":
            t, m, s = evals
            ax.plot(t, m, color="tab:blue", label="eval mean reward")
            ax.fill_between(t, m - s, m + s, color="tab:blue", alpha=0.2)
            ax.set_title("Evaluation reward (Lissajous)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Episode reward")
            ax.legend()

        elif panel == "train_reward":
            x, y = column(rows, "time/total_timesteps", "rollout/ep_rew_mean")
            ax.plot(x, y, color="tab:green", label="train ep_rew_mean")
            ax.set_title("Training reward")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Episode reward")
            ax.legend()

        elif panel == "losses":
            # critic and actor losses live on very different scales -> twin axis
            xc, yc = column(rows, "time/total_timesteps", "train/critic_loss")
            xa, ya = column(rows, "time/total_timesteps", "train/actor_loss")
            ax.set_title("Training losses")
            ax.set_xlabel("Timesteps")
            if len(xc):
                ax.plot(xc, yc, color="tab:red", label="critic loss")
                ax.set_ylabel("Critic loss", color="tab:red")
                ax.tick_params(axis="y", labelcolor="tab:red")
            if len(xa):
                ax2 = ax.twinx()
                ax2.plot(xa, ya, color="tab:orange", label="actor loss")
                ax2.set_ylabel("Actor loss", color="tab:orange")
                ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    out = os.path.join(log_dir, "training_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()
    plot_training(args.logdir)