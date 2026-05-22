import os
import yaml
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from env.tracking_env import EETrackingEnv


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(config: dict, rank: int, eval_mode: bool = False):
    """Factory for creating environment instances."""
    def _init():
        env = EETrackingEnv(config, eval_mode=eval_mode)
        env = Monitor(env)
        return env
    return _init


def train(config_path: str):
    config = load_config(config_path)
    train_cfg = config["training"]

    os.makedirs(train_cfg["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["model_dir"], exist_ok=True)

    # --- parallel training environments ---
    n_envs = train_cfg["n_envs"]
    print(f"Spawning {n_envs} parallel training environments...")
    train_env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    # --- single eval environment (Lissajous trajectory) ---
    eval_env = DummyVecEnv([make_env(config, 0, eval_mode=True)])
    eval_env = VecMonitor(eval_env)

    # --- callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg["save_freq"] // n_envs, 1),
        save_path=train_cfg["model_dir"],
        name_prefix="sac_ee_tracking",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(train_cfg["model_dir"], "best"),
        log_path=train_cfg["log_dir"],
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # --- SAC agent ---
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=train_cfg["learning_rate"],
        buffer_size=train_cfg["buffer_size"],
        batch_size=train_cfg["batch_size"],
        tau=train_cfg["tau"],
        gamma=train_cfg["gamma"],
        learning_starts=train_cfg["learning_starts"],
        verbose=1,
        tensorboard_log=train_cfg["log_dir"],
    )

    print("Starting training...")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        log_interval=train_cfg["log_interval"],
    )

    final_path = os.path.join(train_cfg["model_dir"], "sac_ee_tracking_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)