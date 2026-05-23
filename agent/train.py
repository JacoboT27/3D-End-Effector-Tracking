import os
import yaml
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib import TQC

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


def train(config_path: str, resume_path: str = None):
    config = load_config(config_path)
    train_cfg = config["training"]

    log_dir = train_cfg["log_dir"]
    model_dir = train_cfg["model_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- training environments (randomized Lissajous curves) ---
    n_envs = train_cfg["n_envs"]
    print(f"Creating {n_envs} training environments...")
    train_env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    # --- single eval environment (fixed canonical Lissajous) ---
    eval_env = DummyVecEnv([make_env(config, 0, eval_mode=True)])
    eval_env = VecMonitor(eval_env)

    # eval_freq is counted in agent steps; divide the desired timestep
    # interval by n_envs, since the vec env advances n_envs timesteps per step
    eval_freq = max(train_cfg.get("eval_freq", 10_000) // n_envs, 1)

    # --- early stopping: halt once eval reward stops improving ---
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=train_cfg.get("early_stop_patience", 30),
        min_evals=train_cfg.get("early_stop_min_evals", 20),
        verbose=1,
    )

    # --- callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg["save_freq"] // n_envs, 1),
        save_path=model_dir,
        name_prefix="tqc_ee_tracking",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,                       # writes evaluations.npz
        eval_freq=eval_freq,
        n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
        deterministic=True,
        callback_after_eval=stop_cb,            # early stopping on plateau
        verbose=1,
    )

    # --- TQC agent ---
    # Hyperparameters are set HERE on purpose (not read from the YAML):
    # these are the values that produced the stable, working run.
    if resume_path:
        # warm-start: load an existing policy/critic and keep training.
        # used for the pos_scale-tightening polish run -- the loaded model
        # is already inside the tracking basin, so a sharper reward just
        # pulls it tighter rather than risking a cold-start failure.
        print(f"Warm-starting from: {resume_path}")
        model = TQC.load(resume_path, env=train_env)
    else:
        model = TQC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,
            buffer_size=400_000,
            batch_size=256,
            tau=0.005,
            gamma=0.98,
            learning_starts=10_000,
            gradient_steps=1,
            verbose=1,
        )

    # log to stdout + CSV + TensorBoard, all under log_dir.
    # progress.csv is what plot_training.py reads to draw the curves.
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    print("Starting training...")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        log_interval=train_cfg["log_interval"],
    )

    final_path = os.path.join(model_dir, "tqc_ee_tracking_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="path to a saved model to warm-start from, "
             "e.g. models/best/best_model",
    )
    args = parser.parse_args()
    train(args.config, resume_path=args.resume)