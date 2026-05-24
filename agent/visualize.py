"""
Visualize a trained policy in an interactive MuJoCo viewer window.

Runs the trained TQC policy on an evaluation trajectory (Lissajous or circle)
and opens the MuJoCo viewer. A red sphere marks the current target; an amber
trail shows the target path and a blue trail shows the end-effector path.

Designed to run from the Docker `viewer` service (Linux + X11 forwarding):

    xhost +local:                  # once, on the host
    docker compose up --build viewer

    # or, to view the circle policy on the circle:
    docker compose run --rm viewer python -m agent.visualize --trajectory circle

Press Esc or close the window to quit.
"""

import time
import argparse
from collections import deque

import numpy as np
import yaml
import mujoco
import mujoco.viewer
from sb3_contrib import TQC

from env.tracking_env import EETrackingEnv


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def add_sphere(scn, pos, radius, rgba):
    """Append a sphere geom to a viewer scene, if there is room."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--trajectory", type=str, default=None,
        choices=["lissajous", "circle"],
        help="curve family to view (overrides eval_type in the config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="path to the model; defaults to models/best/best_model, or "
             "models/best/best_model_circular when --trajectory circle",
    )
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier (1.0 = real time)")
    parser.add_argument("--trail", type=int, default=120,
                        help="number of points kept in each path trail")
    args = parser.parse_args()

    # resolve the model path the same way evaluate.py does, so the viewer
    # loads the model that matches the trajectory it is about to draw
    model_path = args.model
    if model_path is None:
        model_path = ("models/best/best_model_circular"
                      if args.trajectory == "circle"
                      else "models/best/best_model")

    config = load_config(args.config)

    # optional curve-family override (--trajectory): without this the viewer
    # always builds the config's eval_type curve, so a circle policy would be
    # shown tracking a Lissajous
    if args.trajectory is not None:
        config["trajectory"]["eval_type"] = args.trajectory

    dt = 1.0 / config["env"]["control_freq"]

    # eval_mode=True selects the evaluation trajectory (eval_type)
    env = EETrackingEnv(config, eval_mode=True)

    # NOTE: the agent is TQC (sb3-contrib). Loading a TQC checkpoint with
    # stable_baselines3.SAC is incorrect -- it must be loaded with TQC.
    policy = TQC.load(model_path)
    print(f"Loaded model from {model_path}")

    obs, _ = env.reset()

    target_trail = deque(maxlen=args.trail)
    ee_trail = deque(maxlen=args.trail)
    errors = []
    episode = 1

    print("Launching viewer - close the window or press Esc to quit.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # frame the workspace
        viewer.cam.lookat[:] = config["trajectory"]["workspace_center"]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            step_start = time.time()

            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            target_pos = np.array(env._current_target[0])
            ee_pos, _ = env._get_ee_pose()
            target_trail.append(target_pos)
            ee_trail.append(np.array(ee_pos))
            errors.append(info["pos_error"])

            scn = viewer.user_scn
            scn.ngeom = 0
            for p in target_trail:
                add_sphere(scn, p, 0.006, [0.95, 0.6, 0.1, 0.55])   # target path
            for p in ee_trail:
                add_sphere(scn, p, 0.006, [0.2, 0.6, 0.95, 0.55])   # end-effector path
            add_sphere(scn, target_pos, 0.02, [0.95, 0.2, 0.2, 1.0])  # current target

            viewer.sync()

            if terminated or truncated:
                print(f"episode {episode}: mean position error = "
                      f"{np.mean(errors) * 100:.1f} cm")
                obs, _ = env.reset()
                target_trail.clear()
                ee_trail.clear()
                errors.clear()
                episode += 1

            sleep = dt / args.speed - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()