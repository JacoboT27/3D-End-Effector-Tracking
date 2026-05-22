"""
Visualize a trained policy in the MuJoCo viewer.

Opens an interactive simulator window and runs the SAC policy on the
Lissajous evaluation trajectory. A red sphere marks the current target;
an amber trail shows the target path and a blue trail shows the
end-effector path, so you can watch the arm trace the curve.

Run locally (needs a display — not inside the headless Docker container):

    python -m agent.visualize --model models/best/best_model

Optional flags:
    --speed 0.5     play at half speed to inspect tracking closely
    --trail 200     keep a longer path trail

Press Esc or close the window to quit.
"""

import time
import argparse
from collections import deque

import numpy as np
import yaml
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from env.tracking_env import EETrackingEnv


def load_config(path):
    with open(path, "r") as f:
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
    parser.add_argument("--model", default="models/best/best_model")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier (1.0 = real time)")
    parser.add_argument("--trail", type=int, default=120,
                        help="number of points kept in each path trail")
    args = parser.parse_args()

    config = load_config(args.config)
    dt = 1.0 / config["env"]["control_freq"]

    # eval_mode=True selects the Lissajous trajectory
    env = EETrackingEnv(config, eval_mode=True)
    policy = SAC.load(args.model)

    obs, _ = env.reset()

    target_trail = deque(maxlen=args.trail)
    ee_trail = deque(maxlen=args.trail)
    errors = []
    episode = 1

    print("Launching viewer — close the window or press Esc to quit.")

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

            # current target + end-effector positions
            target_pos = np.array(env._current_target[0])
            ee_pos, _ = env._get_ee_pose()
            target_trail.append(target_pos)
            ee_trail.append(np.array(ee_pos))
            errors.append(info["pos_error"])

            # redraw markers each frame
            scn = viewer.user_scn
            scn.ngeom = 0
            for p in target_trail:
                add_sphere(scn, p, 0.006, [0.95, 0.6, 0.1, 0.55])   # target path
            for p in ee_trail:
                add_sphere(scn, p, 0.006, [0.2, 0.6, 0.95, 0.55])   # end-effector path
            add_sphere(scn, target_pos, 0.02, [0.95, 0.2, 0.2, 1.0])  # current target

            viewer.sync()

            if terminated or truncated:
                mean_err = np.mean(errors) * 100.0
                print(f"episode {episode}: mean position error = {mean_err:.1f} cm")
                obs, _ = env.reset()
                target_trail.clear()
                ee_trail.clear()
                errors.clear()
                episode += 1

            # real-time pacing
            sleep = dt / args.speed - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()