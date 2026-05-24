"""
Reachability check -- determines, via inverse kinematics, what is and is not
kinematically feasible along the eval Lissajous.

It answers three separate questions for every point on the curve:
  1. position-only      -- can the EE reach the target POSITION at all?
  2. position + down    -- can it reach that position with the EE pointing
                           in the exact target (downward) orientation?
  3. orientation floor  -- if the position is held EXACT, what is the smallest
                           EE tilt away from the target orientation that is
                           kinematically possible? (the hard floor on the
                           orientation error, independent of the controller)

The solver is Levenberg-Marquardt (adaptive damping) with multi-seed restarts,
so it converges to the true closest configuration -- it returns ~0 residual
wherever a solution exists and the genuine minimum residual where one does not.

Note: checks kinematic reachability + joint limits, not collisions.

Drop into agent/ and run (takes ~1-2 min):
    python agent/reachability_check.py
"""
import argparse
import yaml
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from env.tracking_env import EETrackingEnv


def _ee_jacobian(env, q):
    n = env.n_joints
    env.data.qpos[:n] = q
    mujoco.mj_forward(env.model, env.data)
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jac(env.model, env.data, jacp, jacr,
                  env.data.xpos[env._ee_body_id], env._ee_body_id)
    return np.vstack([jacp, jacr])[:, :n]


def solve_ik(env, target_pos, target_R, q_init, w_pos=1.0, w_ori=1.0,
             iters=300, max_damping_tries=40):
    """Levenberg-Marquardt IK for a 6-DOF pose, clipped to joint limits.

    The error is weighted: w_pos on position, w_ori on orientation. Setting
    w_pos >> w_ori makes the solver nail position and minimise orientation in
    the remaining freedom (-> the kinematic orientation floor). Adaptive
    damping makes every accepted step reduce the error, so the solver cannot
    oscillate or diverge; it settles at the true minimum-residual config.
    Returns (q, position_residual_m, orientation_residual_rad)."""
    n = env.n_joints
    lo, hi = env.model.jnt_range[:n, 0], env.model.jnt_range[:n, 1]
    q = np.clip(np.asarray(q_init, dtype=float), lo, hi)
    lam = 1e-2

    def errors(q):
        env.data.qpos[:n] = q
        mujoco.mj_forward(env.model, env.data)
        p, R = env._get_ee_pose()
        e_p = target_pos - p
        e_r = Rotation.from_matrix(target_R @ R.T).as_rotvec()
        return e_p, e_r, np.concatenate([w_pos * e_p, w_ori * e_r])

    e_p, e_r, e = errors(q)
    for _ in range(iters):
        norm = np.linalg.norm(e)
        if norm < 1e-9:
            break
        J = _ee_jacobian(env, q)
        Jw = np.vstack([w_pos * J[:3], w_ori * J[3:]])
        for _ in range(max_damping_tries):           # adaptive-damping search
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(6), e)
            q_new = np.clip(q + dq, lo, hi)
            e_p2, e_r2, e2 = errors(q_new)
            if np.linalg.norm(e2) < norm:            # step helped -> accept
                q, e, e_p, e_r = q_new, e2, e_p2, e_r2
                lam = max(lam * 0.7, 1e-7)
                break
            lam = min(lam * 2.5, 1e3)                # step hurt -> damp more
        else:
            break                                    # cannot improve -> done
    return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


def best_of_seeds(env, target_pos, target_R, warm_q, rng, w_pos, w_ori,
                  n_random=3, key="pos"):
    """Multi-seed IK: warm-start + random restarts, keep the best result."""
    n = env.n_joints
    lo, hi = env.model.jnt_range[:n, 0], env.model.jnt_range[:n, 1]
    seeds = [warm_q] + [rng.uniform(lo, hi) for _ in range(n_random)]
    best = None
    for s in seeds:
        q, pe, oe = solve_ik(env, target_pos, target_R, s, w_pos, w_ori)
        score = pe if key == "pos" else (oe if pe < 5e-4 else pe + 1.0)
        if best is None or score < best[0]:
            best = (score, q, pe, oe)
    return best[1], best[2], best[3]


def run(config_path, plot_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = EETrackingEnv(config, eval_mode=True)
    env.reset()
    traj = env.trajectory
    down_R = traj.down_orientation
    n_steps = int(round(traj.total_duration / traj.dt))
    positions = np.array([traj.step()[0].copy() for _ in range(n_steps)])

    n = env.n_joints
    rng = np.random.default_rng(0)
    home = env.data.qpos[:n].copy()

    pos_res = np.zeros(n_steps)      # position-only residual
    down_res = np.zeros(n_steps)     # position + exact-down residual
    ori_floor = np.zeros(n_steps)    # forced EE tilt if position held exact
    wq = {"pos": home.copy(), "down": home.copy(), "floor": home.copy()}

    for i, p in enumerate(positions):
        # 1. position only
        q, pe, _ = best_of_seeds(env, p, down_R, wq["pos"], rng, 1.0, 0.0, key="pos")
        pos_res[i] = pe; wq["pos"] = q
        # 2. position + exact downward orientation (balanced)
        q, pe, _ = best_of_seeds(env, p, down_R, wq["down"], rng, 1.0, 1.0, key="pos")
        down_res[i] = pe; wq["down"] = q
        # 3. kinematic orientation floor (position prioritised 2000:1)
        q, pe, oe = best_of_seeds(env, p, down_R, wq["floor"], rng,
                                  2000.0, 1.0, key="ori")
        ori_floor[i] = oe; wq["floor"] = q

    feasible_down = int((down_res < 5e-3).sum())
    print(f"Tested {n_steps} poses on the eval Lissajous.\n")
    print(f"  every target POSITION reachable        : "
          f"{int((pos_res < 1e-3).sum())}/{n_steps}   "
          f"(worst residual {pos_res.max()*1e3:.2f} mm)")
    print(f"  reachable with EE pointing exactly down : "
          f"{feasible_down}/{n_steps}   "
          f"(worst miss {down_res.max()*1e3:.0f} mm on the rest)")
    print()
    print(f"  kinematic orientation floor (forced EE tilt if position is exact):")
    print(f"    mean over curve : {np.degrees(ori_floor.mean()):.1f} deg")
    print(f"    max  over curve : {np.degrees(ori_floor.max()):.1f} deg "
          f"(step {int(np.argmax(ori_floor))})")
    print(f"    points forced >10 deg off vertical : "
          f"{int((np.degrees(ori_floor) > 10).sum())}/{n_steps}")
    print()
    if feasible_down == n_steps:
        print("  -> downward orientation feasible everywhere; any orientation")
        print("     error is a control/reward effect.")
    else:
        print("  -> the curve leaves the region where the EE can point straight")
        print("     down. On those segments the orientation error has a hard")
        print("     kinematic floor (above) that no controller or reward can beat.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(down_res * 1e3, lw=2, color="crimson")
        ax[0].axhline(5, ls="--", c="gray")
        ax[0].set_ylabel("position miss with EE\nheld exactly down (mm)")
        ax[0].set_title("Kinematic reachability along the eval Lissajous")
        ax[0].grid(alpha=0.3)
        ax[1].plot(np.degrees(ori_floor), lw=2, color="navy")
        ax[1].set_ylabel("forced EE tilt off target\nif position held exact (deg)")
        ax[1].set_xlabel("trajectory step")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=110)
        print(f"\nSaved {plot_path}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plot", default="logs/reachability_map.png")
    args = parser.parse_args()
    run(args.config, args.plot)