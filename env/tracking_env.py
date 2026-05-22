import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os

from env.trajectory import TrajectoryGenerator
from env.noise import ObservationNoise
from env.utils import rot_mat_to_6d, rot_6d_to_mat, geodesic_distance, quat_to_rot_mat


class EETrackingEnv(gym.Env):
    """
    End-effector tracking environment using MuJoCo position control.

    The agent outputs delta joint angles (Δq). MuJoCo's position actuators
    execute them. The agent implicitly learns inverse kinematics through
    experience. Observations are corrupted with Gaussian noise.

    Observation space (flat):
        ee_pos_noisy    (3,)   noisy end-effector position
        ee_ori_6d_noisy (6,)   noisy EE orientation in 6D repr
        target_pos      (3,)   target position (no noise — it's known)
        target_ori_6d   (6,)   target orientation in 6D repr
        target_lin_vel  (3,)   target linear velocity (predictive info)
        target_ang_vel  (3,)   target angular velocity
        joint_pos_noisy (n,)   noisy joint positions
        prev_action     (n,)   previous delta-q action

    Action space:
        Δq (n_joints,)  clipped to ±max_delta_q radians per step
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict, render_mode: str = None, eval_mode: bool = False):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.eval_mode = eval_mode

        # --- load MuJoCo model ---
        robot = config["env"]["robot"]
        asset_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", f"{robot}.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(asset_path))
        self.data = mujoco.MjData(self.model)

        self.n_joints = self.model.nu  # number of actuators
        self.max_delta_q = config["env"]["max_delta_q"]
        self.episode_steps = config["env"]["episode_steps"]
        self.track_orientation = config["env"]["track_orientation"]

        # --- subsystems ---
        self.trajectory = TrajectoryGenerator(config)
        self.noise = ObservationNoise(config)

        # --- reward weights ---
        rw = config["reward"]
        self.alpha = rw["alpha"]
        self.beta = rw["beta"]
        self.gamma = rw["gamma"]
        self.bonus_threshold = rw["bonus_threshold"]
        self.bonus_value = rw["bonus_value"]

        # --- spaces ---
        obs_dim = 3 + 6 + 3 + 6 + 3 + 3 + self.n_joints + self.n_joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # --- renderer ---
        self.renderer = None
        self._step_count = 0
        self._prev_action = np.zeros(self.n_joints)
        self._current_target = None

        # find EE body id (assumes XML has a body named "ee" or "hand")
        self._ee_body_id = self._find_ee_body()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        traj_type = "lissajous" if self.eval_mode else None
        self.trajectory.reset(traj_type=traj_type)

        self._step_count = 0
        self._prev_action = np.zeros(self.n_joints)

        # warm-up: step trajectory to get initial target
        target_pos, target_lin_vel, target_rot, target_ang_vel = self.trajectory.step()
        self._current_target = (target_pos, target_lin_vel, target_rot, target_ang_vel)

        obs = self._get_obs()
        info = {}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        # scale action from [-1,1] to [-max_delta_q, max_delta_q]
        delta_q = action * self.max_delta_q

        # apply delta to current joint targets
        current_qpos = self.data.qpos[: self.n_joints].copy()
        new_qpos = np.clip(
            current_qpos + delta_q,
            self.model.jnt_range[: self.n_joints, 0],
            self.model.jnt_range[: self.n_joints, 1],
        )
        self.data.ctrl[:] = new_qpos
        mujoco.mj_step(self.model, self.data)

        # advance trajectory
        target_pos, target_lin_vel, target_rot, target_ang_vel = self.trajectory.step()
        self._current_target = (target_pos, target_lin_vel, target_rot, target_ang_vel)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        self._prev_action = action.copy()
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.episode_steps or self.trajectory.is_done()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            if self.render_mode == "human":
                self.renderer = mujoco.Renderer(self.model)
            else:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data)
        if self.render_mode == "rgb_array":
            return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        ee_pos, ee_rot = self._get_ee_pose()

        target_pos, target_lin_vel, target_rot, target_ang_vel = self._current_target

        # apply noise
        ee_pos_noisy = self.noise.apply_ee_pos(ee_pos)
        ee_ori_6d = rot_mat_to_6d(ee_rot)
        ee_ori_6d_noisy = self.noise.apply_ee_ori(ee_ori_6d)
        joint_pos_noisy = self.noise.apply_joints(self.data.qpos[: self.n_joints])

        target_ori_6d = rot_mat_to_6d(target_rot)

        obs = np.concatenate([
            ee_pos_noisy,        # (3,)
            ee_ori_6d_noisy,     # (6,)
            target_pos,          # (3,)
            target_ori_6d,       # (6,)
            target_lin_vel,      # (3,)
            target_ang_vel,      # (3,)
            joint_pos_noisy,     # (n_joints,)
            self._prev_action,   # (n_joints,)
        ])
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> float:
        ee_pos, ee_rot = self._get_ee_pose()
        target_pos, _, target_rot, _ = self._current_target

        # position tracking error
        pos_error = np.linalg.norm(ee_pos - target_pos)

        # smoothness penalty
        smoothness = np.linalg.norm(action - self._prev_action)

        reward = -self.alpha * pos_error - self.beta * smoothness

        # orientation tracking
        if self.track_orientation:
            ori_error = geodesic_distance(ee_rot, target_rot)
            reward -= self.gamma * ori_error

        # close-to-target bonus
        if pos_error < self.bonus_threshold:
            reward += self.bonus_value

        return float(reward)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        ee_pos, ee_rot = self._get_ee_pose()
        target_pos, _, target_rot, _ = self._current_target
        pos_error = np.linalg.norm(ee_pos - target_pos)
        ori_error = geodesic_distance(ee_rot, target_rot) if self.track_orientation else 0.0
        return {
            "pos_error": pos_error,
            "ori_error": ori_error,
            "step": self._step_count,
        }

    # ------------------------------------------------------------------
    # MuJoCo helpers
    # ------------------------------------------------------------------

    def _get_ee_pose(self):
        """Return current EE position (3,) and rotation matrix (3,3)."""
        pos = self.data.xpos[self._ee_body_id].copy()
        quat = self.data.xquat[self._ee_body_id].copy()  # (w, x, y, z)
        rot = quat_to_rot_mat(quat)
        return pos, rot

    def _find_ee_body(self) -> int:
        """Find end-effector body ID by checking common names."""
        candidates = ["hand", "ee", "tool", "tcp", "end_effector", "link_ee",
                      "panda_hand", "wrist_3_link"]
        for name in candidates:
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            except Exception:
                continue
        # fallback: last body in the kinematic chain
        return self.model.nbody - 1