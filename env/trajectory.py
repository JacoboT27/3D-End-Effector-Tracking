import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryGenerator:
    """
    Generates end-effector trajectories for training and evaluation.

    Training:   randomized Lissajous curves -- a fresh curve every episode,
                with random per-axis amplitude, frequency and phase.
    Evaluation: a single fixed ("canonical") Lissajous curve, so the eval
                score is a stable, comparable benchmark across runs.

    Training on randomized Lissajous curves (rather than slow random
    waypoints) closes the train/eval gap: the agent practices the same
    family of motion -- and the same speeds -- it is evaluated on, while
    the per-episode randomization keeps it a genuine tracking skill rather
    than a memorized path.

    Both train and eval curves are placed relative to the end-effector's
    position at reset, shifted by `curve_center_offset` (which relocates the
    curve into the workspace region where the EE can actually point down).

    The position AND orientation targets ramp smoothly onto the curve over
    `ori_ramp_duration` seconds at the start of each episode -- position from
    the EE's reset location, orientation from its reset pose to the fixed
    downward pose. This keeps every episode starting on-target with no
    startup transient, even when the curve is offset from the reset pose.

    Every step returns:
        (position, linear_velocity, orientation_matrix, angular_velocity)
    """

    # Canonical (evaluation) Lissajous amplitude per axis, as a fraction of
    # workspace_radius. Training amplitudes are randomized instead.
    _LIS_AMP = np.array([0.8, 0.8, 0.4])

    def __init__(self, config):
        traj_cfg = config["trajectory"]
        self.traj_type = traj_cfg["train_type"]
        self.n_waypoints = traj_cfg["n_waypoints"]
        self.interp_duration = traj_cfg["interp_duration"]
        self.center = np.array(traj_cfg["workspace_center"], dtype=float)
        self.radius = traj_cfg["workspace_radius"]
        self.dt = 1.0 / config["env"]["control_freq"]

        # --- Lissajous settings ---
        self.lissajous_duration = traj_cfg.get("lissajous_duration", 10.0)
        self.ori_ramp_duration = traj_cfg.get("ori_ramp_duration", 1.5)
        # per-axis randomization ranges for *training* curves
        self.train_amp_range = traj_cfg.get("train_amp_range", [0.3, 1.0])
        self.train_freq_range = traj_cfg.get("train_freq_range", [0.5, 3.0])

        # curve placement: shift every curve (train + eval) into the region
        # where the EE can point straight down; eval_amp_scale additionally
        # shrinks the canonical eval curve. Defaults reproduce old behaviour.
        self.curve_center_offset = np.array(
            traj_cfg.get("curve_center_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.eval_amp_scale = float(traj_cfg.get("eval_amp_scale", 1.0))

        # Downward end-effector target. "Pointing down" only fixes the EE
        # z-axis; the yaw is a free choice. This default is a fallback --
        # reset() recomputes it from the reset pose so its yaw matches the
        # home pose (see _force_z_down), which leaves the orientation ramp
        # nothing to slew.
        self.down_orientation = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

        # waypoint sampling radius -- only used in legacy "waypoint" mode
        self.train_radius = self.radius * float(np.linalg.norm(self._LIS_AMP)) * 1.1

        self.t = 0.0
        self.waypoints = []
        self.orientations = []
        self.total_duration = 0.0
        self.anchor = self.center.copy()
        # EE orientation at reset; the orientation target ramps away from this
        self.start_orientation = self.down_orientation.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, traj_type=None, start_pos=None, start_rot=None, randomize=False):
        """
        Sample a new trajectory. Call at the start of each episode.

        start_pos : current EE position; the curve is anchored here so the
                    agent starts already on-target.
        start_rot : current EE rotation matrix; the orientation target ramps
                    from here to the downward pose.
        randomize : if True, sample a random Lissajous (training); if False,
                    use the fixed canonical curve (evaluation).
        """
        self.t = 0.0
        ttype = traj_type or self.traj_type
        self.traj_type = ttype
        self.anchor = (np.array(start_pos, dtype=float)
                       if start_pos is not None else self.center.copy())
        if start_rot is not None:
            self.start_orientation = np.array(start_rot, dtype=float)
            # Match the downward target's yaw to the reset pose. The reset
            # pose already points straight down, so this makes it on-target
            # in orientation from t=0 and the orientation ramp becomes a
            # no-op -- without it the ramp must slew 90 deg of yaw, faster
            # than the joints can move, producing a large startup error.
            self.down_orientation = self._force_z_down(self.start_orientation)

        if ttype == "waypoint":
            self._init_waypoints()
        elif ttype == "lissajous":
            self._init_lissajous(randomize=randomize)
        else:
            raise ValueError(f"Unknown trajectory type: {ttype}")

    def step(self):
        """Advance time by one control step and return the current target."""
        pos, lin_vel = self._get_position(self.t)
        pos, lin_vel = self._startup_position_ramp(self.t, pos, lin_vel)
        rot_mat, ang_vel = self._get_orientation(self.t)
        self.t += self.dt
        return pos, lin_vel, rot_mat, ang_vel

    def is_done(self):
        return self.t >= self.total_duration

    # ------------------------------------------------------------------
    # Lissajous trajectory (training: randomized; evaluation: fixed)
    # ------------------------------------------------------------------

    def _init_lissajous(self, randomize=False):
        if randomize:
            amp_lo, amp_hi = self.train_amp_range
            frq_lo, frq_hi = self.train_freq_range
            amps = self.radius * np.random.uniform(amp_lo, amp_hi, size=3)
            freqs = np.random.uniform(frq_lo, frq_hi, size=3)
            phases = np.random.uniform(0.0, 2.0 * np.pi, size=3)
        else:
            # canonical evaluation curve -- fixed so the eval score stays
            # comparable across runs (identical shape to the original eval
            # curve, scaled by eval_amp_scale)
            amps = self.radius * self._LIS_AMP * self.eval_amp_scale
            freqs = np.array([1.0, 2.0, 3.0])
            phases = np.zeros(3)

        self.lis_params = {
            "Ax": amps[0], "Ay": amps[1], "Az": amps[2],
            "wx": freqs[0], "wy": freqs[1], "wz": freqs[2],
            "px": phases[0], "py": phases[1], "pz": phases[2],
        }
        # place the curve at (anchor + curve_center_offset); the t=0 point
        # lands there, and the startup ramp eases the target onto it from the
        # EE's reset pose so there is still no positional jump at episode start
        start_offset = amps * np.sin(phases)
        self.lis_center = self.anchor + self.curve_center_offset - start_offset
        self.total_duration = self.lissajous_duration

    def _lissajous_pos(self, t):
        p = self.lis_params
        c = self.lis_center
        x = c[0] + p["Ax"] * np.sin(p["wx"] * t + p["px"])
        y = c[1] + p["Ay"] * np.sin(p["wy"] * t + p["py"])
        z = c[2] + p["Az"] * np.sin(p["wz"] * t + p["pz"])
        pos = np.array([x, y, z])

        # numerical velocity
        dt = 1e-4
        x2 = c[0] + p["Ax"] * np.sin(p["wx"] * (t + dt) + p["px"])
        y2 = c[1] + p["Ay"] * np.sin(p["wy"] * (t + dt) + p["py"])
        z2 = c[2] + p["Az"] * np.sin(p["wz"] * (t + dt) + p["pz"])
        vel = (np.array([x2, y2, z2]) - pos) / dt
        return pos, vel

    # ------------------------------------------------------------------
    # Waypoint trajectory (legacy -- only used if train_type == "waypoint")
    # ------------------------------------------------------------------

    def _init_waypoints(self):
        n = self.n_waypoints
        self.waypoints = [self.anchor.copy()]
        self.waypoints += [self._random_workspace_point() for _ in range(n)]
        self.orientations = [self.down_orientation.copy() for _ in range(n + 1)]
        self.total_duration = n * self.interp_duration

    def _random_workspace_point(self):
        """Uniform sample inside a sphere of train_radius around the anchor."""
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                return self.anchor + p * self.train_radius

    def _waypoint_pos(self, t):
        segment = int(t / self.interp_duration)
        segment = min(segment, len(self.waypoints) - 2)
        tau = (t - segment * self.interp_duration) / self.interp_duration

        p0 = self.waypoints[segment]
        p1 = self.waypoints[segment + 1]

        s = self._min_jerk(tau)
        ds = self._min_jerk_dot(tau) / self.interp_duration

        pos = p0 + s * (p1 - p0)
        vel = ds * (p1 - p0)
        return pos, vel

    # ------------------------------------------------------------------
    # Position dispatch
    # ------------------------------------------------------------------

    def _get_position(self, t):
        if self.traj_type == "lissajous":
            return self._lissajous_pos(t)
        return self._waypoint_pos(t)

    def _startup_position_ramp(self, t, pos, vel):
        """Ease the position target from the EE's reset position (the anchor)
        onto the curve over the startup ramp -- the position counterpart of
        the orientation ramp. Relocating the curve centre opens a gap between
        the EE's reset pose and the curve's t=0 point; this min-jerk blend
        closes it so every episode still starts exactly on-target. The blend
        is C1-smooth at the hand-off: when the ramp ends the blended velocity
        equals the curve velocity, so there is no jolt onto the curve."""
        ramp = self.ori_ramp_duration
        if ramp <= 0.0 or t >= ramp:
            return pos, vel
        s = self._min_jerk(t / ramp)
        s_dot = self._min_jerk_dot(t / ramp) / ramp
        delta = pos - self.anchor
        return self.anchor + s * delta, s_dot * delta + s * vel

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------

    def _get_orientation(self, t):
        """
        Orientation target: ramp from the EE's start orientation to the
        downward pose over `ori_ramp_duration`, then hold downward.

        The ramp removes the startup transient that previously inflated the
        orientation error -- the episode resets to a home pose whose
        orientation is far from 'downward', and snapping the target straight
        to 'downward' forced a large unavoidable error for the first ~1.5 s.
        Applies to both Lissajous and (legacy) waypoint modes.
        """
        if t >= self.ori_ramp_duration:
            return self.down_orientation, np.zeros(3)

        R0 = Rotation.from_matrix(self.start_orientation)
        R1 = Rotation.from_matrix(self.down_orientation)

        s = self._min_jerk(t / self.ori_ramp_duration)
        rot_mat = self._slerp(R0, R1, s).as_matrix()

        # numerical angular velocity
        dt = 1e-4
        s2 = self._min_jerk(min((t + dt) / self.ori_ramp_duration, 1.0))
        R_next = self._slerp(R0, R1, s2).as_matrix()
        dR = R_next @ rot_mat.T
        ang_vel = Rotation.from_matrix(dR).as_rotvec() / dt
        return rot_mat, ang_vel

    @staticmethod
    def _slerp(r0, r1, t):
        """Spherical linear interpolation between two rotations."""
        rotvec = (r1 * r0.inv()).as_rotvec()
        return Rotation.from_rotvec(t * rotvec) * r0

    @staticmethod
    def _force_z_down(R):
        """Return the orientation closest to R whose z-axis points straight
        down, i.e. R rotated by the minimal tilt that verticalizes its
        z-axis. For a pose already pointing down this returns R unchanged
        (it just keeps R's yaw). Used so the downward target matches the
        EE's reset pose, leaving the orientation startup ramp nothing to do."""
        z = np.asarray(R)[:, 2]
        target = np.array([0.0, 0.0, -1.0])
        axis = np.cross(z, target)
        s = np.linalg.norm(axis)
        if s < 1e-9:                       # already vertical (up or down)
            return np.array(R, dtype=float)
        angle = np.arctan2(s, float(np.dot(z, target)))
        align = Rotation.from_rotvec(axis / s * angle).as_matrix()
        return align @ np.asarray(R, dtype=float)

    # ------------------------------------------------------------------
    # Minimum-jerk profile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_jerk(t):
        """Minimum-jerk scalar interpolation s(t), t in [0, 1]."""
        t = np.clip(t, 0.0, 1.0)
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    @staticmethod
    def _min_jerk_dot(t):
        """Derivative of the minimum-jerk profile."""
        t = np.clip(t, 0.0, 1.0)
        return 30 * t**2 - 60 * t**3 + 30 * t**4