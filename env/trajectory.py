import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryGenerator:
    """
    Generates end-effector trajectories for training and evaluation.

    Training:   random waypoints connected with minimum-jerk interpolation.
    Evaluation: a Lissajous curve (a novel path, for the generalization test).

    Both are anchored at the end-effector's position at reset, so every
    episode starts on-target. Training waypoints are sampled from a region
    large enough to contain the full eval Lissajous, so the policy practices
    everywhere the eval curve visits.

    The orientation target is a fixed downward pose for both training and
    evaluation -- a pose reachable throughout the workspace (uniformly random
    orientations are mostly kinematically infeasible and corrupt training).

    Every step returns:
        (position, linear_velocity, orientation_matrix, angular_velocity)
    """

    # Lissajous amplitude per axis, as a fraction of workspace_radius.
    _LIS_AMP = np.array([0.8, 0.8, 0.4])

    def __init__(self, config):
        self.traj_type = config["trajectory"]["train_type"]
        self.n_waypoints = config["trajectory"]["n_waypoints"]
        self.interp_duration = config["trajectory"]["interp_duration"]
        self.center = np.array(config["trajectory"]["workspace_center"], dtype=float)
        self.radius = config["trajectory"]["workspace_radius"]
        self.dt = 1.0 / config["env"]["control_freq"]

        # fixed downward end-effector orientation, used by train and eval alike
        self.down_orientation = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

        # training waypoints are sampled within this radius of the anchor --
        # large enough to contain the eval Lissajous (which reaches
        # norm(_LIS_AMP) * radius from its centre), with a 10% margin
        self.train_radius = self.radius * float(np.linalg.norm(self._LIS_AMP)) * 1.1

        self.t = 0.0
        self.waypoints = []
        self.orientations = []
        self.total_duration = 0.0
        self.anchor = self.center.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, traj_type=None, start_pos=None):
        """
        Sample a new trajectory. Call at the start of each episode.

        start_pos : current end-effector position; the trajectory is anchored
                    here so the agent starts already on-target.
        """
        self.t = 0.0
        ttype = traj_type or self.traj_type
        self.traj_type = ttype
        self.anchor = (np.array(start_pos, dtype=float)
                       if start_pos is not None else self.center.copy())

        if ttype == "waypoint":
            self._init_waypoints()
        elif ttype == "lissajous":
            self._init_lissajous()
        else:
            raise ValueError(f"Unknown trajectory type: {ttype}")

    def step(self):
        """Advance time by one control step and return the current target."""
        pos, lin_vel = self._get_position(self.t)
        rot_mat, ang_vel = self._get_orientation(self.t)
        self.t += self.dt
        return pos, lin_vel, rot_mat, ang_vel

    def is_done(self):
        return self.t >= self.total_duration

    # ------------------------------------------------------------------
    # Waypoint trajectory (training)
    # ------------------------------------------------------------------

    def _init_waypoints(self):
        n = self.n_waypoints
        # first waypoint is the anchor (current EE) -> no startup gap;
        # the rest are random points across the eval-curve region
        self.waypoints = [self.anchor.copy()]
        self.waypoints += [self._random_workspace_point() for _ in range(n)]
        # fixed downward orientation at every waypoint -- matches the eval
        # target and is reachable everywhere, unlike uniformly random poses
        self.orientations = [self.down_orientation.copy() for _ in range(n + 1)]
        self.total_duration = n * self.interp_duration

    def _random_workspace_point(self):
        """Uniform sample inside a sphere of train_radius around the anchor."""
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                return self.anchor + p * self.train_radius

    # ------------------------------------------------------------------
    # Lissajous trajectory (evaluation)
    # ------------------------------------------------------------------

    def _init_lissajous(self):
        # zero phase offsets -> the curve's t=0 point is its own centre, so it
        # starts exactly on the anchor and stays centred on the training
        # region (no startup gap, no train/eval spatial offset)
        ax, ay, az = self.radius * self._LIS_AMP
        self.lis_params = {
            "Ax": ax, "Ay": ay, "Az": az,
            "wx": 1.0, "wy": 2.0, "wz": 3.0,
            "px": 0.0, "py": 0.0, "pz": 0.0,
        }
        self.lis_center = self.anchor.copy()
        self.lis_orientation = self.down_orientation
        self.total_duration = 10.0  # seconds, one full Lissajous cycle

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def _get_position(self, t):
        if self.traj_type == "lissajous":
            return self._lissajous_pos(t)
        return self._waypoint_pos(t)

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
    # Orientation
    # ------------------------------------------------------------------

    def _get_orientation(self, t):
        if self.traj_type == "lissajous":
            return self.lis_orientation, np.zeros(3)

        segment = int(t / self.interp_duration)
        segment = min(segment, len(self.orientations) - 2)
        tau = (t - segment * self.interp_duration) / self.interp_duration

        s = self._min_jerk(tau)
        R0 = Rotation.from_matrix(self.orientations[segment])
        R1 = Rotation.from_matrix(self.orientations[segment + 1])
        R_interp = self._slerp(R0, R1, s)
        rot_mat = R_interp.as_matrix()

        # numerical angular velocity
        dt = 1e-4
        s2 = self._min_jerk(min(tau + dt / self.interp_duration, 1.0))
        R_next = self._slerp(R0, R1, s2)
        dR = R_next.as_matrix() @ rot_mat.T
        ang_vel = Rotation.from_matrix(dR).as_rotvec() / dt
        return rot_mat, ang_vel

    @staticmethod
    def _slerp(r0, r1, t):
        """Spherical linear interpolation between two rotations."""
        rotvec = (r1 * r0.inv()).as_rotvec()
        return Rotation.from_rotvec(t * rotvec) * r0

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