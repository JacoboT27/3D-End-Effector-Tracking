import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryGenerator:
    """
    Generates end-effector trajectories for training and evaluation.

    Training: random waypoints with minimum-jerk interpolation.
    Evaluation: Lissajous curve.

    All trajectories return (position, linear_velocity, orientation_matrix, angular_velocity)
    at each timestep.
    """

    def __init__(self, config):
        self.traj_type = config["trajectory"]["train_type"]
        self.n_waypoints = config["trajectory"]["n_waypoints"]
        self.interp_duration = config["trajectory"]["interp_duration"]
        self.center = np.array(config["trajectory"]["workspace_center"])
        self.radius = config["trajectory"]["workspace_radius"]
        self.dt = 1.0 / config["env"]["control_freq"]

        self.t = 0.0
        self.waypoints = []
        self.orientations = []
        self.total_duration = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, traj_type=None):
        """Sample a new trajectory. Call at the start of each episode."""
        self.t = 0.0
        ttype = traj_type or self.traj_type

        if ttype == "waypoint":
            self._init_waypoints()
        elif ttype == "lissajous":
            self._init_lissajous()
        else:
            raise ValueError(f"Unknown trajectory type: {ttype}")

    def step(self):
        """
        Advance time by one control step.
        Returns:
            pos      : (3,)  target position
            lin_vel  : (3,)  target linear velocity
            rot_mat  : (3,3) target orientation as rotation matrix
            ang_vel  : (3,)  target angular velocity
        """
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
        # sample random positions within a sphere
        self.waypoints = [self._random_workspace_point() for _ in range(n + 1)]
        # sample random orientations
        self.orientations = [Rotation.random().as_matrix() for _ in range(n + 1)]
        self.total_duration = n * self.interp_duration

    def _random_workspace_point(self):
        """Uniform sample inside a sphere."""
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                return self.center + p * self.radius

    # ------------------------------------------------------------------
    # Lissajous trajectory (evaluation)
    # ------------------------------------------------------------------

    def _init_lissajous(self):
        # frequency ratios give visually rich 3D curves
        self.lis_params = {
            "Ax": self.radius * 0.8,
            "Ay": self.radius * 0.8,
            "Az": self.radius * 0.4,
            "wx": 1.0,
            "wy": 2.0,
            "wz": 3.0,
            "px": 0.0,
            "py": np.pi / 4,
            "pz": np.pi / 2,
        }
        # fixed orientation: keep end-effector pointing downward
        self.lis_orientation = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
        self.total_duration = 10.0  # seconds, full Lissajous cycle

    # ------------------------------------------------------------------
    # Position evaluation
    # ------------------------------------------------------------------

    def _get_position(self, t):
        if self.traj_type == "lissajous" or (
            hasattr(self, "lis_params") and self.traj_type != "waypoint"
        ):
            return self._lissajous_pos(t)
        return self._waypoint_pos(t)

    def _lissajous_pos(self, t):
        p = self.lis_params
        x = self.center[0] + p["Ax"] * np.sin(p["wx"] * t + p["px"])
        y = self.center[1] + p["Ay"] * np.sin(p["wy"] * t + p["py"])
        z = self.center[2] + p["Az"] * np.sin(p["wz"] * t + p["pz"])
        pos = np.array([x, y, z])

        # numerical velocity
        dt = 1e-4
        x2 = self.center[0] + p["Ax"] * np.sin(p["wx"] * (t + dt) + p["px"])
        y2 = self.center[1] + p["Ay"] * np.sin(p["wy"] * (t + dt) + p["py"])
        z2 = self.center[2] + p["Az"] * np.sin(p["wz"] * (t + dt) + p["pz"])
        vel = (np.array([x2, y2, z2]) - pos) / dt

        return pos, vel

    def _waypoint_pos(self, t):
        segment = int(t / self.interp_duration)
        segment = min(segment, len(self.waypoints) - 2)
        tau = (t - segment * self.interp_duration) / self.interp_duration  # [0,1]

        p0 = self.waypoints[segment]
        p1 = self.waypoints[segment + 1]

        # minimum-jerk interpolation
        s = self._min_jerk(tau)
        ds = self._min_jerk_dot(tau) / self.interp_duration

        pos = p0 + s * (p1 - p0)
        vel = ds * (p1 - p0)
        return pos, vel

    # ------------------------------------------------------------------
    # Orientation evaluation
    # ------------------------------------------------------------------

    def _get_orientation(self, t):
        if hasattr(self, "lis_orientation"):
            return self.lis_orientation, np.zeros(3)

        segment = int(t / self.interp_duration)
        segment = min(segment, len(self.orientations) - 2)
        tau = (t - segment * self.interp_duration) / self.interp_duration

        s = self._min_jerk(tau)
        R0 = Rotation.from_matrix(self.orientations[segment])
        R1 = Rotation.from_matrix(self.orientations[segment + 1])
        # SLERP between orientations
        R_interp = Rotation.slerp_single(R0, R1, s) if False else self._slerp(R0, R1, s)
        rot_mat = R_interp.as_matrix()

        # numerical angular velocity
        dt = 1e-4
        s2 = self._min_jerk(min(tau + dt / self.interp_duration, 1.0))
        R_next = self._slerp(R0, R1, s2)
        dR = R_next.as_matrix() @ rot_mat.T
        ang_vel = Rotation.from_matrix(dR).as_rotvec() / dt

        return rot_mat, ang_vel

    @staticmethod
    def _slerp(r0: Rotation, r1: Rotation, t: float) -> Rotation:
        """Simple SLERP between two Rotations."""
        rotvec = (r1 * r0.inv()).as_rotvec()
        return Rotation.from_rotvec(t * rotvec) * r0

    # ------------------------------------------------------------------
    # Minimum-jerk profile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_jerk(t):
        """Minimum-jerk scalar interpolation s(t), t in [0,1]."""
        t = np.clip(t, 0.0, 1.0)
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    @staticmethod
    def _min_jerk_dot(t):
        """Derivative of minimum-jerk profile."""
        t = np.clip(t, 0.0, 1.0)
        return 30 * t**2 - 60 * t**3 + 30 * t**4