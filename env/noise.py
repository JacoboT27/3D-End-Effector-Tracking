import numpy as np


class ObservationNoise:
    """
    Adds Gaussian noise to end-effector and joint observations.
    Only uncertainty source in this system.
    """

    def __init__(self, config):
        noise_cfg = config["noise"]
        self.ee_pos_std = noise_cfg["ee_pos_std"]
        self.ee_ori_std = noise_cfg["ee_ori_std"]
        self.joint_pos_std = noise_cfg["joint_pos_std"]

    def apply_ee_pos(self, pos: np.ndarray) -> np.ndarray:
        """Add noise to end-effector position. pos: (3,)"""
        return pos + np.random.normal(0.0, self.ee_pos_std, pos.shape)

    def apply_ee_ori(self, rot_6d: np.ndarray) -> np.ndarray:
        """Add noise to 6D orientation representation. rot_6d: (6,)"""
        return rot_6d + np.random.normal(0.0, self.ee_ori_std, rot_6d.shape)

    def apply_joints(self, qpos: np.ndarray) -> np.ndarray:
        """Add noise to joint positions. qpos: (n_joints,)"""
        return qpos + np.random.normal(0.0, self.joint_pos_std, qpos.shape)