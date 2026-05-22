import numpy as np
from scipy.spatial.transform import Rotation


def rot_mat_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D representation (Zhou et al. 2019).
    Takes the first two columns of R, flattened to (6,).
    Network output can be freely in R^6, then reconstructed to valid SO(3).
    """
    return R[:, :2].T.flatten()  # shape (6,)


def rot_6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    """
    Reconstruct valid rotation matrix from 6D representation via Gram-Schmidt.
    r6d: (6,) -> R: (3,3)
    """
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # (3,3)


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Geodesic distance between two rotation matrices on SO(3).
    Returns angle in radians: 0 (identical) to pi (opposite).
    """
    R_rel = R1.T @ R2
    # numerical clamp to valid arccos domain
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def quat_to_rot_mat(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to 3x3 rotation matrix."""
    # MuJoCo uses (w, x, y, z) convention
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()