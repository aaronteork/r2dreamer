import numpy as np
from scipy.spatial.transform import Rotation


def compute_ego_motion(
    previous_position,
    previous_rotation,
    current_position,
    current_rotation,
):
    """Return the torso pose increment expressed in the previous torso frame.

    Both rotation matrices map torso-frame coordinates into world coordinates.
    """
    previous_position = np.asarray(previous_position, dtype=np.float64)
    previous_rotation = np.asarray(previous_rotation, dtype=np.float64)
    current_position = np.asarray(current_position, dtype=np.float64)
    current_rotation = np.asarray(current_rotation, dtype=np.float64)

    delta_translation = previous_rotation.T @ (
        current_position - previous_position
    )
    delta_rotation = previous_rotation.T @ current_rotation
    delta_rotvec = Rotation.from_matrix(delta_rotation).as_rotvec()
    ego_motion = np.concatenate((delta_translation, delta_rotvec)).astype(
        np.float32
    )
    assert ego_motion.shape == (6,)
    assert np.all(np.isfinite(ego_motion))
    return ego_motion
