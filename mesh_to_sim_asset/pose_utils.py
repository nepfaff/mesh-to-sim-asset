import numpy as np


def pose_list_to_matrix(pose_list: list[float]) -> np.ndarray:
    """Convert pose list [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix.

    Args:
        pose_list: List of 6 floats representing [x, y, z, roll, pitch, yaw].

    Returns:
        4x4 transformation matrix.
    """
    x, y, z, roll, pitch, yaw = pose_list

    # Create rotation matrix from Euler angles (roll, pitch, yaw).
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Rotation matrices for each axis.
    R_x = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    R_y = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    R_z = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

    # Combined rotation matrix (ZYX order).
    R = R_z @ R_y @ R_x

    # Create 4x4 transformation matrix.
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 transformation matrix to pose list [x, y, z, roll, pitch, yaw].

    Args:
        matrix: 4x4 transformation matrix.

    Returns:
        List of 6 floats representing [x, y, z, roll, pitch, yaw].
    """
    # Extract translation.
    x, y, z = matrix[:3, 3]

    # Extract rotation matrix.
    R = matrix[:3, :3]

    # Convert rotation matrix to Euler angles (ZYX order).
    # Handle singularities.
    if abs(R[2, 0]) >= 1:
        # Gimbal lock case.
        yaw = 0  # Set yaw to 0.
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    return [x, y, z, roll, pitch, yaw]
