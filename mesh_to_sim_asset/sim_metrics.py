import numpy as np

from scipy.spatial.transform import Rotation


def _states_to_poses(states: np.ndarray) -> np.ndarray:
    """
    Args:
        states: States of shape (N,13) where N is the number of trajectory points.
            Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, vz],
            where q are quaternions, t are translations, w are angular velocities, and v
            are translational velocities.

    Returns:
        Homogenous transformation matrices of shape (N,4,4).
    """
    poses = np.eye(4)[np.newaxis, :].repeat(len(states), axis=0)
    # Drake used (qw, qx, qy, qz) and scipy uses (qx, qy, qz, qw).
    poses[:, :3, :3] = Rotation.from_quat(
        np.concatenate((states[:, 1:4], states[:, :1]), axis=-1)
    ).as_matrix()
    poses[:, :3, 3] = states[:, 4:7]
    return poses


def orientation_considered_final_displacement_error(
    gt_state_trajectory: np.ndarray, state_trajectory: np.ndarray
) -> float:
    """
    Final Displacement Error (FDE) metric that consideres orientation by sampling points
    relative to the object pose and taking the mean displacement error of these points.

    Args:
        gt_state_trajectory: The trajectory of GT object states of shape
            (N,13) where N is the number of trajectory points. Each point has the form
            [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, vz], where q are
            quaternions, t are translations, w are angular velocities, and v are
            translational velocities.
        state_trajectory: The trajectory of the object states of same shape as
            `gt_state_trajectory`.

    Returns:
        The final displacement error considering orientation.
    """
    final_state_gt = gt_state_trajectory[-1][np.newaxis, :]
    final_state_pred = state_trajectory[-1][np.newaxis, :]

    final_pose_gt = _states_to_poses(final_state_gt).squeeze(0)
    final_pose_pred = _states_to_poses(final_state_pred).squeeze(0)

    # Sample 3 orthogonalpoints in object frame to completely define the orientation.
    points_object_frame = np.eye(3)
    points_gt_world_frame = (
        points_object_frame @ final_pose_gt[:3, :3].T + final_pose_gt[:3, 3]
    )
    points_pred_world_frame = (
        points_object_frame @ final_pose_pred[:3, :3].T + final_pose_pred[:3, 3]
    )

    pointwise_error = np.linalg.norm(
        points_gt_world_frame - points_pred_world_frame, axis=-1
    )
    mean_error = float(np.mean(pointwise_error))
    return mean_error


def orientation_considered_average_displacement_error(
    gt_state_trajectory: np.ndarray,
    state_trajectory: np.ndarray,
    num_points_per_axis: int = 1,
) -> float:
    """
    Average Displacement Error (ADE) metric that considers orientation by sampling
    points along each coordinate axis and measuring their displacement in world frame.

    The metric works by:
    1. Sampling points along each coordinate axis in the object's local frame
    2. Transforming these points to world frame for both trajectories
    3. Computing the mean displacement between corresponding points at each timestep
    4. Averaging these displacements across the entire trajectory

    Args:
        gt_state_trajectory: The trajectory of GT object states of shape
            (N,13) where N is the number of trajectory points. Each point has the form
            [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, vz], where q are
            quaternions, t are translations, w are angular velocities, and v are
            translational velocities.
        state_trajectory: The trajectory of the object states of same shape as
            `gt_state_trajectory`.
        num_points_per_axis: The number of points to sample per coordinate axis.
            When set to 1, samples unit vectors along each axis. Higher values sample
            additional points between 0 and 1 along each axis for more robust
            measurement.

    Returns:
        The average displacement error incorporating both position and orientation.
    """
    poses_gt = _states_to_poses(gt_state_trajectory)  # Shape (N,4,4)
    poses_pred = _states_to_poses(state_trajectory)  # Shape (N,4,4)

    # Sample points in the object frame.
    if num_points_per_axis == 1:
        points_object_frame = np.eye(3)
    else:
        points_object_frame = np.zeros((num_points_per_axis * 3, 3))
        line = np.linspace(0, 1, num_points_per_axis)
        points_object_frame[:num_points_per_axis, 0] = line
        points_object_frame[num_points_per_axis : 2 * num_points_per_axis, 1] = line
        points_object_frame[2 * num_points_per_axis : 3 * num_points_per_axis, 2] = line

    # Transform to world frame.
    points_gt_world_frame = (
        points_object_frame @ poses_gt[:, :3, :3].transpose((0, 2, 1))
        + poses_gt[:, :3, 3][:, np.newaxis, :]
    )  # Shape (N,3,3)
    points_pred_world_frame = (
        points_object_frame @ poses_pred[:, :3, :3].transpose((0, 2, 1))
        + poses_pred[:, :3, 3][:, np.newaxis, :]
    )  # Shape (N,3,3)

    pointwise_error = np.linalg.norm(
        points_gt_world_frame - points_pred_world_frame, axis=-1
    )  # Shape (N,3)
    mean_error = np.mean(pointwise_error, axis=-1)  # Shape (N,)
    ade = float(np.mean(mean_error))
    return ade
