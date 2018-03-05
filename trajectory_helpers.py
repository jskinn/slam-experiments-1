import typing
import numpy as np
import arvet.util.transform as tf


def zero_trajectory(trajectory: typing.Mapping[float, tf.Transform]) -> typing.Mapping[float, tf.Transform]:
    first_pose = trajectory[min(trajectory.keys())]
    return {
        stamp: first_pose.find_relative(pose)
        for stamp, pose in trajectory.items()
    }


def find_trajectory_scale(trajectory: typing.Mapping[float, tf.Transform]) -> float:
    timestamps = sorted(trajectory.keys())
    speeds = []
    for idx in range(1, len(timestamps)):
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        dist = np.linalg.norm(trajectory[t1].location - trajectory[t0].location)
        speeds.append(dist / (t1 - t0))
    return float(np.mean(speeds))


def rescale_trajectory(trajectory: typing.Mapping[float, tf.Transform], scale: float) \
        -> typing.Mapping[float, tf.Transform]:
    current_scale = find_trajectory_scale(trajectory)

    timestamps = sorted(trajectory.keys())
    scaled_trajectory = {timestamps[0]: trajectory[timestamps[0]]}
    for idx in range(1, len(timestamps)):
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        motion = trajectory[t0].find_relative(trajectory[t1])
        scaled_trajectory[t1] = scaled_trajectory[t0].find_independent(tf.Transform(
            location=(scale / current_scale) * motion.location,
            rotation=motion.rotation_quat(w_first=True),
            w_first=True
        ))
    return scaled_trajectory


def trajectory_to_motion_sequence(trajectory: typing.Mapping[float, tf.Transform]) -> \
        typing.Mapping[float, tf.Transform]:
    """
    Convert a trajectory into a sequence of relative motions
    :param trajectory:
    :return:
    """
    times = sorted(trajectory.keys())
    prev_time = times[0]
    motions = {}
    for time in times[1:]:
        motion = trajectory[prev_time].find_relative(trajectory[time])
        prev_time = time
        motions[time] = motion
    return motions
