import os
import typing
import json
import bson
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.unreal_transform as uetf
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.database.client


def plot_trajectory(axis, trajectory: typing.Mapping[float, tf.Transform], label: str, style: str = '-') \
        -> typing.Tuple[float, float]:
    """
    Simple helper to plot a trajectory on a 3D axis.
    Will normalise the trajectory to start at (0,0,0) and facing down the x axis,
    that is, all poses are relative to the first one.
    :param axis: The axis on which to plot
    :param trajectory: A map of timestamps to camera poses
    :param label: The label for the series
    :param style: The line style to use for the trajectory. Lets us distinguish virtual and real world results.
    :return: The minimum and maximum coordinate values, for axis sizing.
    """
    x = []
    y = []
    z = []
    max_point = 0
    min_point = 0
    times = sorted(trajectory.keys())
    first_pose = None
    for timestamp in times:
        pose = trajectory[timestamp]
        if first_pose is None:
            first_pose = pose
            x.append(0)
            y.append(0)
            z.append(0)
        else:
            pose = first_pose.find_relative(pose)
            max_point = max(max_point, pose.location[0], pose.location[1], pose.location[2])
            min_point = min(min_point, pose.location[0], pose.location[1], pose.location[2])
            x.append(pose.location[0])
            y.append(pose.location[1])
            z.append(pose.location[2])
    axis.plot(x, y, z, style, label=label, alpha=0.7)
    return min_point, max_point


def plot_component(ax, trajectories: typing.List[typing.Mapping[float, tf.Transform]],
                   get_value: typing.Callable[[tf.Transform], float], label: str = '', alpha: float = 1.0, **kwargs):
    """
    Plot a particular trajectory component over time on an axis for a number of trajectories
    :param ax: The axis on which to plot
    :param trajectories: The list of trajectories to plot
    :param get_value: A getter to retrieve the y value from the Transform object
    :param label: The label
    :param alpha: The total alpha value of the line, which will be divided amonst the given trajectories
    :return:
    """
    du.defaults(kwargs, {
        'markersize': 2,
        'marker': '.',
        'linestyle': 'None'
    })
    for idx, traj in enumerate(trajectories):
        x = sorted(traj.keys())
        ax.plot(x, [get_value(traj[t]) for t in x], alpha=alpha / len(trajectories),
                label="{0} {1}".format(label, idx), **kwargs)


def create_axis_plot(title: str, trajectory_groups: typing.List[
                         typing.Tuple[str, typing.List[typing.Mapping[float, tf.Transform]], dict]
                     ], save_path: str = None):
    """
    Create a plot of location and rotation coordinate values as a function time.
    This is useful for checking if different trajectories are the same
    :param title: The plot title
    :param trajectory_groups: A list of tuples, each containing a name, a list of trajectories, and a plot style
    :param save_path: An optional path to save the plot to, rather than display it
    :return:
    """
    import matplotlib.pyplot as pyplot
    import matplotlib.patches as mpatches

    figure = pyplot.figure(figsize=(20, 10), dpi=80)
    figure.suptitle(title)
    legend_handles = []
    for idx, (subtitle, units, get_value) in enumerate([
        ('x axis', 'meters', lambda t: t.location[0]),
        ('y axis', 'meters', lambda t: t.location[1]),
        ('z axis', 'meters', lambda t: t.location[2]),
        ('roll', 'degrees', lambda t: 180 * t.euler[0] / np.pi),
        ('pitch', 'degrees', lambda t: 180 * t.euler[1] / np.pi),
        ('yaw', 'degrees', lambda t: 180 * t.euler[2] / np.pi)
    ]):
        ax = figure.add_subplot(231 + idx)
        ax.set_title(subtitle)
        ax.set_xlabel('time')
        ax.set_ylabel(units)
        for label, traj_group, plot_kwargs in trajectory_groups:
            plot_component(ax, traj_group, get_value, alpha=0.5, **plot_kwargs)
            if idx == 0:
                legend_handles.append(mpatches.Patch(color=plot_kwargs['c'], alpha=0.5, label=label))
    pyplot.figlegend(handles=legend_handles, loc='upper right')
    figure.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # figure.savefig(os.path.join(save_path, title + '.svg'))
        figure.savefig(os.path.join(save_path, title + '.png'))
        pyplot.close(figure)


def export_trajectory_as_json(trial_results: typing.Mapping[str, bson.ObjectId], filename: str,
                              db_client: arvet.database.client.DatabaseClient) -> None:
    """
    Export trajectories from trial results to json, so that the web gui can display them.
    Allows for multiple trial results with different labels, final json format
    is the label to the computed trajectory, plus a key 'ground_truth' containing the ground truth trajectory
    :param trial_results: A map from label to
    :param filename: The name of the file to save, without suffix {filename}.json
    :param db_client: The database client, for loading trial results
    :return:
    """
    if len(trial_results) >= 1:
        json_data = {}
        added_ground_truth = False

        # For each trial result
        for label, trial_result_id in trial_results.items():
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            if trial_result is not None:
                if trial_result.success:
                    if not added_ground_truth:
                        added_ground_truth = True
                        trajectory = trial_result.get_ground_truth_camera_poses()
                        if len(trajectory) > 0:
                            first_pose = trajectory[min(trajectory.keys())]
                            json_data['ground_truth'] = [
                                [time] + location_to_json(first_pose.find_relative(pose))
                                for time, pose in trajectory.items()
                            ]
                    trajectory = trial_result.get_computed_camera_poses()
                    if len(trajectory) > 0:
                        first_pose = trajectory[min(trajectory.keys())]
                        json_data[label] = [[time] + location_to_json(first_pose.find_relative(pose))
                                            for time, pose in trajectory.items()]

        with open('{0}.json'.format(filename), 'w') as json_file:
            json.dump(json_data, json_file)


def dump_ue4_trajectory(name: str, trajectory: typing.Mapping[float, tf.Transform]) -> None:
    """
    Save a trajectory to a csv file so that it can be imported by unreal.
    Handles conversion to unreal coordinate frames.
    :param name: The name of the trajectory, filename will be "unreal_trajectory_{name}"
    :param trajectory: The trajectory as a map from timestamp to transform
    :return: None
    """
    with open('unreal_trajectory_{0}.csv'.format(name), 'w') as output_file:
        output_file.write('Name,X,Y,Z,Roll,Pitch,Yaw\n')
        for idx, timestamp in enumerate(sorted(trajectory.keys())):
            ue_pose = uetf.transform_to_unreal(trajectory[timestamp])
            output_file.write('{name},{x},{y},{z},{roll},{pitch},{yaw}\n'.format(
                name=idx,
                x=ue_pose.location[0],
                y=ue_pose.location[1],
                z=ue_pose.location[2],
                roll=ue_pose.euler[0],
                pitch=ue_pose.euler[1],
                yaw=ue_pose.euler[2]))


def location_to_json(pose: tf.Transform) -> typing.List[float]:
    """
    A simple helper to pull location from a transform and return it
    :param pose: A Transform object
    :return: The list of coordinates of it's location
    """
    return [
        pose.location[0],
        pose.location[1],
        pose.location[2]
    ]


def compute_window(data: np.ndarray, std_deviations: float = 3.0) -> typing.Tuple[float, float]:
    """

    :param data:
    :param std_deviations:
    :return:
    """
    mean = float(np.mean(data))
    deviance_from_mean = data - mean
    median_absolute_deviation = np.median(np.abs(deviance_from_mean))
    outlier_threshold = std_deviations * median_absolute_deviation

    range_min = np.max((mean - outlier_threshold, np.min(data)))
    range_max = np.min((mean + outlier_threshold, np.max(data)))
    return range_min, range_max


def compute_outliers(data: np.ndarray, data_range: typing.Tuple[float, float]) -> int:
    """
    Compute the number of outliers in a set of data if it is reduced to the given range
    :param data: An array of data values (floats)
    :param data_range: The min and max values for a window on this data
    :return:
    """
    return np.count_nonzero((data >= data_range[0]) & (data < data_range[1]))


def quat_angle(quat):
    """
    Get the angle of rotation indicated by a quaternion, independent of axis
    :param quat:
    :return:
    """
    return 2 * float(np.arccos(min(1, max(-1, quat[0]))))


def quat_diff(q1, q2):
    """
    Find the angle between two quaternions
    Basically, we compose them, and derive the angle from the composition
    :param q1:
    :param q2:
    :return:
    """
    q1 = np.asarray(q1)
    if np.dot(q1, q2) < 0:
        # Quaternions have opposite handedness, flip q1 since it's already an ndarray
        q1 = -1 * q1
    q_inv = q1 * np.array([1.0, -1.0, -1.0, -1.0])
    q_inv = q_inv / np.dot(q_inv, q_inv)

    # We only coare about the scalar component, compose only that
    z0 = q_inv[0] * q2[0] - q_inv[1] * q2[1] - q_inv[2] * q2[2] - q_inv[3] * q2[3]
    return 2 * float(np.arccos(min(1, max(-1, z0))))


def quat_mean(quaternions):
    """
    Find the mean of a bunch of quaternions
    :param quaternions:
    :return:
    """
    if len(quaternions) <= 0:
        return np.nan
    elif len(quaternions) == 1:
        # Only one quaternion, it is the average of itself
        return quaternions[0]
    elif len(quaternions) == 2:
        # We have weird errors for 2 quaternions using the matrix
        # We use the closed form solution given in
        # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
        q1 = np.asarray(quaternions[0])
        q2 = np.asarray(quaternions[1])
        dot = np.dot(q1, q2)
        if dot < 0:
            # The vectors don't have the same handedness, invert one
            q2 = -1 * q2
            dot = -dot
        if dot == 0:
            if q1[0] > q2[0]:
                return q1
            return q2
        z = np.sqrt((q1[0] - q2[2]) * (q1[0] - q2[2]) + 4 * q1[0] * q2[0] * dot * dot)
        result = 2 * q1[0] * dot * q1 + (q2[0] - q1[0] + z) * q2
        return result / np.linalg.norm(result)
    else:
        # Quaternion average from the eigenvectors of the sum matrix
        # See: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
        # We have at least 3 quaternions, make sure they're of the same handedness
        q_mat = np.asarray([
            q if np.dot(q, quaternions[0]) > 0 else -1 * np.asarray(q)
            for q in quaternions
        ])
        product = np.dot(q_mat.T, q_mat)    # Computes sum([q * q.T for q in quaterions])
        evals, evecs = np.linalg.eig(product)
        best = -1
        result = None
        for idx in range(len(evals)):
            if evals[idx] > best:
                best = evals[idx]
                result = evecs[idx]
        if np.any(np.iscomplex(result)):
            # Mean is complex, which means the quaternions are all too close together (I think?)
            # Instead, return the Mode, the most common quaternion
            counts = [
                sum(1 for q2 in quaternions if np.array_equal(q1, q2))
                for q1 in quaternions
            ]
            best = 0
            for idx in range(len(counts)):
                if counts[idx] > best:
                    best = counts[idx]
                    result = quaternions[idx]
            print("Passing off mode as mean with {0} of {1} identical vectors".format(best, len(quaternions)))
        return result
