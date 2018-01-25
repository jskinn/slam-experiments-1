import typing
import json
import bson
import arvet.util.database_helpers as dh
import arvet.util.unreal_transform as uetf
import arvet.util.transform as tf
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
