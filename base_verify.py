# Copyright (c) 2017, John Skinner
import typing
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.transform as tf
import arvet.util.associate as ass
import arvet.database.client
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager


class VerificationExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param datasets:
        :param benchmarks:
        :param trial_map:
        :param result_map:
        :param enabled:
        :param id_:
        """
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks,
                         id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)

    def get_reference(self) -> typing.List[typing.Tuple[str, str, typing.List[str]]]:
        """
        Get a list of reference passes, and the system & dataset names
        :return: A list of tuples (reference_filename, system_name, dataset_name)
        """
        return []

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        for system_name, dataset_name, reference_filenames in self.get_reference():
            trial_result_list = self.get_trial_results(self.systems[system_name], self.datasets[dataset_name])
            reference_trajectories = [load_ref_trajectory(filename) for filename in reference_filenames]
            computed_trajectories = []
            for trial_result_id in trial_result_list:
                trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                if trial_result is not None:
                    computed_trajectories.append(trial_result.get_computed_camera_poses())
            plot_difference(reference_trajectories, computed_trajectories,
                            '{0} on {1}'.format(system_name, dataset_name))


def plot_difference(reference_trajectories: typing.List[typing.Mapping[float, tf.Transform]],
                    computed_trajectories: typing.List[typing.Mapping[float, tf.Transform]],
                    name: str) -> None:
    import matplotlib.pyplot as pyplot

    reference_keys = [{
        first_time: other_time
        for first_time, other_time in ass.associate(reference_trajectories[0], reference_trajectories[idx],
                                                    offset=0, max_difference=0.001)
    } for idx in range(len(reference_trajectories))]
    computed_keys = [{
        ref_time: comp_time
        for ref_time, comp_time in ass.associate(reference_trajectories[0], computed_trajectories[idx],
                                                 offset=0, max_difference=0.001)
    } for idx in range(len(computed_trajectories))]

    x = list(reference_trajectories[0].keys())
    min_diffs = []
    max_diffs = []
    rot_min_diffs = []
    rot_max_diffs = []
    for idx in range(len(x)):
        for ref_idx in range(len(reference_trajectories)):
            for comp_idx in range(len(computed_trajectories)):
                ref_pose = reference_trajectories[ref_idx][reference_keys[x[idx]]]
                comp_pose = computed_trajectories[comp_idx][computed_keys[x[idx]]]
                diff = np.linalg.norm(ref_pose.location - comp_pose.location)
                rot_diff = np.linalg.norm(ref_pose.rotation_quat(True) - comp_pose.rotation_quat(True))

                if len(min_diffs) <= idx:
                    min_diffs.append(diff)
                else:
                    min_diffs[idx] = min(min_diffs[idx], diff)

                if len(max_diffs) <= idx:
                    max_diffs.append(diff)
                else:
                    max_diffs[idx] = max(max_diffs[idx], diff)

                if len(rot_min_diffs) <= idx:
                    rot_min_diffs.append(rot_diff)
                else:
                    rot_min_diffs[idx] = min(rot_min_diffs[idx], rot_diff)

                if len(rot_max_diffs) <= idx:
                    rot_max_diffs.append(rot_diff)
                else:
                    rot_max_diffs[idx] = max(rot_max_diffs[idx], rot_diff)

    figure = pyplot.figure(figsize=(14, 10), dpi=80)
    figure.suptitle("Difference in trajectories for {0}".format(name))

    # Plot the change in the location
    ax = figure.add_subplot(121)
    ax.set_xlabel('time')
    ax.set_ylabel('difference')
    ax.plot(x, min_diffs, label='min')
    ax.plot(x, max_diffs, label='max')
    ax.legend()

    # Plot the change in the rotation
    ax = figure.add_subplot(122)
    ax.set_xlabel('time')
    ax.set_ylabel('difference')
    ax.plot(x, rot_min_diffs, label='min')
    ax.plot(x, rot_max_diffs, label='max')
    ax.legend()

    pyplot.show()


def plot_axis_difference(reference_trajectory, computed_trajectory, name):
    import matplotlib.pyplot as pyplot

    # ref_mean_location, ref_std_location, ref_mean_rot, ref_std_rot = create_aggregate_trajectory(reference_trajectories)
    # comp_mean_location, comp_std_location, comp_mean_rot, comp_std_rot = \
    #    create_aggregate_trajectory(computed_trajectories)
    
    # Match reference to computed timestamps
    matches = ass.associate(reference_trajectory, computed_trajectory, offset=0, max_difference=0.001)

    times = []
    x = []
    y = []
    z = []
    qx = []
    qy = []
    qz = []
    qw = []
    for ref_stamp, comp_stamp in matches:
        ref_pose = reference_trajectory[ref_stamp]
        comp_pose = computed_trajectory[comp_stamp]
        diff = ref_pose.location - comp_pose.location
        times.append(ref_stamp)
        x.append(diff[0])
        y.append(diff[1])
        z.append(diff[2])
        diff = ref_pose.rotation_quat(True) - comp_pose.rotation_quat(True)
        qw.append(diff[0])
        qx.append(diff[1])
        qy.append(diff[2])
        qz.append(diff[3])
    figure = pyplot.figure(figsize=(14, 10), dpi=80)
    figure.suptitle("Difference in trajectories for {0}".format(name))
    ax = figure.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel('abosolute difference')
    ax.plot(times, x, label='x')
    ax.plot(times, y, label='y')
    ax.plot(times, z, label='z')
    ax.plot(times, qw, label='qw')
    ax.plot(times, qx, label='qx')
    ax.plot(times, qy, label='qy')
    ax.plot(times, qz, label='qz')
    ax.legend()
    pyplot.show()


def create_aggregate_trajectory(trajectories: typing.List[typing.Mapping[float, tf.Transform]]) \
        -> typing.Tuple[typing.Mapping[float, np.ndarray], typing.Mapping[float, np.ndarray],
                        typing.Mapping[float, np.ndarray], typing.Mapping[float, np.ndarray]]:
    """

    :param trajectories:
    :return:
    """
    matches = [{k: v} for idx in range(1, len(trajectories))
               for k, v in ass.associate(trajectories[0], trajectories[idx], max_difference=0.0001, offset=0).items()]

    location_mean_trajectory = {}
    location_std_trajectory = {}
    orientation_mean_trajectory = {}
    orientation_std_trajectory = {}
    for time in trajectories[0].keys():
        points = [trajectories[0][time].location] + [
            trajectories[idx][matches[idx][time]].location
            for idx in range(1, len(trajectories))
            if time in matches[idx]
        ]
        orientations = [trajectories[0][time].rotation_quat(True)] + [
            trajectories[idx][matches[idx][time]].rotation_quat(True)
            for idx in range(1, len(trajectories))
            if time in matches[idx]
        ]
        points = np.asarray(points)
        location_mean_trajectory[time] = np.mean(points)
        location_std_trajectory[time] = np.std(points)
        orientation_mean_trajectory[time] = np.mean(orientations)
        orientation_std_trajectory[time] = np.std(orientations)
    return location_mean_trajectory, location_std_trajectory, orientation_mean_trajectory, orientation_std_trajectory


def load_ref_trajectory(filename: str) -> typing.Mapping[float, tf.Transform]:
    trajectory = {}
    coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
    first_stamp = None
    with open(filename, 'r') as trajectory_file:
        for line in trajectory_file:
            parts = line.split(' ')
            if len(parts) >= 13:
                stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = parts[0:13]
                if first_stamp is None:
                    first_stamp = float(stamp)
                pose = np.matrix([
                    [float(r00), float(r01), float(r02), float(t0)],
                    [float(r10), float(r11), float(r12), float(t1)],
                    [float(r20), float(r21), float(r22), float(t2)],
                    [0, 0, 0, 1]
                ])
                pose = np.dot(np.dot(coordinate_exchange, pose), coordinate_exchange.T)
                trajectory[float(stamp) - first_stamp] = tf.Transform(pose)
    return trajectory
