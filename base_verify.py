# Copyright (c) 2017, John Skinner
import typing
import logging
import os.path
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.transform as tf
import arvet.database.client
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import data_helpers


class VerificationExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None, repeats=1,
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
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=repeats,
                         id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)

    def create_plot(self, db_client: arvet.database.client.DatabaseClient, system_name: str, dataset_name: str,
                    reference_filenames: typing.List[str], rescale: bool = False,
                    extra_filenames: typing.List[typing.Tuple[str, typing.List[str], str]] = None):
        if system_name not in self.systems:
            logging.getLogger(__name__).warning("Missing system {0}".format(system_name))
            return
        if dataset_name not in self.datasets:
            logging.getLogger(__name__).warning("Missing dataset {0}".format(dataset_name))
            return
        if extra_filenames is None:
            extra_filenames = []

        trial_result_list = self.get_trial_results(self.systems[system_name], self.datasets[dataset_name])
        reference_trajectories = [load_ref_trajectory(filename) for filename in reference_filenames
                                  if os.path.isfile(filename)]

        computed_trajectories = []
        ground_truth_trajectories = []
        for trial_result_id in trial_result_list:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            if trial_result is not None:
                computed_trajectories.append(zero_trajectory(trial_result.get_computed_camera_poses()))
                if len(ground_truth_trajectories) <= 0:
                    ground_truth_trajectories.append(zero_trajectory(trial_result.get_ground_truth_camera_poses()))

        # Find the scale of the ground truth trajectory
        gt_scale = 1
        rescale = rescale and len(ground_truth_trajectories) >= 1
        if rescale:
            gt_scale = find_trajectory_scale(ground_truth_trajectories[0])
            reference_trajectories = [rescale_trajectory(traj, gt_scale) for traj in reference_trajectories]
            computed_trajectories = [rescale_trajectory(traj, gt_scale) for traj in computed_trajectories]

        extra_trajectory_groups = []
        for group_name, trajectory_files, style in extra_filenames:
            trajectories = [load_ref_trajectory(filename) for filename in trajectory_files if os.path.isfile(filename)]
            if rescale:
                trajectories = [rescale_trajectory(traj, gt_scale) for traj in trajectories]
            extra_trajectory_groups.append((group_name, trajectories, style))

        # Build the graph
        title = "Trajectory for {0} on {1}".format(system_name, dataset_name)
        if rescale:
            title += " (rescaled)"
        data_helpers.create_axis_plot(title, [
            ('locally from example', reference_trajectories, 'b-'),
            ('through framework on HPC', computed_trajectories, 'r--'),
            ('ground truth', ground_truth_trajectories, 'k.')
        ] + extra_trajectory_groups)


def load_ref_trajectory(filename: str, exchange_coordinates=True) -> typing.Mapping[float, tf.Transform]:
    trajectory = {}

    if exchange_coordinates:
        coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                         [-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 0, 1]])
    else:
        coordinate_exchange = np.identity(4)

    first_stamp = None
    with open(filename, 'r') as trajectory_file:
        for line in trajectory_file:
            parts = line.split(' ')
            if len(parts) >= 13:
                # Line is pose as homogenous matrix
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
            elif len(parts) >= 8:
                # Line is pose as transform, followed by quaternion orientation
                stamp, tx, ty, tz, qx, qy, qz, qw = parts[0:8]
                if first_stamp is None:
                    first_stamp = float(stamp)
                trajectory[float(stamp) - first_stamp] = tf.Transform(
                    location=(tz, -tx, -ty),
                    rotation=(qw, qz, -qx, -qy),
                    w_first=True
                )
    return zero_trajectory(trajectory)


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
