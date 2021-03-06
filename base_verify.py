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
import trajectory_helpers as th


class VerificationExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None, repeats=1,
                 trial_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param datasets:
        :param benchmarks:
        :param trial_map:
        :param enabled:
        :param id_:
        """
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=repeats,
                         id_=id_, trial_map=trial_map, enabled=enabled)

    def create_plot(self, db_client: arvet.database.client.DatabaseClient, system_name: str, dataset_name: str,
                    reference_filenames: typing.List[str], rescale: bool = False,
                    extra_filenames: typing.List[typing.Tuple[str, typing.List[str], dict]] = None):
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
                computed_trajectories.append(th.zero_trajectory(trial_result.get_computed_camera_poses()))
                if len(ground_truth_trajectories) <= 0:
                    ground_truth_trajectories.append(th.zero_trajectory(trial_result.get_ground_truth_camera_poses()))

        # Find the scale of the ground truth trajectory
        gt_scale = 1
        rescale = rescale and len(ground_truth_trajectories) >= 1
        if rescale:
            gt_scale = th.find_trajectory_scale(ground_truth_trajectories[0])
            reference_trajectories = [th.rescale_trajectory(traj, gt_scale) for traj in reference_trajectories]
            computed_trajectories = [th.rescale_trajectory(traj, gt_scale) for traj in computed_trajectories]

        extra_trajectory_groups = []
        for group_name, trajectory_files, style in extra_filenames:
            trajectories = [load_ref_trajectory(filename, ref_timestamps=sorted(reference_trajectories[0].keys()))
                            for filename in trajectory_files if os.path.isfile(filename)]
            if rescale:
                trajectories = [th.rescale_trajectory(traj, gt_scale) for traj in trajectories]
            extra_trajectory_groups.append((group_name, trajectories, style))

        # Build the graph
        title = "Trajectory for {0} on {1}".format(system_name, dataset_name)
        if rescale:
            title += " (rescaled)"
        data_helpers.create_axis_plot(title, [
            ('locally from example', reference_trajectories, {'c': 'blue', 'linestyle': '-', 'marker': 'None'}),
            ('through framework on HPC', computed_trajectories, {'c': 'red', 'linestyle': '--', 'marker': 'None'}),
            ('ground truth', ground_truth_trajectories, {'c': 'black'})
        ] + extra_trajectory_groups, os.path.join('figures', type(self).__name__))


def load_ref_trajectory(filename: str, exchange_coordinates=True, ref_timestamps=None) \
        -> typing.Mapping[float, tf.Transform]:
    trajectory = {}

    if exchange_coordinates:
        coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                         [-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 0, 1]])
    else:
        coordinate_exchange = np.identity(4)

    first_stamp = None
    line_num = 0
    with open(filename, 'r') as trajectory_file:
        for line in trajectory_file:
            line_num += 1
            parts = line.split(' ')
            if len(parts) >= 12:
                # Line is pose as homogenous matrix
                if len(parts) >= 13:
                    stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = parts[0:13]
                else:
                    # Sometimes the stamp is missing, presumably an error in orbslam
                    r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = parts[0:12]
                    if ref_timestamps is not None and line_num < len(ref_timestamps):
                        stamp = float(ref_timestamps[line_num])
                    else:
                        print("Missing timestamp number {0}".format(line_num))
                        continue
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
                    location=(float(tz), -float(tx), -float(ty)),
                    rotation=(float(qw), float(qz), -float(qx), -float(qy)),
                    w_first=True
                )
    return th.zero_trajectory(trajectory)
