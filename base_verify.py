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

    def get_reference(self) -> typing.List[typing.Tuple[str, str, typing.List[str], typing.List[str]]]:
        """
        Get a list of reference passes, and the system & dataset names
        :return: A list of tuples (system_name, dataset_name, reference_filenames, extra filenames)
        """
        return []

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot

        # Visualize the different trajectories in each group
        for system_name, dataset_name, reference_filenames, extra_filenames in self.get_reference():
            if system_name not in self.systems:
                logging.getLogger(__name__).warning("Missing system {0}".format(system_name))
                continue
            if dataset_name not in self.datasets:
                logging.getLogger(__name__).warning("Missing dataset {0}".format(dataset_name))
                continue

            trial_result_list = self.get_trial_results(self.systems[system_name], self.datasets[dataset_name])
            reference_trajectories = [load_ref_trajectory(filename) for filename in reference_filenames
                                      if os.path.isfile(filename)]
            extra_trajectories = [load_ref_trajectory(filename) for filename in extra_filenames
                                  if os.path.isfile(filename)]
            computed_trajectories = []
            for trial_result_id in trial_result_list:
                trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                if trial_result is not None:
                    computed_trajectories.append(trial_result.get_computed_camera_poses())

            data_helpers.create_axis_plot("Trajectory for {0} on {1}".format(system_name, dataset_name), [
                ('locally from example', reference_trajectories, 'b-'),
                ('through framework on HPC', computed_trajectories, 'r--'),
                ('locally without delays', extra_trajectories, 'g--')
            ])
        pyplot.show()


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
