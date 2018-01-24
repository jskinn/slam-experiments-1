# Copyright (c) 2017, John Skinner
import typing
import os
import logging
import json
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.util.associate as ass
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.dataset.tum.tum_manager
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe
import arvet_slam.benchmarks.ate.absolute_trajectory_error as ate
import arvet_slam.benchmarks.trajectory_drift.trajectory_drift as traj_drift
import arvet_slam.benchmarks.tracking.tracking_benchmark as tracking_benchmark


class OrbslamConsistencyExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

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
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=10,
                         id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)

    def do_imports(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                   path_manager: arvet.config.path_manager.PathManager,
                   db_client: arvet.database.client.DatabaseClient):
        """
        Import image sources for evaluation in this experiment
        :param task_manager: The task manager, for creating import tasks
        :param path_manager: The path manager, for resolving file system paths
        :param db_client: The database client, for saving declared objects too small to need a task
        :return:
        """
        # --------- REAL WORLD DATASETS -----------
        # Import KITTI datasets
        for sequence_num in range(11):
            self.import_dataset(
                name='KITTI {0:02}'.format(sequence_num),
                module_name='arvet_slam.dataset.kitti.kitti_loader',
                path=os.path.join('datasets', 'KITTI', 'dataset'),
                additional_args={'sequence_number': sequence_num},
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import EuRoC datasets
        for name, path in [
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy')),
            ('EuRoC MH_02_easy', os.path.join('datasets', 'EuRoC', 'MH_02_easy')),
            ('EuRoC MH_02_medium', os.path.join('datasets', 'EuRoC', 'MH_03_medium')),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult')),
            ('EuRoC MH_05_difficult', os.path.join('datasets', 'EuRoC', 'MH_05_difficult')),
            ('EuRoC V1_01_easy', os.path.join('datasets', 'EuRoC', 'V1_01_easy')),
            ('EuRoC V1_02_medium', os.path.join('datasets', 'EuRoC', 'V1_02_medium')),
            ('EuRoC V1_03_difficult', os.path.join('datasets', 'EuRoC', 'V1_03_difficult')),
            ('EuRoC V2_01_easy', os.path.join('datasets', 'EuRoC', 'V2_01_easy')),
            ('EuRoC V2_02_medium', os.path.join('datasets', 'EuRoC', 'V2_02_medium')),
            ('EuRoC V2_03_difficult', os.path.join('datasets', 'EuRoC', 'V2_03_difficult'))
        ]:
            self.import_dataset(
                name=name,
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import TUM datasets without using the manager, it is unnecessary
        for folder in arvet_slam.dataset.tum.tum_manager.dataset_names:
            self.import_dataset(
                name="TUM {0}".format(folder),
                module_name='arvet_slam.dataset.tum.tum_loader',
                path=os.path.join('datasets', 'TUM', folder),
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # ORBSLAM2 - Create 3 variants, with different procesing modes
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}:
            self.import_system(
                name='ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()),
                db_client=db_client,
                system=orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
                    settings={'ORBextractor': {'nFeatures': 1500}}
                )
            )

        # --------- BENCHMARKS -----------
        # Create and store the benchmarks for camera trajectories
        # Just using the default settings for now
        self.import_benchmark(
            name='Relative Pose Error',
            db_client=db_client,
            benchmark=rpe.BenchmarkRPE(
                max_pairs=10000,
                fixed_delta=False,
                delta=1.0,
                delta_unit='s',
                offset=0,
                scale_=1
            )
        )
        self.import_benchmark(
            name='Absolute Trajectory Error',
            db_client=db_client,
            benchmark=ate.BenchmarkATE(
                offset=0,
                max_difference=0.2,
                scale=1
            )
        )
        self.import_benchmark(
            name='Trajectory Drift',
            db_client=db_client,
            benchmark=traj_drift.BenchmarkTrajectoryDrift(
                segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
                step_size=10
            )
        )
        self.import_benchmark(
            name='Tracking Statistics',
            db_client=db_client,
            benchmark=tracking_benchmark.TrackingBenchmark(initializing_is_lost=True)
        )

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_variations(db_client)

    def _plot_variations(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the ground-truth and computed trajectories for each system for each trajectory.
        This is important for validation
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        logging.getLogger(__name__).info("Plotting variations...")

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... variations for {0}".format(system_name))

            # Collect statistics on all the trajectories
            times = []
            trans_precision = []
            trans_error = []
            distance = []
            distance_to_prev_frame = []
            for dataset_name, dataset_id in self.datasets.items():

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_traj = None
                computed_trajectories = []
                timestamps = []

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_traj is None:
                            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
                        traj = trial_result.get_computed_camera_poses()
                        computed_trajectories.append(traj)
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_traj, traj)})

                # Now that we have all the trajectories, we can measure consistency
                prev_location = None
                total_distance = 0
                for time in sorted(ground_truth_traj.keys()):
                    # Find the mean estimated location for this time
                    mean_estimate = np.mean([computed_trajectories[idx][timestamps[idx][time]].location
                                             for idx in range(len(computed_trajectories))
                                             if time in timestamps[idx]])

                    # Find the distance to the prev frame
                    current_location = ground_truth_traj[time].location
                    if prev_location is not None:
                        to_prev_frame = np.linalg.norm(current_location - prev_location)
                    else:
                        to_prev_frame = 0
                    total_distance += to_prev_frame
                    prev_location = current_location

                    # For each computed pose at this time
                    for idx in range(len(computed_trajectories)):
                        if time in timestamps[idx]:
                            computed_location = computed_trajectories[idx][timestamps[idx][time]].location
                            times.append(time)
                            trans_precision.append(np.linalg.norm(mean_estimate - computed_location))
                            trans_error.append(np.linalg.norm(current_location - computed_location))
                            distance.append(total_distance)
                            distance_to_prev_frame.append(distance_to_prev_frame)

            # Plot precision vs error
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} precision vs error".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('error')
            ax.set_ylabel('precision')
            ax.plot(trans_error, trans_precision)

            # Plot precision vs frame distance
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} precision vs frame distance".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('distance to previous frame')
            ax.set_ylabel('precision')
            ax.plot(distance_to_prev_frame, trans_precision)

            # Plot precision vs total distance
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} precision vs total distance".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('total distance travelled')
            ax.set_ylabel('precision')
            ax.plot(distance, trans_precision)

            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)
            pyplot.show()

    def export_data(self, db_client: arvet.database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to
        :param db_client:
        :return:
        """
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)

        for dataset_name, dataset_id in self._datasets.items():
            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in systems.items():
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) > 0:
                    label = "{0} on {1}".format(system_name, dataset_name)
                    trial_results[label] = trial_result_list[0]

            # Make sure we have at least one result to plot
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

                with open('{0}.json'.format(dataset_name), 'w') as json_file:
                    json.dump(json_data, json_file)


def plot_trajectory(axis, trajectory, label, style='-'):
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
