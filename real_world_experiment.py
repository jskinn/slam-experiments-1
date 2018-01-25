# Copyright (c) 2017, John Skinner
import typing
import os
import logging
import json
import arvet.util.database_helpers as dh
import arvet.util.transform as tf
import arvet.database.client
import arvet.util.unreal_transform as uetf
import arvet.util.trajectory_helpers as traj_help
import arvet.config.path_manager
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.dataset.tum.tum_manager
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe
import arvet_slam.benchmarks.ate.absolute_trajectory_error as ate
import arvet_slam.benchmarks.trajectory_drift.trajectory_drift as traj_drift
import arvet_slam.benchmarks.tracking.tracking_benchmark as tracking_benchmark
import data_helpers


class RealWorldExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

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

        self.import_system(name='LibVisO', system=libviso2.LibVisOSystem(), db_client=db_client)

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
        self._plot_trajectories(db_client)

    def _plot_trajectories(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the ground-truth and computed trajectories for each system for each trajectory.
        This is important for validation
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        logging.getLogger(__name__).info("Plotting trajectories...")

        for dataset_name, dataset_id in self.datasets.items():
            # Collect the trial results for this dataset
            trial_results = {}
            style = {}
            for system_name, system_id in self.systems.items():
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) > 0:
                    label = "{0} on {1}".format(system_name, dataset_name)
                    trial_results[label] = trial_result_list[0]
                    style[label] = '--' if dataset_name == 'reference dataset' else '-'

            # Make sure we have at least one result to plot
            if len(trial_results) > 1:
                figure = pyplot.figure(figsize=(14, 10), dpi=80)
                figure.suptitle("Computed trajectories for {0}".format(dataset_name))
                ax = figure.add_subplot(111, projection='3d')
                ax.set_xlabel('x-location')
                ax.set_ylabel('y-location')
                ax.set_zlabel('z-location')
                ax.plot([0], [0], [0], 'ko', label='origin')
                added_ground_truth = False

                # For each trial result
                for label, trial_result_id in trial_results.items():
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None:
                        if trial_result.success:
                            if not added_ground_truth:
                                lower, upper = data_helpers.plot_trajectory(
                                    ax, trial_result.get_ground_truth_camera_poses(), 'ground truth trajectory')
                                mean = (upper + lower) / 2
                                lower = 1.2 * lower - mean
                                upper = 1.2 * upper - mean
                                ax.set_xlim(lower, upper)
                                ax.set_ylim(lower, upper)
                                ax.set_zlim(lower, upper)
                                added_ground_truth = True
                            data_helpers.plot_trajectory(ax, trial_result.get_computed_camera_poses(),
                                                         label=label,
                                                         style=style[label])
                        else:
                            print("Got failed trial: {0}".format(trial_result.reason))

                logging.getLogger(__name__).info("... plotted trajectories for {0}".format(dataset_name))
                ax.legend()
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

        for dataset_name, dataset_id in self.datasets.items():

            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in self.systems.items():
                trial_result_list = self.get_trial_result(system_id, dataset_id)
                for idx, trial_result_id in enumerate(trial_result_list):
                    label = "{0} on {1} repeat {2}".format(system_name, dataset_name, idx)
                    trial_results[label] = trial_result_id
            data_helpers.export_trajectory_as_json(trial_results, "Real World " + dataset_name, db_client)
