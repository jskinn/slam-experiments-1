# Copyright (c) 2017, John Skinner
import typing
import os
import logging
import json
import arvet.util.database_helpers as dh
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe
import arvet_slam.benchmarks.ate.absolute_trajectory_error as ate
import arvet_slam.benchmarks.trajectory_drift.trajectory_drift as traj_drift
import arvet_slam.benchmarks.tracking.tracking_benchmark as tracking_benchmark


class RealWorldExperiment(arvet.batch_analysis.experiment.Experiment):

    def __init__(self, libviso_system=None, orbslam_systems=None,
                 datasets=None,
                 benchmark_rpe=None, benchmark_ate=None, benchmark_trajectory_drift=None, benchmark_tracking=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param libviso_system:
        :param datasets:
        :param benchmark_rpe:
        :param benchmark_ate:
        :param benchmark_trajectory_drift:
        :param id_:
        """
        super().__init__(id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)
        # Systems
        self._libviso_system = libviso_system
        self._orbslam_systems = orbslam_systems if orbslam_systems is not None else {}

        # Image sources
        self._datasets = datasets if datasets is not None else {}

        # Benchmarks
        self._benchmark_rpe = benchmark_rpe
        self._benchmark_ate = benchmark_ate
        self._benchmark_trajectory_drift = benchmark_trajectory_drift
        self._benchmark_tracking = benchmark_tracking

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
        for sequence_num in range(11):
            try:
                path = path_manager.find_dir(os.path.join('datasets', 'KITTI', 'dataset'))
                # Also check the particular sequence exists
                path_manager.find_dir(os.path.join(path, 'sequences', "{0:02}".format(sequence_num)))
            except FileNotFoundError:
                path = None
            if path is not None:
                task = task_manager.get_import_dataset_task(
                    module_name='arvet_slam.dataset.kitti.kitti_loader',
                    path=path,
                    additional_args={'sequence_number': sequence_num},
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='3GB',
                    expected_duration='12:00:00'
                )
                if task.is_finished:
                    name = 'KITTI trajectory {}'.format(sequence_num)
                    self._datasets[name] = task.result
                    self._set_property('datasets.{0}'.format(name), task.result)
                else:
                    task_manager.do_task(task)

        # --------- SYSTEMS -----------
        if self._libviso_system is None:
            self._libviso_system = dh.add_unique(db_client.system_collection, libviso2.LibVisOSystem())
            self._set_property('libviso', self._libviso_system)

        # ORBSLAM2 - Create 9 variants, with different procesing modes
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}:
                name = 'ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower())
                vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
                is_valid_path = True
                try:
                    path_manager.find_file(vocab_path)
                except FileNotFoundError:
                    is_valid_path = False

                if name not in self._orbslam_systems and is_valid_path:
                    orbslam_id = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                        vocabulary_file=vocab_path,
                        mode=sensor_mode,
                        settings={
                            'ORBextractor': {
                                'nFeatures': 1500
                            }
                        }
                    ))
                    self._orbslam_systems[name] = orbslam_id
                    self._set_property('orbslam_systems.{}'.format(name), orbslam_id)

        # --------- BENCHMARKS -----------
        # Create and store the benchmarks for camera trajectories
        # Just using the default settings for now
        if self._benchmark_rpe is None:
            self._benchmark_rpe = dh.add_unique(db_client.benchmarks_collection, rpe.BenchmarkRPE(
                max_pairs=10000,
                fixed_delta=False,
                delta=1.0,
                delta_unit='s',
                offset=0,
                scale_=1))
            self._set_property('benchmark_rpe', self._benchmark_rpe)
        if self._benchmark_ate is None:
            self._benchmark_ate = dh.add_unique(db_client.benchmarks_collection, ate.BenchmarkATE(
                offset=0,
                max_difference=0.2,
                scale=1))
            self._set_property('benchmark_ate', self._benchmark_ate)
        if self._benchmark_trajectory_drift is None:
            self._benchmark_trajectory_drift = dh.add_unique(
                db_client.benchmarks_collection,
                traj_drift.BenchmarkTrajectoryDrift(
                    segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
                    step_size=10
                ))
            self._set_property('benchmark_trajectory_drift', self._benchmark_trajectory_drift)
        if self._benchmark_tracking is None:
            self._benchmark_tracking = dh.add_unique(db_client.benchmarks_collection,
                                                     tracking_benchmark.TrackingBenchmark(initializing_is_lost=True))
            self._set_property('benchmark_tracking', self._benchmark_tracking)

    def schedule_tasks(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       db_client: arvet.database.client.DatabaseClient):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :param task_manager:
        :param db_client:
        :return:
        """
        # Group everything up
        # All systems
        systems = [self._libviso_system] + list(self._orbslam_systems.values())
        # All image datasets
        datasets = list(self._datasets.values())
        # All benchmarks
        benchmarks = [self._benchmark_rpe, self._benchmark_ate,
                      self._benchmark_trajectory_drift, self._benchmark_tracking]

        # Schedule all combinations of systems with the generated datasets
        self.schedule_all(task_manager=task_manager,
                          db_client=db_client,
                          systems=systems,
                          image_sources=datasets,
                          benchmarks=benchmarks)

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
        # Map system ids and simulator ids to printable names
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)

        for dataset_name, dataset_id in self._datasets.items():
            # Collect the trial results for this dataset
            trial_results = {}
            style = {}
            for system_name, system_id in systems.items():
                trial_result_id = self.get_trial_result(system_id, dataset_id)
                if trial_result_id is not None:
                    label = "{0} on {1}".format(system_name, dataset_name)
                    trial_results[label] = trial_result_id
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
                                lower, upper = plot_trajectory(ax, trial_result.get_ground_truth_camera_poses(),
                                                               'ground truth trajectory')
                                mean = (upper + lower) / 2
                                lower = 1.2 * lower - mean
                                upper = 1.2 * upper - mean
                                ax.set_xlim(lower, upper)
                                ax.set_ylim(lower, upper)
                                ax.set_zlim(lower, upper)
                                added_ground_truth = True
                            plot_trajectory(ax, trial_result.get_computed_camera_poses(),
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
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)

        for dataset_name, dataset_id in self._datasets.items():
            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in systems.items():
                trial_result_id = self.get_trial_result(system_id, dataset_id)
                if trial_result_id is not None:
                    label = "{0} on {1}".format(system_name, dataset_name)
                    trial_results[label] = trial_result_id

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
                                first_pose = trajectory[min(trajectory.keys())]
                                json_data['ground_truth'] = [[time] + location_to_json(first_pose.find_relative(pose))
                                                             for time, pose in trajectory.items()]
                            trajectory = trial_result.get_computed_camera_poses()
                            first_pose = trajectory[min(trajectory.keys())]
                            json_data[label] = [[time] + location_to_json(first_pose.find_relative(pose))
                                                for time, pose in trajectory.items()]

                with open('{0}.json'.format(dataset_name), 'w') as json_file:
                    json.dump(json_data, json_file)

    def serialize(self):
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment', 2)

        # Systems
        serialized['libviso'] = self._libviso_system
        serialized['orbslam_systems'] = self._orbslam_systems

        # Image Sources
        serialized['datasets'] = self._datasets

        # Benchmarks
        serialized['benchmark_rpe'] = self._benchmark_rpe
        serialized['benchmark_ate'] = self._benchmark_ate
        serialized['benchmark_trajectory_drift'] = self._benchmark_trajectory_drift
        serialized['benchmark_tracking'] = self._benchmark_tracking

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        update_schema(serialized_representation, db_client)

        # Systems
        if 'libviso' in serialized_representation:
            kwargs['libviso_system'] = serialized_representation['libviso']
        if 'orbslam_systems' in serialized_representation:
            kwargs['orbslam_systems'] = serialized_representation['orbslam_systems']

        # Datasets
        if 'datasets' in serialized_representation:
            kwargs['datasets'] = serialized_representation['datasets']

        # Benchmarks
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']
        if 'benchmark_ate' in serialized_representation:
            kwargs['benchmark_ate'] = serialized_representation['benchmark_ate']
        if 'benchmark_trajectory_drift' in serialized_representation:
            kwargs['benchmark_trajectory_drift'] = serialized_representation['benchmark_trajectory_drift']
        if 'benchmark_tracking' in serialized_representation:
            kwargs['benchmark_tracking'] = serialized_representation['benchmark_tracking']

        return super().deserialize(serialized_representation, db_client, **kwargs)


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


def update_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment')
    pass
