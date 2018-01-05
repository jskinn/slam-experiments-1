# Copyright (c) 2017, John Skinner
import typing
import os
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.transform as tf
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe


class OrbslamKITTIVerify(arvet.batch_analysis.experiment.Experiment):

    def __init__(self, orbslam_mono=None, orbslam_stereo=None,
                 datasets=None,
                 benchmark_rpe=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param orbslam_mono:
        :param orbslam_stereo:
        :param datasets:
        :param benchmark_rpe:
        :param id_:
        """
        super().__init__(id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)
        # Systems
        self._orbslam_mono = orbslam_mono
        self._orbslam_stereo = orbslam_stereo

        # Image sources
        self._datasets = datasets if datasets is not None else {}

        # Benchmarks
        self._benchmark_rpe = benchmark_rpe

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
        # --------- KITTI DATASETS -----------
        # import specific kitti datasets that we have reference results for
        for sequence_num in {0}:
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
        # ORBSLAM2 - Create 2 variants, with different procesing modes
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        try:
            path_manager.find_file(vocab_path)
        except FileNotFoundError:
            vocab_path = None

        if vocab_path is not None and self._orbslam_mono is None:
            self._orbslam_mono = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                vocabulary_file=vocab_path,
                mode=orbslam2.SensorMode.MONOCULAR,
                settings={
                    'ORBextractor': {
                        'nFeatures': 2000,
                        'scaleFactor': 1.2,
                        'nLevels': 8,
                        'iniThFAST': 20,
                        'minThFAST': 7
                    }
                }
            ))
            self._set_property('orbslam_mono', self._orbslam_mono)

        if vocab_path is not None and self._orbslam_stereo is None:
            self._orbslam_stereo = dh.add_unique(db_client.system_collection, orbslam2.ORBSLAM2(
                vocabulary_file=vocab_path,
                mode=orbslam2.SensorMode.STEREO,
                settings={
                    'ThDepth': 35,
                    'ORBextractor': {
                        'nFeatures': 2000,
                        'scaleFactor': 1.2,
                        'nLevels': 8,
                        'iniThFAST': 20,
                        'minThFAST': 7
                    }
                }
            ))
            self._set_property('orbslam_stereo', self._orbslam_stereo)

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
        systems = [self._orbslam_mono, self._orbslam_stereo]
        # All image datasets
        datasets = list(self._datasets.values())
        # All benchmarks
        benchmarks = [self._benchmark_rpe]

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
        for system_id, reference_filename, name in [
            (self._orbslam_mono, 'trajectory-kitti-00-mono.txt', 'mono'),
            (self._orbslam_stereo, 'trajectory-kitti-00-stereo.txt', 'stereo')
        ]:
            trial_result_id = self.get_trial_result(system_id, self._datasets['KITTI trajectory 0'])
            if trial_result_id is not None:
                trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                if trial_result is not None:
                    plot_difference(trial_result.get_computed_camera_poses(), reference_filename,
                                    '{0} on KITTI sequence 00'.format(name))

    def serialize(self):
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment', 2)

        # Systems
        serialized['orbslam_mono'] = self._orbslam_mono
        serialized['orbslam_stereo'] = self._orbslam_stereo

        # Image Sources
        serialized['datasets'] = self._datasets

        # Benchmarks
        serialized['benchmark_rpe'] = self._benchmark_rpe

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        update_schema(serialized_representation, db_client)

        # Systems
        if 'orbslam_mono' in serialized_representation:
            kwargs['orbslam_mono'] = serialized_representation['orbslam_mono']
        if 'orbslam_stereo' in serialized_representation:
            kwargs['orbslam_stereo'] = serialized_representation['orbslam_stereo']

        # Datasets
        if 'datasets' in serialized_representation:
            kwargs['datasets'] = serialized_representation['datasets']

        # Benchmarks
        if 'benchmark_rpe' in serialized_representation:
            kwargs['benchmark_rpe'] = serialized_representation['benchmark_rpe']

        return super().deserialize(serialized_representation, db_client, **kwargs)


def update_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:VisualSlamExperiment')

    # Clean out invalid ids
    if 'orbslam_mono' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['orbslam_mono']):
        del serialized['orbslam_mono']
    if 'orbslam_stereo' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['orbslam_stereo']):
        del serialized['orbslam_stereo']
    if 'datasets' in serialized:
        keys = list(serialized['datasets'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['datasets'][key]):
                del serialized['datasets'][key]
    if 'benchmark_rpe' in serialized and \
            not dh.check_reference_is_valid(db_client.system_collection, serialized['benchmark_rpe']):
        del serialized['benchmark_rpe']


def plot_difference(computed_trajectory: typing.Mapping[float, tf.Transform], reference_filename: str, name: str):
    import matplotlib.pyplot as pyplot

    reference_trajectory = load_ref_trajectory(reference_filename)

    comp_stamps = set(computed_trajectory.keys())
    reference_stamps = set(reference_trajectory.keys())
    missing_ref = reference_stamps - comp_stamps
    if len(missing_ref) > 0:
        print("missing reference stamps: {0}".format(missing_ref))
    extra_stamps = comp_stamps - reference_stamps
    if len(extra_stamps) > 0:
        print("extra computed stamps: {0}".format(extra_stamps))

    times = sorted(comp_stamps & reference_stamps)
    x = []
    y = []
    z = []
    qx = []
    qy = []
    qz = []
    qw = []
    for stamp in times:
        comp_pose = computed_trajectory[stamp]
        ref_pose = reference_trajectory[stamp]
        diff = comp_pose.location - ref_pose.location
        x.append(abs(diff[0]))
        y.append(abs(diff[1]))
        z.append(abs(diff[2]))
        diff = comp_pose.rotation_quat(True) - ref_pose.rotation_quat(True)
        qw.append(abs(diff[0]))
        qx.append(abs(diff[1]))
        qy.append(abs(diff[2]))
        qz.append(abs(diff[3]))
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
    pyplot.show()


def load_ref_trajectory(filename: str) -> typing.Mapping[float, tf.Transform]:
    trajectory = {}
    coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
    with open(filename, 'r') as trajectory_file:
        for line in trajectory_file:
            parts = line.split(' ')
            if len(parts) >= 13:
                stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = parts[0:13]
                pose = np.matrix([
                    [r00, r01, r02, t0],
                    [r10, r11, r12, t1],
                    [r20, r21, r22, t2],
                    [0, 0, 0, 1]
                ])
                pose = np.dot(np.dot(coordinate_exchange, pose), coordinate_exchange.T)
                trajectory[stamp] = tf.Transform(pose)
    return trajectory
