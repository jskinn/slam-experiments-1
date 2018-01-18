# Copyright (c) 2017, John Skinner
import typing
import os
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.transform as tf
import arvet.util.associate as ass
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe


class OrbslamKITTIVerify(arvet.batch_analysis.simple_experiment.SimpleExperiment):

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
        # --------- KITTI DATASETS -----------
        # import specific kitti datasets that we have reference results for
        for sequence_num in {0}:
            self.import_dataset(
                name='KITTI {0:02}'.format(sequence_num),
                module_name='arvet_slam.dataset.kitti.kitti_loader',
                path=os.path.join('datasets', 'KITTI', 'dataset'),
                additional_args={'sequence_number': sequence_num},
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # ORBSLAM2 - Create 2 variants, with different procesing modes
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.MONOCULAR}:
            self.import_system(
                name='ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()),
                db_client=db_client,
                system=orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
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


def plot_difference(computed_trajectory: typing.Mapping[float, tf.Transform], reference_filename: str, name: str):
    import matplotlib.pyplot as pyplot

    reference_trajectory = load_ref_trajectory(reference_filename)

    matches = ass.associate(reference_trajectory, computed_trajectory, offset=0, max_difference=0.000001)

    missing_ref = set(reference_trajectory.keys()) - {m[0] for m in matches}
    if len(missing_ref) > 0:
        print("missing reference stamps: {0}".format(missing_ref))
    extra_stamps = set(computed_trajectory.keys()) - {m[1] for m in matches}
    if len(extra_stamps) > 0:
        print("extra computed stamps: {0}".format(extra_stamps))

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
        diff = comp_pose.location - ref_pose.location
        times.append(ref_stamp)
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
    ax.legend()
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
                    [float(r00), float(r01), float(r02), float(t0)],
                    [float(r10), float(r11), float(r12), float(t1)],
                    [float(r20), float(r21), float(r22), float(t2)],
                    [0, 0, 0, 1]
                ])
                pose = np.dot(np.dot(coordinate_exchange, pose), coordinate_exchange.T)
                trajectory[float(stamp)] = tf.Transform(pose)
    return trajectory
