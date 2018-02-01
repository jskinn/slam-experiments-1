# Copyright (c) 2017, John Skinner
import typing
import os
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe
import base_verify


class OrbslamTUMVerify(base_verify.VerificationExperiment):

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
        # --------- TUM DATASETS -----------
        # Import TUM datasets without using the manager, it is unnecessary
        for folder in [
            'rgbd_dataset_freiburg1_xyz',
            'rgbd_dataset_freiburg1_rpy'
        ]:
            self.import_dataset(
                name="TUM {0}".format(folder),
                module_name='arvet_slam.dataset.tum.tum_loader',
                path=os.path.join('datasets', 'TUM', folder),
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # ORBSLAM2 - Create 2 variants, with different procesing modes
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}:
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

    def get_reference(self) -> typing.List[typing.Tuple[str, str, typing.List[str]]]:
        """
        Get a list of reference passes, and the system & dataset names
        :return: A list of tuples (reference_filename, system_name, dataset_name)
        """
        return [
            ('ORBSLAM2 monocular', 'TUM freiburg1_xyz',
             ['reference-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_xyz-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 rgbd', 'KITTI 00',
             ['reference-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_xyz-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 monocular', 'TUM freiburg1_xyz',
             ['reference-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 rgbd', 'KITTI 00',
             ['reference-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)])
        ]
