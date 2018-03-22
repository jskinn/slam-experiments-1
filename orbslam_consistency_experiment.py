# Copyright (c) 2017, John Skinner
import os
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.dataset.tum.tum_manager
import base_consistency_experiment


class OrbslamConsistencyExperiment(base_consistency_experiment.BaseConsistencyExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None,
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
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks,
                         id_=id_, trial_map=trial_map, enabled=enabled)

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
