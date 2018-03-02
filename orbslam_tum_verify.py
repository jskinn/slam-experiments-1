# Copyright (c) 2017, John Skinner
import os
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.task_manager
import arvet_slam.systems.slam.orbslam2 as orbslam2
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
            'rgbd_dataset_freiburg1_desk'
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
                            'nFeatures': 1000,
                            'scaleFactor': 1.2,
                            'nLevels': 8,
                            'iniThFAST': 20,
                            'minThFAST': 7
                        }
                    }
                )
            )

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot

        for system_name, dataset_name, rescale, reference_trajectories, external_trajectories in [
            ('ORBSLAM2 monocular', 'TUM rgbd_dataset_freiburg1_xyz', True,
             ['orbslam-trajectories/trajectory-TUM-rgbd_dataset_frieburg1_xyz-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)],
             ['orbslam-external-trajectories/trajectory-TUM-rgbd_dataset_frieburg1_xyz-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 rgbd', 'TUM rgbd_dataset_freiburg1_xyz', False,
             ['orbslam-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_xyz-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)],
             ['orbslam-external-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_xyz-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 monocular', 'TUM rgbd_dataset_freiburg1_desk', True,
             ['orbslam-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)],
             ['orbslam-external-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-mono-{0}.txt'.format(idx)
              for idx in range(1, 11)]),
            ('ORBSLAM2 rgbd', 'TUM rgbd_dataset_freiburg1_desk', False,
             ['orbslam-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)],
             ['orbslam-external-trajectories/trajectory-TUM-rgbd_dataset_freiburg1_desk-rgbd-{0}.txt'.format(idx)
              for idx in range(1, 11)])
        ]:
            self.create_plot(
                db_client=db_client,
                system_name=system_name,
                dataset_name=dataset_name,
                reference_filenames=reference_trajectories,
                rescale=rescale,
                extra_filenames=[
                    ('run on someone else\'s PC', external_trajectories, {'c': 'cyan', 'linestyle': '--'})
                ]
            )
        pyplot.show()
