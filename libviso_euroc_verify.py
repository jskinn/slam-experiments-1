# Copyright (c) 2017, John Skinner
import os
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.task_manager
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2
import base_verify


class LibVisOEuRoCVerify(base_verify.VerificationExperiment):

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
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=1,
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
        # --------- EuRoC DATASETS -----------
        # import specific EuRoC datasets that we have reference results for
        for name, path in [
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy')),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult')),
        ]:
            self.import_dataset(
                name=name,
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        self.import_system(name='LibVisO2', system=libviso2.LibVisOSystem(), db_client=db_client)

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot

        for system_name, dataset_name, reference_trajectories in [
            ('LibVisO2', 'EuRoC MH_01_easy',
             ['libviso-trajectories/trajectory-EuRoC-MH_01_easy-{0}.txt'.format(idx) for idx in range(1, 11)]),
            ('LibVisO2', 'EuRoC MH_04_difficult',
             ['libviso-trajectories/trajectory-EuRoC-MH_04_difficult-{0}.txt'.format(idx) for idx in range(1, 11)])
        ]:
            self.create_plot(
                db_client=db_client,
                system_name=system_name,
                dataset_name=dataset_name,
                reference_filenames=reference_trajectories
            )
        pyplot.show()
