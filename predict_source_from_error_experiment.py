# Copyright (c) 2017, John Skinner
import logging
import typing
import bson
import os.path
import numpy as np

import arvet.util.database_helpers as dh
import arvet.util.trajectory_helpers as traj_help
import arvet.database.client
import arvet.config.path_manager
import arvet.core.system
import arvet.core.benchmark
import arvet.metadata.image_metadata as imeta
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager
import arvet.simulation.unrealcv.unrealcv_simulator as uecv_sim

import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2

import data_helpers
import trajectory_group as tg
import euroc_origins
import tum_origins
import kitti_origins
import estimate_errors_benchmark
import frame_errors_benchmark


class PredictSourceFromErrorExperiment(arvet.batch_analysis.experiment.Experiment):

    def __init__(self, systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmarks=None,
                 trial_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param simulators:
        :param trajectory_groups:
        :param benchmarks:
        :param trial_map:
        :param enabled:
        :param id_:
        """
        super().__init__(id_=id_, trial_map=trial_map, enabled=enabled)
        # Systems
        self._systems = systems if systems is not None else {}

        # Image sources
        self._simulators = simulators if simulators is not None else {}
        self._trajectory_groups = trajectory_groups if trajectory_groups is not None else {}

        # Benchmarks
        self._benchmarks = benchmarks if benchmarks is not None else {}

    @property
    def systems(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._systems

    @property
    def simulators(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._simulators

    @property
    def trajectory_groups(self) -> typing.Mapping[str, tg.TrajectoryGroup]:
        return self._trajectory_groups

    @property
    def benchmarks(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._benchmarks

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

        # --------- SIMULATORS -----------
        # Add simulators explicitly, they have different metadata, so we can't just search
        for exe, world_name, environment_type, light_level, time_of_day in [
            (
                    'simulators/AIUE_V01_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_002/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_002', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_003/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_003', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_004/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_004', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_005/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_005', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
                    # ), (
                    #         'simulators/AIUE_V02_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    #         'AIUE_V02_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    #         imeta.TimeOfDay.DAY
            )
        ]:
            self.import_simulator(
                executable_path=exe,
                world_name=world_name,
                environment_type=environment_type,
                light_level=light_level,
                time_of_day=time_of_day,
                db_client=db_client
            )

        # --------- REAL WORLD DATASETS -----------

        # Import EuRoC datasets with lists of trajectory start points for each simulator
        for name, path, mappings in [
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy'), euroc_origins.get_MH_01_easy()),
            ('EuRoC MH_02_easy', os.path.join('datasets', 'EuRoC', 'MH_02_easy'), euroc_origins.get_MH_02_easy()),
            ('EuRoC MH_03_medium', os.path.join('datasets', 'EuRoC', 'MH_03_medium'),
             euroc_origins.get_MH_03_medium()),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult'),
             euroc_origins.get_MH_04_difficult()),
            ('EuRoC MH_05_difficult', os.path.join('datasets', 'EuRoC', 'MH_05_difficult'),
             euroc_origins.get_MH_05_difficult()),
            ('EuRoC V1_01_easy', os.path.join('datasets', 'EuRoC', 'V1_01_easy'), euroc_origins.get_V1_01_easy()),
            ('EuRoC V1_02_medium', os.path.join('datasets', 'EuRoC', 'V1_02_medium'),
             euroc_origins.get_V1_02_medium()),
            ('EuRoC V1_03_difficult', os.path.join('datasets', 'EuRoC', 'V1_03_difficult'),
             euroc_origins.get_V1_03_difficult()),
            ('EuRoC V2_01_easy', os.path.join('datasets', 'EuRoC', 'V2_01_easy'), euroc_origins.get_V2_01_easy()),
            ('EuRoC V2_02_medium', os.path.join('datasets', 'EuRoC', 'V2_02_medium'),
             euroc_origins.get_V2_02_medium()),
            ('EuRoC V2_03_difficult', os.path.join('datasets', 'EuRoC', 'V2_03_difficult'),
             euroc_origins.get_V2_03_difficult())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                name=name,
                mappings=mappings,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import TUM datasets with lists of trajectory start points for each simulator
        for folder, mappings in [
            ('rgbd_dataset_freiburg1_360', tum_origins.get_frieburg1_360()),
            ('rgbd_dataset_frieburg1_rpy', tum_origins.get_frieburg1_rpy()),
            ('rgbd_dataset_frieburg1_xyz', tum_origins.get_frieburg1_xyz()),
            ('rgbd_dataset_frieburg2_desk', tum_origins.get_frieburg2_desk()),
            ('rgbd_dataset_frieburg2_rpy', tum_origins.get_frieburg2_rpy()),
            ('rgbd_dataset_frieburg2_xyz', tum_origins.get_frieburg2_xyz()),
            ('rgbd_dataset_frieburg3_structure_texture_far', tum_origins.get_frieburg3_structure_texture_far()),
            ('rgbd_dataset_frieburg3_walking_xyz', tum_origins.get_frieburg3_walking_xyz())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.tum.tum_loader',
                path=os.path.join('datasets', 'TUM', folder),
                name="TUM {0}".format(folder),
                mappings=mappings,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import KITTI datasets
        for sequence_num in range(11):
            self.import_dataset(
                module_name='arvet_slam.dataset.kitti.kitti_loader',
                name='KITTI trajectory {}'.format(sequence_num),
                path=os.path.join('datasets', 'KITTI', 'dataset'),
                additional_args={'sequence_number': sequence_num},
                mappings=kitti_origins.get_mapping(sequence_num),
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # LibVisO2
        self.import_system(
            name='LibVisO',
            system=libviso2.LibVisOSystem(),
            db_client=db_client
        )

        # ORBSLAM2 - Create 2 variants, stereo and mono
        # These datasets don't have
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.MONOCULAR}:
            self.import_system(
                name='ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()),
                system=orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
                    settings={'ORBextractor': {'nFeatures': 1500}}
                ),
                db_client=db_client
            )

        # --------- BENCHMARKS -----------
        # Add benchmarks to calculate the errors on a per-estimate and per-frame basis
        self.import_benchmark(
            name='Estimate Errors',
            benchmark=estimate_errors_benchmark.EstimateErrorsBenchmark(),
            db_client=db_client
        )
        self.import_benchmark(
            name='Frame Errors',
            benchmark=frame_errors_benchmark.FrameErrorsBenchmark(),
            db_client=db_client
        )

        # --------- TRAJECTORY GROUPS -----------
        # Update the trajectory groups
        # We call this at the end so that any new ones created by import datasets will be updated and saved.
        for trajectory_group in self.trajectory_groups.values():
            self.update_trajectory_group(trajectory_group, task_manager, db_client)

    def schedule_tasks(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       db_client: arvet.database.client.DatabaseClient):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :param task_manager:
        :param db_client:
        :return:
        """
        # All image datasets
        datasets = set()
        for group in self._trajectory_groups.values():
            datasets = datasets | group.get_all_dataset_ids()

        # Schedule all combinations of systems with the generated datasets
        changes, anticipated_changes = self.schedule_all(task_manager=task_manager,
                                                         db_client=db_client,
                                                         systems=list(self.systems.values()),
                                                         image_sources=datasets,
                                                         benchmarks=list(self.benchmarks.values()),
                                                         repeats=10)

        if not os.path.isdir(type(self).get_output_folder()) or changes > 100:
            task_manager.do_analysis_task(
                experiment_id=self.identifier,
                num_cpus=2,
                memory_requirements='32GB',
                expected_duration='12:00:00'
            )

    def import_system(self, name: str, system: arvet.core.system.VisionSystem,
                      db_client: arvet.database.client.DatabaseClient) -> None:
        """
        Import a system into the experiment. It will be run with all the image sources.
        :param name: The name of the system
        :param system: The system object, to serialize and save if necessary
        :param db_client: The database client, to use to save the system
        :return:
        """
        if name not in self._systems:
            self._systems[name] = dh.add_unique(db_client.system_collection, system)
            self._set_property('systems.{0}'.format(name), self._systems[name])

    def import_simulator(self, world_name: str, executable_path: str, environment_type: imeta.EnvironmentType,
                         light_level: imeta.LightingLevel, time_of_day: imeta.TimeOfDay,
                         db_client: arvet.database.client.DatabaseClient) -> None:
        """
        Add a simulator to the experiment
        :param world_name: The world name of the simulator, used as an identifier
        :param executable_path: The path to the executable
        :param environment_type: The environment type of this simulation world
        :param light_level: The light level in this simulation world
        :param time_of_day: The time of day in this simulation world
        :param db_client: The database client, for storing the simulator
        :return: void
        """
        simulator_id = dh.add_unique(db_client.image_source_collection, uecv_sim.UnrealCVSimulator(
            executable_path=executable_path,
            world_name=world_name,
            environment_type=environment_type,
            light_level=light_level,
            time_of_day=time_of_day
        ))
        self._simulators[world_name] = simulator_id
        self._set_property('simulators.{0}'.format(world_name), simulator_id)

    def import_dataset(self, name: str, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       path_manager: arvet.config.path_manager.PathManager,
                       mappings: typing.List[typing.Tuple[str, dict]],
                       module_name: str, path: str, additional_args: dict = None,
                       num_cpus: int = 1, num_gpus: int = 0,
                       memory_requirements: str = '3GB', expected_duration: str = '12:00:00') -> None:
        """
        Import a dataset at a given path, using a given module.
        Has all the arguments of get_import_dataset_task, which are passed through
        :param name: The name to store the dataset as
        :param task_manager: The task manager, for scheduling
        :param path_manager: The path manager, for checking the path
        :param mappings: List of simulator names and origins for this dataset trajectory
        :param module_name: The
        :param path:
        :param additional_args:
        :param num_cpus:
        :param num_gpus:
        :param memory_requirements:
        :param expected_duration:
        :return:
        """
        task = task_manager.get_import_dataset_task(
            module_name=module_name,
            path=path,
            additional_args=additional_args if additional_args is not None else {},
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_requirements=memory_requirements,
            expected_duration=expected_duration
        )
        if task.is_finished:
            if name not in self.trajectory_groups:
                trajectory_group = tg.TrajectoryGroup(name=name, reference_id=task.result, mappings=mappings)
                self._trajectory_groups[name] = trajectory_group
        elif path_manager.check_path(path):
            task_manager.do_task(task)

    def import_benchmark(self, name: str, benchmark: arvet.core.benchmark.Benchmark,
                         db_client: arvet.database.client.DatabaseClient) -> None:
        """
        Import a benchmark, it will be used for all trials
        :param name: The name of the benchmark
        :param benchmark:
        :param db_client:
        :return:
        """
        if name not in self._benchmarks:
            self._benchmarks[name] = dh.add_unique(db_client.benchmarks_collection, benchmark)
            self._set_property('benchmarks.{0}'.format(name), self._benchmarks[name])

    def update_trajectory_group(self, trajectory_group: tg.TrajectoryGroup,
                                task_manager: arvet.batch_analysis.task_manager.TaskManager,
                                db_client: arvet.database.client.DatabaseClient,
                                save_changes: bool = True) -> None:
        """
        Perform updates and imports
        sets of simulators.
        :param trajectory_group:
        :param task_manager:
        :param db_client:
        :param save_changes: Whether we should save any changes to the trajectory group
        :return: void
        """
        quality_variations = [('max quality', {
            # }), ('reduced resolution', {
            #     'resolution': {'width': 256, 'height': 144}  # Extremely low res
            # }), ('narrow fov', {
            #     'fov': 15
            # }), ('no depth-of-field', {
            #     'depth_of_field_enabled': False
            # }), ('no textures', {
            #     'texture_mipmap_bias': 8
            # }), ('no normal maps', {
            #     'normal_maps_enabled': False,  # No normal maps
            # }), ('no reflections', {
            #     'roughness_enabled': False  # No reflections
            # }), ('simple geometry', {
            #     'geometry_decimation': 4,   # Simple geometry
            # }), ('low quality', {
            #     # low quality
            #     'depth_of_field_enabled': False,
            #     'texture_mipmap_bias': 8,
            #     'normal_maps_enabled': False,
            #     'roughness_enabled': False,
            #     'geometry_decimation': 4,
        }), ('worst visual quality', {
            # absolute minimum visual quality, can still reduce FOV and resolution
            'lit_mode': False,
            'depth_of_field_enabled': False,
            'texture_mipmap_bias': 8,
            'normal_maps_enabled': False,
            'roughness_enabled': False,
            'geometry_decimation': 4,
        })]

        # Do the imports for the group, and save any changes
        if trajectory_group.schedule_generation(self.simulators, quality_variations, task_manager, db_client):
            if save_changes:
                self._set_property('trajectory_groups.{0}'.format(trajectory_group.name), trajectory_group.serialize())

    def perform_analysis(self, db_client: arvet.database.client.DatabaseClient):
        """
        Use the results from generated data to try and predict the performance on some selected real world data
        We're going to do this system by system because different types of system use different datasets.
        :param db_client:
        :return:
        """
        random_state = np.random.RandomState(1241)
        euroc_sets = ['EuRoC MH_01_easy', 'EuRoC MH_02_easy', 'EuRoC MH_03_medium', 'EuRoC MH_04_difficult',
                      'EuRoC MH_05_difficult', 'EuRoC V1_01_easy', 'EuRoC V1_02_medium', 'EuRoC V1_03_difficult',
                      'EuRoC V2_01_easy', 'EuRoC V2_02_medium', 'EuRoC V2_03_difficult']
        tum_sets = ['rgbd_dataset_freiburg1_360', 'rgbd_dataset_frieburg1_rpy', 'rgbd_dataset_frieburg1_xyz',
                    'rgbd_dataset_frieburg2_desk', 'rgbd_dataset_frieburg2_rpy', 'rgbd_dataset_frieburg2_xyz',
                    'rgbd_dataset_frieburg3_structure_texture_far', 'rgbd_dataset_frieburg3_walking_xyz']
        kitti_sets = ['KITTI trajectory {}'.format(sequence_num) for sequence_num in range(11)]

        # --------- MONOCULAR -----------
        # In the first experiment, we want to to pick a validation dataset from each group to validate on.
        self.analyse_validation_groups(
            system_name='ORBSLAM2 monocular',
            validation_sets=[{
                random_state.choice(euroc_sets),
                random_state.choice(tum_sets),
                random_state.choice(kitti_sets)
            } for _ in range(10)],
            output_folder=os.path.join(type(self).get_output_folder(), 'ORBSLAM monocular', 'one from each domain'),
            db_client=db_client
        )
        # In the second set, we want to test cross domain evaluation by excluding entire datasets as validation
        self.analyse_validation_groups(
            system_name='ORBSLAM2 monocular',
            validation_sets=[set(euroc_sets), set(tum_sets), set(kitti_sets)],
            output_folder=os.path.join(type(self).get_output_folder(), 'ORBSLAM monocular', 'cross domain'),
            db_client=db_client
        )

        # --------- STEREO -----------
        for system_name in ['ORBSLAM2 stereo', 'LibVisO']:
            # In the first experiment, we want to to pick a validation dataset from each group to validate on.
            self.analyse_validation_groups(
                system_name=system_name,
                validation_sets=[{
                    random_state.choice(euroc_sets),
                    random_state.choice(euroc_sets),
                    random_state.choice(kitti_sets),
                    random_state.choice(kitti_sets)
                } for _ in range(10)],
                output_folder=os.path.join(type(self).get_output_folder(), system_name, 'one from each domain'),
                db_client=db_client
            )
            # In the second set, we want to test cross domain evaluation by excluding entire datasets as validation
            self.analyse_validation_groups(
                system_name='ORBSLAM2 monocular',
                validation_sets=[set(euroc_sets), set(kitti_sets)],
                output_folder=os.path.join(type(self).get_output_folder(), 'ORBSLAM monocular', 'cross domain'),
                db_client=db_client
            )

        # --------- RGB-D -----------
        # For RGB-D, we only have data from one domain (TUM) so we test excluding parts of that.
        self.analyse_validation_groups(
            system_name='ORBSLAM2 rgbd',
            validation_sets=[{
                random_state.choice(tum_sets),
                random_state.choice(tum_sets),
                random_state.choice(tum_sets)
            } for _ in range(10)],
            output_folder=os.path.join(type(self).get_output_folder(), 'ORBSLAM rgbd'),
            db_client=db_client
        )

    def analyse_validation_groups(self, system_name: str, validation_sets: typing.Iterable[typing.Set[str]],
                                  output_folder: str, db_client: arvet.database.client.DatabaseClient):
        import pandas as pd
        import matplotlib.pyplot as pyplot

        if system_name not in self.systems:
            logging.getLogger(__name__).info("Cannot find system \"{0}\"").format(system_name)
            return

        scores_by_quality = {}
        random_state = np.random.RandomState(16323)
        for validation_names in validation_sets:
            output = self.split_datasets_validation_and_training(self.systems[system_name], validation_names)
            validation_real_world_results, training_real_world_results, virtual_datasets_by_results = output

            for quality_name, (validation_virtual_datasets, training_virtual_datasets) in \
                    virtual_datasets_by_results.items():
                if quality_name not in scores_by_quality:
                    scores_by_quality[quality_name] = []

                # Load the data from the results
                training_data = load_data_from_results(training_real_world_results,
                                                       training_virtual_datasets, db_client)
                validation_data = load_data_from_results(validation_real_world_results,
                                                         validation_virtual_datasets, db_client)

                if len(training_data) <= 0 or len(validation_data) <= 0:
                    continue

                # Shuffle the training and validation data, and convert back to np arrays
                random_state.shuffle(training_data)
                random_state.shuffle(validation_data)
                training_data = np.array(training_data)
                validation_data = np.array(validation_data)

                # Do the classification
                score = classify(
                    data=(training_data[:, :-1], training_data[:, -1]),     # Watch these indexes, they're tricky
                    target_data=(validation_data[:, :-1], validation_data[:, -1])
                )
                scores_by_quality[quality_name].append(score)
                logging.getLogger(__name__).info("Output from {0} at {1} can be predicted with F1 score {2}".format(
                    system_name, quality_name, score))

        # Build the pandas dataframe
        df_data = {'quality': [], 'score': []}
        for quality_name, scores in scores_by_quality.items():
            df_data['score'] += scores
            df_data['quality'] += [quality_name for _ in range(len(scores))]
        dataframe = pd.DataFrame(data=df_data)

        # Boxplot the prediction score for each
        os.makedirs(output_folder, exist_ok=True)
        title = "{0} source prediction score".format(system_name)
        figure, ax = pyplot.subplots(1, 1, figsize=(14, 10), dpi=80)
        figure.suptitle(title)

        ax.tick_params(axis='x', rotation=70)
        dataframe.boxplot(column='score', by='quality', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('F1 Score')

        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.90, right=0.99)

        figure.savefig(os.path.join(output_folder, title + '.png'))
        pyplot.close(figure)

    def split_datasets_validation_and_training(self, system_id: bson.ObjectId, validation_datasets: typing.Set[str]) \
            -> typing.Tuple[
                typing.Set[bson.ObjectId],
                typing.Set[bson.ObjectId],
                typing.Mapping[str, typing.Tuple[typing.Set[bson.ObjectId], typing.Set[bson.ObjectId]]]
            ]:
        """
        Split the datasets into 4 groups, based on which world they come from:
        - Training real world datasets
        - Validation real world datasets
        - Training virtual datasets
        - Validataion virtual datasets

        Returns the result ids in each of these groups
        """
        training_real_world_results = set()
        validation_real_world_results = set()
        virtual_results_by_quality = {}
        for trajectory_group in self.trajectory_groups.values():
            result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                  self.benchmarks['Estimate Errors'])
            if result_id is not None:
                if trajectory_group.name in validation_datasets:
                    validation_real_world_results.add(result_id)
                else:
                    training_real_world_results.add(result_id)

            for world_name, quality_map in trajectory_group.generated_datasets.items():
                for quality_name, dataset_id in quality_map.items():
                    if quality_name not in virtual_results_by_quality:
                        virtual_results_by_quality[quality_name] = (set(), set())

                    result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                          self.benchmarks['Estimate Errors'])
                    if result_id is not None:
                        if trajectory_group.name in validation_datasets:
                            virtual_results_by_quality[quality_name][0].add(result_id)
                        else:
                            virtual_results_by_quality[quality_name][1].add(result_id)
        return validation_real_world_results, training_real_world_results, virtual_results_by_quality

    def export_data(self, db_client: arvet.database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to
        :param db_client:
        :return:
        """
        # Save trajectory files for import into unreal
        for trajectory_group in self._trajectory_groups.values():
            data_helpers.dump_ue4_trajectory(
                name=trajectory_group.name,
                trajectory=traj_help.get_trajectory_for_image_source(db_client, trajectory_group.reference_dataset)
            )

        # Group and print the trajectories for graphing
        for trajectory_group in self._trajectory_groups.values():

            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in self.systems.items():
                for dataset_name, dataset_id in trajectory_group.datasets.items():
                    trial_result_list = self.get_trial_result(system_id, dataset_id)
                    for idx, trial_result_id in enumerate(trial_result_list):
                        label = "{0} on {1} repeat {2}".format(system_name, dataset_name, idx)
                        trial_results[label] = trial_result_id
            data_helpers.export_trajectory_as_json(trial_results, "Generated Data " + trajectory_group.name, db_client)

    def serialize(self):
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'experiments:visual_slam:BaseGeneratedDataExperiment', 2)

        # Systems
        serialized['systems'] = self.systems

        # Image Sources
        serialized['simulators'] = self.simulators
        serialized['trajectory_groups'] = {str(name): group.serialize()
                                           for name, group in self._trajectory_groups.items()}

        # Benchmarks
        serialized['benchmarks'] = self.benchmarks

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        update_schema(serialized_representation, db_client)

        # Systems
        if 'systems' in serialized_representation:
            kwargs['systems'] = serialized_representation['systems']

        # Generated datasets
        if 'simulators' in serialized_representation:
            kwargs['simulators'] = serialized_representation['simulators']
        if 'trajectory_groups' in serialized_representation:
            kwargs['trajectory_groups'] = {name: tg.TrajectoryGroup.deserialize(s_group, db_client)
                                           for name, s_group in
                                           serialized_representation['trajectory_groups'].items()}

        # Benchmarks
        if 'benchmarks' in serialized_representation:
            kwargs['benchmarks'] = serialized_representation['benchmarks']

        return super().deserialize(serialized_representation, db_client, **kwargs)


def update_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:BaseGeneratedDataExperiment')

    # Check references
    if 'systems' in serialized:
        keys = list(serialized['systems'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.system_collection, serialized['systems'][key]):
                del serialized['systems'][key]
    if 'simulators' in serialized:
        keys = list(serialized['simulators'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['simulators'][key]):
                del serialized['simulators'][key]
    if 'benchmarks' in serialized:
        keys = list(serialized['benchmarks'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.benchmarks_collection, serialized['benchmarks'][key]):
                del serialized['benchmarks'][key]


def load_data_from_results(real_world_result_ids: typing.Iterable[bson.ObjectId],
                           virtual_result_ids: typing.Iterable[bson.ObjectId],
                           db_client: arvet.database.client.DatabaseClient):
    """
    Given two sets of results, load the observations into a single unified list.
    Each observation has an extra value on the end, 1 for a real world result, and 0 for a virtual result.
    :param real_world_result_ids: The set of real world ids, observations from these results have real==1
    :param virtual_result_ids: The set of virtual result ids, observations have real==0
    :param db_client: The database client for loading
    :return:
    """
    data = []
    for result in dh.load_many_objects(db_client, db_client.results_collection, real_world_result_ids):
        data += np.hstack((result.observations, np.ones((result.observations.shape[0], 1)))).tolist()
    for result in dh.load_many_objects(db_client, db_client.results_collection, virtual_result_ids):
        data += np.hstack((result.observations, np.zeros((result.observations.shape[0], 1)))).tolist()
    return data


def classify(data, target_data):
    """
    Train on the first set of data, and evaluate on the second set of data.
    Returns the mean squared error on the target data, which is not used during training.
    :param data:
    :param target_data:
    :return:
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import Imputer, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score

    train_x, train_y = data
    val_x, val_y = target_data

    # Prune out nans in the output and convert to integer
    valid_indices = np.nonzero(np.invert(np.isnan(train_y)))
    train_x = train_x[valid_indices]
    train_y = np.asarray(train_y[valid_indices], dtype=np.int)
    valid_indices = np.nonzero(np.invert(np.isnan(val_y)))
    val_x = val_x[valid_indices]
    val_y = np.asarray(val_y[valid_indices], dtype=np.int)

    if len(train_y) <= 0 or len(val_y) <= 0:
        return np.nan

    # Build the data processing pipeline, including preprocessing for missing values
    model = Pipeline([
        ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear'))
    ])

    # Fit and evaluate the regressor
    model.fit(train_x, train_y)
    predict_y = model.predict(val_x)
    return f1_score(val_y, predict_y)
