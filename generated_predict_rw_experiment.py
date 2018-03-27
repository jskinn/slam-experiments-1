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


class GeneratedPredictRealWorldExperiment(arvet.batch_analysis.experiment.Experiment):

    def __init__(self, systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmarks=None, repeats=1,
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
        self._repeats = int(repeats)

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
                                                         repeats=self._repeats)

        if not os.path.isdir(type(self).get_output_folder()) or changes > 100:
            task_manager.do_analysis_task(
                experiment_id=self.identifier,
                memory_requirements='12GB',
                expected_duration='2:00:00'
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
        :param db_client:
        :return:
        """
        # First, we need to flip our generated datasets around to map by quality first
        # At the same time, collect all the real world datasets
        real_world_datasets = {}
        generated_datasets_by_quality = {}
        for trajectory_group in self.trajectory_groups.values():
            real_world_datasets[trajectory_group.name] = trajectory_group.reference_dataset
            for world_name, quality_map in trajectory_group.generated_datasets.items():
                for quality_name, dataset_id in quality_map.items():
                    if quality_name not in generated_datasets_by_quality:
                        generated_datasets_by_quality[quality_name] = {}
                    if world_name not in generated_datasets_by_quality[quality_name]:
                        generated_datasets_by_quality[quality_name][world_name] = dataset_id

        # Now, partition our real world datasets into test and validation sets
        # This time, we're going to take one image sequence from each group of datasets (TUM, EuRoC, and KITTI)
        validation, test = partition_by_name(real_world_datasets, {
            'EuRoC MH_03_medium',
            'rgbd_dataset_frieburg2_desk',
            'KITTI trajectory 0'
        })
        # Now, predict the error on the validation set, using the remainder of the real datasets as control
        self.predict_errors(
            systems=self.systems,
            validation_datasets=validation,
            real_world_datasets=test,
            generated_datasets_by_quality=generated_datasets_by_quality,
            per_estimate_benchmark=self.benchmarks['Estimate Errors'],
            output_folder=os.path.join(type(self).get_output_folder(), 'one from each domain'),
            db_client=db_client
        )

        # Partition into different datasets, this time taking all the datasets from a partcular group.
        # Specifically, the KITTI datasets are going to be our validation set,
        # So the only prediction set comes from EuRoC and/or TUM
        validation, test = partition_by_name(real_world_datasets, {
            'KITTI trajectory {}'.format(idx)
            for idx in range(11)
        })
        # Now, predict the error on the validation set, using the remainder of the real datasets as control
        self.predict_errors(
            systems=self.systems,
            validation_datasets=validation,
            real_world_datasets=test,
            generated_datasets_by_quality=generated_datasets_by_quality,
            per_estimate_benchmark=self.benchmarks['Estimate Errors'],
            output_folder=os.path.join(type(self).get_output_folder(), 'cross domain'),
            db_client=db_client
        )

    def collect_observations(self, system_id: bson.ObjectId, dataset_ids: typing.Iterable[bson.ObjectId],
                             benchmark_id: bson.ObjectId, db_client: arvet.database.client.DatabaseClient):
        """
        Collect together error observations of a given system from many image datasets
        :param system_id:
        :param dataset_ids:
        :param benchmark_id:
        :param db_client:
        :return:
        """
        result_ids = [self.get_benchmark_result(system_id, dataset_id, benchmark_id)
                      for dataset_id in dataset_ids]
        results = dh.load_many_objects(db_client, db_client.results_collection,
                                       [result_id for result_id in result_ids if result_id is not None])
        collected_observations = []
        for result in results:
            collected_observations += result.observations.tolist()
        return np.asarray(collected_observations)

    def collect_errors_and_input(self, system_id: bson.ObjectId, dataset_ids: typing.Iterable[bson.ObjectId],
                                 db_client: arvet.database.client.DatabaseClient):
        """
        Collect together error observations of a given system from many image datasets
        :param system_id:
        :param dataset_ids:
        :param db_client:
        :return:
        """
        result_ids = [self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Estimate Errors'])
                      for dataset_id in dataset_ids]
        results = dh.load_many_objects(db_client, db_client.results_collection,
                                       [result_id for result_id in result_ids if result_id is not None])
        collected_errors = []
        collected_characteristics = []
        for result in results:
            # the first 13 values in an estimate error observation are errors,
            # The remainder are the features we're going to use to predict
            collected_errors += result.observations[:13].tolist()
            collected_errors += result.observations[13:].tolist()
        return collected_characteristics, collected_errors

    def predict_errors(self,
                       systems: typing.Mapping[str, bson.ObjectId],
                       validation_datasets: typing.Mapping[str, bson.ObjectId],
                       real_world_datasets: typing.Mapping[str, bson.ObjectId],
                       generated_datasets_by_quality: typing.Mapping[str, typing.Mapping[str, bson.ObjectId]],
                       per_estimate_benchmark: bson.ObjectId,
                       output_folder: str,
                       db_client: arvet.database.client.DatabaseClient):
        if len(validation_datasets) <= 0:
            logging.getLogger(__name__).info("Error, no validation datasets available")
        if len(real_world_datasets) <= 0:
            logging.getLogger(__name__).info("Error, no real world datasets available")
        if len(generated_datasets_by_quality) <= 0:
            logging.getLogger(__name__).info("Error, no generated datasets available")
        for system_name, system_id in systems.items():
            logging.getLogger(__name__).info("Predicting errors for {0} ...".format(system_name))

            # Collect the data we want to predict
            validation = self.collect_errors_and_input(system_id, validation_datasets.values(), db_client)
            if len(validation[0]) <= 0 or len(validation[1]) <= 0:
                logging.getLogger(__name__).info("   No validation data available for {0}".format(system_name))
                continue

            # Predict the error
            train_x, train_y = self.collect_errors_and_input(system_id, real_world_datasets.values(), db_client)
            if len(train_x) <= 0 or len(train_y) <= 0:
                logging.getLogger(__name__).info("   No real world data available for {0}".format(system_name))
                continue
            error = predict(
                data=(train_x, train_y),
                target_data=validation
            )
            logging.getLogger(__name__).info("    MSE from real-world data: {0}".format(error))
            for quality_name, world_map in generated_datasets_by_quality:
                train_x, train_y = self.collect_errors_and_input(system_id, world_map.values(), db_client)
                if len(train_x) <= 0 or len(train_y) <= 0:
                    logging.getLogger(__name__).info("   No data available for {0} on {0}".format(
                        system_name, quality_name))
                    continue
                error = predict(
                    data=(train_x, train_y),
                    target_data=validation
                )
                logging.getLogger(__name__).info("    MSE from {0} data: {1}".format(quality_name, error))

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


def partition_by_name(group: typing.Mapping[str, bson.ObjectId], names_to_include: typing.Set[str]):
    return ({name: oid for name, oid in group.items() if name in names_to_include},
            {name: oid for name, oid in group.items() if name not in names_to_include})


def predict(data, target_data):
    """
    Train on the first set of data, and evaluate on the second set of data.
    Returns the mean squared error on the target data, which is not used during training.
    :param data:
    :param target_data:
    :return:
    """
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import cross_val_score

    train_x, train_y = data
    val_x, val_y = target_data

    model = AdaBoostRegressor(n_estimators=300)
    model.fit(train_x, train_y)
    return cross_val_score(model, val_x, val_y, scoring='neg_mean_squared_error')
