# Copyright (c) 2017, John Skinner
import logging
import typing
import bson
import os.path
import arvet.util.database_helpers as dh
import arvet.util.dict_utils as du
import arvet.util.trajectory_helpers as traj_help
import arvet.database.entity
import arvet.database.client
import arvet.config.path_manager
import arvet.core.system
import arvet.core.image_source
import arvet.core.sequence_type as sequence_type
import arvet.core.benchmark
import arvet.core.image_collection
import arvet.metadata.image_metadata as imeta
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager
import arvet.simulation.unrealcv.unrealcv_simulator as uecv_sim
import arvet.simulation.controllers.trajectory_follow_controller as follow_cont
import arvet_slam.benchmarks.rpe.relative_pose_error
import data_helpers
import trajectory_helpers as th


class GeneratedDataExperiment(arvet.batch_analysis.experiment.Experiment):

    def __init__(self, systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmarks=None, repeats=1,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param simulators:
        :param trajectory_groups:
        :param benchmarks:
        :param trial_map:
        :param result_map:
        :param enabled:
        :param id_:
        """
        super().__init__(id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)
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
    def trajectory_groups(self) -> typing.Mapping[str, 'TrajectoryGroup']:
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
        # Update the trajectory groups
        for trajectory_group in self.trajectory_groups.values():
            self.update_trajectory_group(trajectory_group, task_manager, db_client)

        # Add a RPE benchmark for all experiments
        self.import_benchmark(
            name='Relative Pose Error',
            benchmark=arvet_slam.benchmarks.rpe.relative_pose_error.BenchmarkRPE(max_pairs=0),
            db_client=db_client
        )

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
        self.schedule_all(task_manager=task_manager,
                          db_client=db_client,
                          systems=list(self.systems.values()),
                          image_sources=datasets,
                          benchmarks=list(self.benchmarks.values()),
                          repeats=self._repeats)

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
                       db_client: arvet.database.client.DatabaseClient,
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
        :param db_client: The database client, for performing initial updates
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
                trajectory_group = TrajectoryGroup(name=name, reference_id=task.result, mappings=mappings)
                self._trajectory_groups[name] = trajectory_group
                self.update_trajectory_group(trajectory_group, task_manager, db_client, save_changes=False)
                self._set_property('trajectory_groups.{0}'.format(name), trajectory_group.serialize())
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

    def update_trajectory_group(self, trajectory_group: 'TrajectoryGroup',
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

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot

        colours = ['red', 'green', 'cyan', 'gold', 'magenta', 'brown', 'purple', 'orange',
                   'navy', 'darkkhaki', 'darkgreen', 'crimson']

        # Group and print the trajectories for graphing
        for trajectory_group in self.trajectory_groups.values():
            logging.getLogger(__name__).info("Plotting results for {0} ...".format(trajectory_group.name))

            # Collect the trial results for each image source in this group
            ground_truth_trajectory = None
            for system_name, system_id in self.systems.items():

                show = False
                plot_groups = []
                results_by_world = {}
                computed_trajectories = []

                # Real world trajectory
                trial_result_list = self.get_trial_results(system_id, trajectory_group.reference_dataset)
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None:
                        if ground_truth_trajectory is None:
                            ground_truth_trajectory = th.zero_trajectory(trial_result.get_ground_truth_camera_poses())
                        traj = th.zero_trajectory(trial_result.get_computed_camera_poses())
                        computed_trajectories.append(traj)

                plot_groups.append(('Real world dataset', computed_trajectories, {'c': 'blue'}))

                # Synthetic data trajectories
                for idx, (world_name, quality_map) in enumerate(trajectory_group.generated_datasets.items()):
                    if world_name not in results_by_world:
                        results_by_world[world_name] = {}

                    for qual_idx, (quality_name, dataset_id) in enumerate(quality_map.items()):
                        if quality_name not in results_by_world[world_name]:
                            results_by_world[world_name][quality_name] = []

                        computed_trajectories = []
                        trial_result_list = self.get_trial_results(system_id, dataset_id)
                        for trial_result_id in trial_result_list:
                            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                            if trial_result is not None:
                                show = True

                                if ground_truth_trajectory is None:
                                    ground_truth_trajectory = th.zero_trajectory(
                                        trial_result.get_ground_truth_camera_poses())

                                computed_trajectories.append(th.zero_trajectory(
                                    trial_result.get_computed_camera_poses()))

                                # Get and store the RPE benchmark result for this trial
                                result_id = self.get_benchmark_result(trial_result_id,
                                                                      self.benchmarks['Relative Pose Error'])
                                if result_id is not None:
                                    results_by_world[world_name][quality_name].append(
                                        dh.load_object(db_client, db_client.results_collection, result_id)
                                    )

                        plot_groups.append((world_name + ' ' + quality_name,
                                            computed_trajectories, {
                                                'c': colours[(idx * len(quality_map) + qual_idx) % len(colours)]
                                            }))
                if ground_truth_trajectory is not None:
                    plot_groups.append(('Ground truth', [ground_truth_trajectory], {'c': 'black'}))
                if show or True:
                    logging.getLogger(__name__).info("    creating plot \"Trajectories for {0} on {1}\"".format(
                        system_name, trajectory_group.name))
                    data_helpers.create_axis_plot(
                        title="Trajectories for {0} on {1}".format(system_name, trajectory_group.name),
                        trajectory_groups=plot_groups,
                        save_path=os.path.join('figures', type(self).__name__, 'trajectories')
                    )
                    logging.getLogger(__name__).info("    creating plot \"Error distributions for {0} on {1}\"".format(
                        system_name, trajectory_group.name))
                    create_error_distribution_plot(
                        title="Error distributions for {0} on {1}".format(system_name, trajectory_group.name),
                        results_by_world=results_by_world,
                        save_path=os.path.join('figures', type(self).__name__, 'error distributions')
                    )
                    create_error_vs_time_plot(
                        title="Error vs time for {0} on {1}".format(system_name, trajectory_group.name),
                        results_by_world=results_by_world,
                        save_path=os.path.join('figures', type(self).__name__, 'error vs time')
                    )
        pyplot.show()

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
            kwargs['trajectory_groups'] = {name: TrajectoryGroup.deserialize(s_group, db_client)
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


class TrajectoryGroup:
    """
    A Trajectory Group is a helper structure to manage image datasets grouped by camera trajectory.
    In this class of experiment, it is created from a single reference real-world dataset,
    and produces many synthetic datasets with the same camera trajectory.

    For convenience, it serializes and deserialzes as a group.
    """

    def __init__(self, name: str, reference_id: bson.ObjectId, mappings: typing.List[typing.Tuple[str, dict]],
                 baseline_configuration: dict = None,
                 controller_id: bson.ObjectId = None, generated_datasets: dict = None):
        self.name = name
        self.reference_dataset = reference_id
        self.mappings = mappings
        self.baseline_configuration = baseline_configuration

        self.follow_controller_id = controller_id
        self.generated_datasets = generated_datasets if generated_datasets is not None else {}

    def get_all_dataset_ids(self) -> set:
        """
        Get all the datasets in this group, as a set
        :return:
        """
        return {self.reference_dataset} | set(
            dataset_id
            for quality_map in self.generated_datasets.values()
            for dataset_id in quality_map.values()
        )

    def get_datasets_for_sim(self, sim_name: str) -> typing.Mapping[str, bson.ObjectId]:
        """
        Get all the generated datasets from a particular simulator
        :param sim_name:
        :return:
        """
        return self.generated_datasets[sim_name]

    def get_datasets_for_quality(self, quality_name: str) -> typing.Mapping[str, bson.ObjectId]:
        """
        Get all the generated datasets at a particular quality
        :param quality_name:
        :return:
        """
        return {
            sim_name: quality_map[quality_name]
            for sim_name, quality_map in self.generated_datasets.items()
            if quality_name in quality_map
        }

    def schedule_generation(self, simulators: typing.Mapping[str, bson.ObjectId],
                            quality_variations: typing.List[typing.Tuple[str, dict]],
                            task_manager: arvet.batch_analysis.task_manager.TaskManager,
                            db_client: arvet.database.client.DatabaseClient) -> bool:
        """
        Do imports and dataset generation for this trajectory group.
        Will create a controller, and then generate reduced quality synthetic datasets.
        :param simulators: A Map of simulators, indexed by name
        :param quality_variations: A list of names and quality variations
        :param task_manager:
        :param db_client:
        :return: True if part of the group has changed, and it needs to be re-saved
        """
        changed = False
        # First, make a follow controller for the base dataset if we don't have one.
        # This will be used to generate reduced-quality datasets following the same trajectory
        # as the root dataset
        if self.follow_controller_id is None:
            self.follow_controller_id = follow_cont.create_follow_controller(
                db_client, self.reference_dataset, sequence_type=sequence_type.ImageSequenceType.SEQUENTIAL)
            changed = True

        # Next, if we haven't already, compute baseline configuration from the reference dataset
        if self.baseline_configuration is None or len(self.baseline_configuration) == 0:
            reference_dataset = dh.load_object(db_client, db_client.image_source_collection, self.reference_dataset)
            if isinstance(reference_dataset, arvet.core.image_collection.ImageCollection):
                intrinsics = reference_dataset.get_camera_intrinsics()
                self.baseline_configuration = {
                        # Simulation execution config
                        'stereo_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_stereo_available else 0,
                        'provide_rgb': True,
                        'provide_ground_truth_depth': False,    # We don't care about this
                        'provide_labels': reference_dataset.is_labels_available,
                        'provide_world_normals': reference_dataset.is_normals_available,

                        # Depth settings
                        'provide_depth': reference_dataset.is_depth_available,
                        'depth_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_depth_available else 0,
                        'projector_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_depth_available else 0,

                        # Simulator camera settings, be similar to the reference dataset
                        'resolution': {'width': intrinsics.width, 'height': intrinsics.height},
                        'fov': max(intrinsics.horizontal_fov, intrinsics.vertical_fov),
                        'depth_of_field_enabled': False,
                        'focus_distance': None,
                        'aperture': 2.2,

                        # Quality settings - Maximum quality
                        'lit_mode': True,
                        'texture_mipmap_bias': 0,
                        'normal_maps_enabled': True,
                        'roughness_enabled': True,
                        'geometry_decimation': 0,
                        'depth_noise_quality': 1,

                        # Simulation server config
                        'host': 'localhost',
                        'port': 9000,
                    }
                changed = True

        # Then, for each simulator listed for this trajectory group
        origin_counts = {}
        for sim_name, origin in self.mappings:
            # Count how many times each simulator is used, so we can assign a unique world name to each start point
            if sim_name not in origin_counts:
                origin_counts[sim_name] = 1
            else:
                origin_counts[sim_name] += 1

            # Schedule generation of quality variations that don't exist yet
            if sim_name in simulators:
                # For every quality variation
                for quality_name, config in quality_variations:
                    generate_dataset_task = task_manager.get_generate_dataset_task(
                        controller_id=self.follow_controller_id,
                        simulator_id=simulators[sim_name],
                        simulator_config=du.defaults({'origin': origin}, config, self.baseline_configuration),
                        num_cpus=1,
                        num_gpus=0,
                        memory_requirements='3GB',
                        expected_duration='4:00:00'
                    )
                    if generate_dataset_task.is_finished:
                        world_name = "{0} {1}".format(sim_name, origin_counts[sim_name])
                        if world_name not in self.generated_datasets:
                            self.generated_datasets[world_name] = {}
                        self.generated_datasets[world_name][quality_name] = generate_dataset_task.result
                        changed = True
                    else:
                        task_manager.do_task(generate_dataset_task)
        return changed

    def serialize(self) -> dict:
        serialized = {
            'name': self.name,
            'reference_id': self.reference_dataset,
            'mappings': self.mappings,
            'baseline_configuration': self.baseline_configuration,
            'controller_id': self.follow_controller_id,
            'generated_datasets': self.generated_datasets
        }
        dh.add_schema_version(serialized, 'experiments:visual_slam:BaseGeneratedDataExperiment:TrajectoryGroup', 1)
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation: dict, db_client: arvet.database.client.DatabaseClient) \
            -> 'TrajectoryGroup':
        update_trajectory_group_schema(serialized_representation, db_client)
        return cls(
            name=serialized_representation['name'],
            reference_id=serialized_representation['reference_id'],
            mappings=serialized_representation['mappings'],
            baseline_configuration=serialized_representation['baseline_configuration'],
            controller_id=serialized_representation['controller_id'],
            generated_datasets=serialized_representation['generated_datasets']
        )


def update_trajectory_group_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:BaseGeneratedDataExperiment:TrajectoryGroup')

    # Remove invalid ids
    if 'reference_id' in serialized and \
            not dh.check_reference_is_valid(db_client.image_source_collection, serialized['reference_id']):
        del serialized['reference_id']
    if 'controller_id' in serialized and \
            not dh.check_reference_is_valid(db_client.image_source_collection, serialized['controller_id']):
        del serialized['controller_id']
    if 'generated_datasets' in serialized:

        # Remove invalid dataset ids in each map
        for quality_map in serialized['generated_datasets'].values():
            keys = list(quality_map.keys())
            for key in keys:
                if not dh.check_reference_is_valid(db_client.image_source_collection, quality_map[key]):
                    del quality_map[key]

        # Remove sim names with no results
        keys = [sim_name for sim_name, quality_map in serialized['generated_datasets'].items() if len(quality_map) <= 0]
        for sim_name in keys:
            del serialized['generated_datasets'][sim_name]


def create_error_distribution_plot(title, results_by_world, save_path=None):
    import matplotlib.pyplot as pyplot
    import matplotlib.patches as mpatches
    figure, axes = pyplot.subplots(len(results_by_world), 2, squeeze=False,
                                   figsize=(14, 5 * len(results_by_world)), dpi=80)
    figure.suptitle(title)

    # Pick colours for each quality level, to be consistent across all the graphs
    colours = ['red', 'blue', 'green', 'cyan', 'gold', 'magenta', 'brown', 'purple', 'orange']
    color_idx = 0
    colour_map = {}
    legend_handles = []
    for results_by_quality in results_by_world.values():
        for quality_name in results_by_quality.keys():
            if quality_name not in colour_map:
                colour_map[quality_name] = colours[color_idx]
                color_idx += 1
                legend_handles.append(mpatches.Patch(color=colour_map[quality_name], alpha=0.5, label=quality_name))

    for sim_idx, (sim_name, results_by_quality) in enumerate(results_by_world.items()):
        trans_ax = axes[sim_idx][0]
        trans_ax.set_title('{0} translational error'.format(sim_name))
        trans_ax.set_xlabel('error (m)')
        trans_ax.set_ylabel('frequency')

        rot_ax = axes[sim_idx][1]
        rot_ax.set_title('{0} rotational error'.format(sim_name))
        rot_ax.set_xlabel('error (radians)')
        rot_ax.set_ylabel('frequency')

        for quality_name, results_list in results_by_quality.items():
            trans_errors = []
            rot_errors = []
            for result in results_list:
                trans_errors += list(result.translational_error.values())
                rot_errors += list(result.rotational_error.values())
            trans_ax.hist(trans_errors, 200, alpha=0.5, label=quality_name, facecolor=colour_map[quality_name])
            rot_ax.hist(rot_errors, 200, alpha=0.5, label=quality_name, facecolor=colour_map[quality_name])

    pyplot.figlegend(handles=legend_handles, loc='upper right')
    figure.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # figure.savefig(os.path.join(save_path, title + '.svg'))
        figure.savefig(os.path.join(save_path, title + '.png'))
        pyplot.close(figure)


def create_error_vs_time_plot(title, results_by_world, save_path=None):
    import matplotlib.pyplot as pyplot
    figure, axes = pyplot.subplots(len(results_by_world), 2, squeeze=False,
                                   figsize=(14, 8 * len(results_by_world)), dpi=80)
    figure.suptitle(title)

    # Pick colours for each quality level, to be consistent across all the graphs
    colours = ['red', 'blue', 'green', 'cyan', 'gold', 'magenta', 'brown', 'purple', 'orange']
    color_idx = 0
    colour_map = {}
    for results_by_quality in results_by_world.values():
        for quality_name in results_by_quality.keys():
            if quality_name not in colour_map:
                colour_map[quality_name] = colours[color_idx]
                color_idx += 1

    legend_labels = []
    legend_lines = {}

    for sim_idx, (sim_name, results_by_quality) in enumerate(results_by_world.items()):
        trans_ax = axes[sim_idx][0]
        trans_ax.set_title('{0} translational error'.format(sim_name))
        trans_ax.set_xlabel('time')
        trans_ax.set_ylabel('error (m)')

        rot_ax = axes[sim_idx][1]
        rot_ax.set_title('{0} rotational error'.format(sim_name))
        rot_ax.set_xlabel('time')
        rot_ax.set_ylabel('error (radians)')

        for quality_name, results_list in results_by_quality.items():
            for result in results_list:
                times = sorted(result.translational_error.keys())
                errors = [result.translational_error[t] for t in times]
                line = trans_ax.plot(times, errors, alpha=1 / len(results_list), label=quality_name,
                                     linestyle='None', marker='.', markersize=2, c=colour_map[quality_name])

                times = sorted(result.rotational_error.keys())
                errors = [result.rotational_error[t] for t in times]
                rot_ax.plot(times, errors, alpha=1 / len(results_list), label=quality_name,
                            linestyle='None', marker='.', markersize=2, c=colour_map[quality_name])

                if quality_name not in legend_lines:
                    legend_labels.append(quality_name)
                    legend_lines[quality_name] = line[0]

    pyplot.figlegend([legend_lines[label] for label in legend_labels], legend_labels, loc='upper right')
    figure.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # figure.savefig(os.path.join(save_path, title + '.svg'))
        figure.savefig(os.path.join(save_path, title + '.png'))
        pyplot.close(figure)
