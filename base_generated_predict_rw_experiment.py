# Copyright (c) 2017, John Skinner
import logging
import typing
import bson
import os.path
import json
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

import data_helpers
import trajectory_group as tg
import estimate_errors_benchmark
import estimate_trial_errors_benchmark
import frame_errors_benchmark


class BaseGeneratedPredictRealWorldExperiment(arvet.batch_analysis.experiment.Experiment):

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
        self.import_benchmark(
            name='Trial Errors',
            benchmark=estimate_trial_errors_benchmark.EstimateTrialErrorsBenchmark(),
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
                expected_duration='96:00:00'
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
        # Do the imports for the group, and save any changes
        if trajectory_group.schedule_generation(
                self.simulators, self.get_quality_variations(), task_manager, db_client):
            if save_changes:
                self._set_property('trajectory_groups.{0}'.format(trajectory_group.name), trajectory_group.serialize())

    def get_quality_variations(self) -> typing.List[typing.Tuple[str, dict]]:
        return []

    def analyse_ks_score(self, system_name: str, output_folder: str, dataset_names: typing.Set[str],
                         db_client: arvet.database.client.DatabaseClient, results_cache: dict):
        from scipy.stats import ks_2samp

        if system_name not in self.systems:
            logging.getLogger(__name__).info("Cannot find system \"{0}\"".format(system_name))
            return
        system_id = self.systems[system_name]
        os.makedirs(output_folder, exist_ok=True)

        # Preload results
        result_ids = set()
        for trajectory_group in self.trajectory_groups.values():
            if trajectory_group.name not in dataset_names:
                continue
            result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                  self.benchmarks['Trial Errors'])
            if result_id is not None:
                result_ids.add(result_id)

                for quality_map in trajectory_group.generated_datasets.values():
                    for quality_name, dataset_id in quality_map.items():
                        result_id = self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Trial Errors'])
                        if result_id is not None:
                            result_ids.add(result_id)
        results = dh.load_many_objects(db_client, db_client.results_collection, result_ids - set(results_cache.keys()))
        for result in results:
            results_cache[result.identifier] = result

        lines = []
        rw_to_rw_cache = {}
        json_data = {}
        # For each error, measure ks similarity between
        for get_error, error_name, units, is_integer, bounds in [
            (lambda errs: np.array(errs)[:, 3], 'translation error', 'm', False, (0, None)),
            (lambda errs: np.array(errs)[:, 5], 'orientation error', 'radians', False, (0, np.pi)),
            #(lambda errs: np.array(errs)[:, 9], 'translation consistency', 'm', False, (0, None)),
            #(lambda errs: np.array(errs)[:, 11], 'orientation consistency', 'rad', False, (0, np.pi)),
            #(lambda errs: np.array(errs)[:, 14], 'feature matches', None, True, (0, None))
        ]:
            print(error_name + ':')
            lines.append(error_name + ':')
            json_data[error_name] = {}

            # For each trajectory group
            for trajectory_group in self.trajectory_groups.values():
                if trajectory_group.name not in dataset_names:
                    continue

                print('  {0}:'.format(trajectory_group.name))
                lines.append('  {0}:'.format(trajectory_group.name))
                json_data[error_name][trajectory_group.name] = {}

                result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                      self.benchmarks['Trial Errors'])
                all_rw_errors = {}
                inter_run_scores = []
                if result_id is not None and result_id in results_cache:
                    result = results_cache[result_id]
                    trial_ids = sorted(result.errors_by_trial.keys())
                    for i in range(len(trial_ids) - 1):
                        rw_error = get_error(result.errors_by_trial[trial_ids[i]])
                        all_rw_errors[trial_ids[i]] = (rw_error)
                        for j in range(i + 1, len(trial_ids)):
                            ks_stat, _ = ks_2samp(
                                rw_error,
                                get_error(result.errors_by_trial[trial_ids[j]])
                            )
                            inter_run_scores.append(ks_stat)
                    print('    between runs: {0}'.format(sorted(inter_run_scores)))
                    lines.append('    between runs: {0}'.format(sorted(inter_run_scores)))
                    json_data[error_name][trajectory_group.name]['self'] = inter_run_scores

                other_rw_scores = []
                if len(all_rw_errors) > 0:
                    for other_trajectory_group in self.trajectory_groups.values():
                        if other_trajectory_group.name != trajectory_group.name and \
                                other_trajectory_group.name in dataset_names:
                            result_id = self.get_benchmark_result(system_id, other_trajectory_group.reference_dataset,
                                                                  self.benchmarks['Trial Errors'])
                            if result_id is not None and result_id in results_cache:
                                result = results_cache[result_id]
                                for other_trial_id, other_rw_errors in result.errors_by_trial.items():
                                    for rw_trial_id, rw_error in all_rw_errors.items():
                                        if other_trial_id in rw_to_rw_cache and \
                                                rw_trial_id in rw_to_rw_cache[other_trial_id]:
                                            ks_stat = rw_to_rw_cache[other_trial_id][rw_trial_id]
                                        elif rw_trial_id in rw_to_rw_cache and \
                                                other_trial_id in rw_to_rw_cache[rw_trial_id]:
                                            ks_stat = rw_to_rw_cache[rw_trial_id][other_trial_id]
                                        else:
                                            ks_stat, _ = ks_2samp(
                                                rw_error,
                                                get_error(other_rw_errors)
                                            )
                                            if other_trial_id not in rw_to_rw_cache:
                                                rw_to_rw_cache[other_trial_id] = {}
                                            if rw_trial_id not in rw_to_rw_cache:
                                                rw_to_rw_cache[rw_trial_id] = {}
                                            if other_trial_id not in rw_to_rw_cache[rw_trial_id]:
                                                rw_to_rw_cache[rw_trial_id][other_trial_id] = ks_stat
                                            if rw_trial_id not in rw_to_rw_cache[other_trial_id]:
                                                rw_to_rw_cache[other_trial_id][rw_trial_id] = ks_stat

                                        other_rw_scores.append(ks_stat)
                    print('    other real world: {0}'.format(sorted(other_rw_scores)))
                    lines.append('    other real world: {0}'.format(sorted(other_rw_scores)))
                    json_data[error_name][trajectory_group.name]['other real'] = other_rw_scores

                scores_by_quality = {}
                for quality_map in trajectory_group.generated_datasets.values():
                    for quality_name, dataset_id in quality_map.items():
                        result_id = self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Trial Errors'])
                        if result_id is not None and result_id in results_cache:
                            result = results_cache[result_id]
                            if quality_name not in scores_by_quality:
                                scores_by_quality[quality_name] = []
                            for sim_errors in result.errors_by_trial.values():
                                for rw_error in all_rw_errors.values():
                                    ks_stat, _ = ks_2samp(
                                        rw_error,
                                        get_error(sim_errors)
                                    )
                                    scores_by_quality[quality_name].append(ks_stat)
                for quality_name, scores in scores_by_quality.items():
                    print('    {0}: {1}'.format(quality_name, sorted(scores)))
                    lines.append('    {0}: {1}'.format(quality_name, sorted(scores)))
                    json_data[error_name][trajectory_group.name][quality_name] = scores
        with open(os.path.join(output_folder, 'self_to_sim.txt'), 'w') as fp:
            fp.write('\n'.join(lines))
        with open(os.path.join(output_folder, 'self_to_sim.json'), 'w') as fp:
            json.dump(json_data, fp)

    def analyse_distributions(self, system_name: str, output_folder: str,
                              db_client: arvet.database.client.DatabaseClient, results_cache: dict):
        if system_name not in self.systems:
            logging.getLogger(__name__).info("Cannot find system \"{0}\"".format(system_name))
            return
        system_id = self.systems[system_name]
        os.makedirs(output_folder, exist_ok=True)

        all_errors_by_quality = {}
        for trajectory_group in self.trajectory_groups.values():
            result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                  self.benchmarks['Estimate Errors'])
            if result_id is None:
                continue
            errors = collect_all_behaviour({result_id}, db_client, results_cache)

            if 'Real World' not in all_errors_by_quality:
                all_errors_by_quality['Real World'] = errors
            else:
                all_errors_by_quality['Real World'] = np.vstack((all_errors_by_quality['Real World'], errors))

            for quality_map in trajectory_group.generated_datasets.values():
                for quality_name, dataset_id in quality_map.items():
                    result_id = self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Estimate Errors'])
                    if result_id is not None:
                        errors = collect_all_behaviour({result_id}, db_client, results_cache)

                        if quality_name not in all_errors_by_quality:
                            all_errors_by_quality[quality_name] = errors
                        else:
                            all_errors_by_quality[quality_name] = np.vstack(
                                (all_errors_by_quality[quality_name], errors))

        create_distribution_plots(
            system_name=system_name,
            group_name='all data',
            errors_by_quality=all_errors_by_quality,
            output_folder=output_folder
        )
        # create_tracking_plot(
        #     system_name=system_name,
        #     group_name='all data',
        #     errors_by_quality=all_errors_by_quality,
        #     output_folder=output_folder,
        # )

    def analyse_validation_groups(self, system_name: str, validation_sets: typing.Iterable[typing.Set[str]],
                                  output_folder: str, db_client: arvet.database.client.DatabaseClient,
                                  results_cache: dict):
        if system_name not in self.systems:
            logging.getLogger(__name__).info("Cannot find system \"{0}\"".format(system_name))
            return
        group_predictions = {}
        for validation_set_names in validation_sets:
            logging.getLogger(__name__).info("Predicting errors for {0} ...".format(system_name))
            output = self.split_datasets_validation_and_training(self.systems[system_name], validation_set_names)
            validation_real_world_datasets, training_real_world_datasets, virtual_datasets_by_quality = output

            # Check we've actually got data to use
            if len(validation_real_world_datasets) <= 0:
                logging.getLogger(__name__).info("Error, no validation datasets available")
                continue
            if len(training_real_world_datasets) <= 0:
                logging.getLogger(__name__).info("Error, no real world datasets available")
                continue
            if len(virtual_datasets_by_quality) <= 0:
                logging.getLogger(__name__).info("Error, no generated datasets available")
                continue

            rw_errors, virtual_data_errors = predict_real_and_virtual_errors(
                validation_results=validation_real_world_datasets,
                real_world_results=training_real_world_datasets,
                virtual_results_by_quality=virtual_datasets_by_quality,
                db_client=db_client,
                results_cache=results_cache
            )
            if 'Real World' not in group_predictions:
                group_predictions['Real World'] = rw_errors
            else:
                for idx in range(len(group_predictions['Real World'])):
                    group_predictions['Real World'][idx] += rw_errors[idx]

            for group_name, errors in virtual_data_errors.items():
                if group_name not in group_predictions:
                    group_predictions[group_name] = errors
                else:
                    for idx in range(len(group_predictions[group_name])):
                        group_predictions[group_name][idx] += errors[idx]

        # plot the errors
        os.makedirs(output_folder, exist_ok=True)
        create_errors_plots(
            indexes_and_names=[
                (0, '{0} forward accuracy'.format(system_name), 'm'),
                (1, '{0} sideways accuracy'.format(system_name), 'm'),
                (2, '{0} vertical accuracy'.format(system_name), 'm'),
                (3, '{0} translational accuracy length'.format(system_name), 'm'),
                (5, '{0} rotational accuracy'.format(system_name), 'radians'),
                (6, '{0} forward precision'.format(system_name), 'm'),
                (7, '{0} sideways precision'.format(system_name), 'm'),
                (8, '{0} vertical precision'.format(system_name), 'm'),
                (9, '{0} translational precision length'.format(system_name), 'm'),
                (11, '{0} rotational precision'.format(system_name), 'rad')
            ],
            group_predictions=group_predictions,
            output_folder=output_folder
        )

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
        training_real_world_datasets = set()
        validation_real_world_datasets = set()
        virtual_datasets_by_quality = {}
        for trajectory_group in self.trajectory_groups.values():
            result_id = self.get_benchmark_result(system_id, trajectory_group.reference_dataset,
                                                  self.benchmarks['Estimate Errors'])
            if result_id is not None:
                if trajectory_group.name in validation_datasets:
                    validation_real_world_datasets.add(result_id)
                else:
                    training_real_world_datasets.add(result_id)

            for world_name, quality_map in trajectory_group.generated_datasets.items():
                for quality_name, dataset_id in quality_map.items():
                    if quality_name not in virtual_datasets_by_quality:
                        virtual_datasets_by_quality[quality_name] = (set(), set())

                    result_id = self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Estimate Errors'])
                    if result_id is not None:
                        if trajectory_group.name in validation_datasets:
                            virtual_datasets_by_quality[quality_name][0].add(result_id)
                        else:
                            virtual_datasets_by_quality[quality_name][1].add(result_id)
        return validation_real_world_datasets, training_real_world_datasets, virtual_datasets_by_quality

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


def partition_by_name(group: typing.Mapping[str, bson.ObjectId], names_to_include: typing.Set[str]) \
        -> typing.Tuple[typing.Mapping[str, bson.ObjectId], typing.Mapping[str, bson.ObjectId]]:
    """
    Split named is into two groups based on their names
    First group only includes the given names, and the second group everything else.
    :param group:
    :param names_to_include:
    :return:
    """
    return ({name: oid for name, oid in group.items() if name in names_to_include},
            {name: oid for name, oid in group.items() if name not in names_to_include})


def collect_errors_and_input(result_ids: typing.Iterable[bson.ObjectId],
                             db_client: arvet.database.client.DatabaseClient,
                             results_cache: dict):
    """
    Collect together error observations from a given set of result ids
    :param result_ids:
    :param db_client:
    :param results_cache:
    :return:
    """
    # Add the results we don't already have to the results cache
    result_ids = set(result_ids)
    results = dh.load_many_objects(db_client, db_client.results_collection, result_ids - set(results_cache.keys()))
    for result in results:
        results_cache[result.identifier] = result.observations

    # Then pull the errors from the cache
    collected_errors = []
    collected_characteristics = []
    for result_id in result_ids:
        # the first 13 values in an estimate error observation are errors,
        # The remainder are the features we're going to use to predict
        collected_errors += results_cache[result_id][:, :13].tolist()
        collected_characteristics += results_cache[result_id][:, 13:].tolist()
    return np.array(collected_characteristics), np.array(collected_errors)


def collect_all_behaviour(result_ids: typing.Iterable[bson.ObjectId],
                          db_client: arvet.database.client.DatabaseClient,
                          results_cache: dict):
    """
    Collect together error observations from a given set of result ids
    :param result_ids:
    :param db_client:
    :param results_cache:
    :return:
    """
    # Add the results we don't already have to the results cache
    result_ids = set(result_ids)
    results = dh.load_many_objects(db_client, db_client.results_collection, result_ids - set(results_cache.keys()))
    for result in results:
        results_cache[result.identifier] = result.observations

    # Then pull the errors from the cache
    collected_behaviour = []
    for result_id in result_ids:
        # Take all the behaviour characteristics together
        collected_behaviour += results_cache[result_id].tolist()
    return np.array(collected_behaviour)


def predict_real_and_virtual_errors(validation_results: typing.Iterable[bson.ObjectId],
                                    real_world_results: typing.Iterable[bson.ObjectId],
                                    virtual_results_by_quality: typing.Mapping[
                                        str, typing.Tuple[typing.Set[bson.ObjectId],
                                                          typing.Set[bson.ObjectId]]
                                    ],
                                    db_client: arvet.database.client.DatabaseClient,
                                    results_cache: dict) \
        -> typing.Tuple[typing.List[typing.List[typing.Tuple[float, float]]],
                        typing.Mapping[str, typing.List[typing.List[typing.Tuple[float, float]]]]]:

    # Load the validation data
    val_x, val_y = collect_errors_and_input(validation_results, db_client, results_cache)
    if len(val_x) <= 0 or len(val_y) <= 0:
        logging.getLogger(__name__).info("   No validation data available")
        return [], {}

    # Predict the error using real world data
    train_x, train_y = collect_errors_and_input(real_world_results, db_client, results_cache)
    if len(train_x) <= 0 or len(train_y) <= 0:
        logging.getLogger(__name__).info("   No real world data available")
        real_world_scores = []
    else:
        logging.getLogger(__name__).info("    predicting from real-world data: ...")
        real_world_scores = predict_errors(
            data=(train_x, train_y),
            target_data=(val_x, val_y)
        )

    # Predict the error using different groups of virtual data
    # When choosing our training set, we have three choices,
    # We can use all the virtual data,
    # exclude the virtual data with the same path as the validation set,
    # or train only on virtual data using the same trajectory as the validation set.
    errors_by_group = {}
    for quality_name, (validation_virtual_datasets, training_virtual_datasets) in virtual_results_by_quality.items():
        logging.getLogger(__name__).info("    predicting from {0} data: ...".format(quality_name))

        # First, all data
        train_x, train_y = collect_errors_and_input(validation_virtual_datasets |
                                                    training_virtual_datasets, db_client, results_cache)
        if len(train_x) > 0 and len(train_y) > 0:
            errors_by_group['{0} all data'.format(quality_name)] = predict_errors(
                data=(train_x, train_y),
                target_data=(val_x, val_y)
            )

        # Missing same trajectory as validation set
        train_x, train_y = collect_errors_and_input(training_virtual_datasets, db_client, results_cache)
        if len(train_x) > 0 and len(train_y) > 0:
            errors_by_group['{0} no validation trajectory'.format(quality_name)] = predict_errors(
                data=(train_x, train_y),
                target_data=(val_x, val_y)
            )

        # Only using data with the same trajectory as the validation set
        train_x, train_y = collect_errors_and_input(validation_virtual_datasets, db_client, results_cache)
        if len(train_x) > 0 and len(train_y) > 0:
            errors_by_group['{0} only validation trajectory'.format(quality_name)] = predict_errors(
                data=(train_x, train_y),
                target_data=(val_x, val_y)
            )
    return real_world_scores, errors_by_group


def predict_errors(data, target_data) -> typing.List[typing.List[typing.Tuple[float, float]]]:
    """
    Try and predict each dimension of the output data separately, from the same input.
    Returns a list that is the width of the output,
    :param data: a tuple of n,i training samples with n,j true outputs
    :param target_data: a tuple of m,i validation samples with m,j true outputs
    :return: A list of length j,m,2, consisting of m pairs of predicted/true values for each of j errors
    """
    scores = []
    train_x, train_y = data
    val_x, val_y = target_data
    assert train_y.shape[1] == val_y.shape[1]
    for idx in range(train_y.shape[1]):
        if idx == 12:
            # metric 12 is whether or not the system is lost, a binary value, so we classify
            scores.append(predict_classification((train_x, train_y[:, idx]), (val_x, val_y[:, idx])))
        else:
            # All other values are real-valued, so we do regression.
            scores.append(predict_regression((train_x, train_y[:, idx]), (val_x, val_y[:, idx])))
    return scores


def predict_regression(data, target_data) -> typing.List[typing.Tuple[float, float]]:
    """
    Train on the first set of data, and evaluate on the second set of data.
    Returns the mean squared error on the target data, which is not used during training.
    This performs regression using an Support Vector Machine, use when the value it learn in continuous
    :param data:
    :param target_data:
    :return:
    """
    # from sklearn.svm import SVR
    from sklearn.preprocessing import Imputer, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDRegressor

    train_x, train_y = data
    val_x, val_y = target_data

    # Prune out nans in the output
    valid_indices = np.nonzero(np.invert(np.isnan(train_y)))
    train_x = train_x[valid_indices]
    train_y = train_y[valid_indices]
    valid_indices = np.nonzero(np.invert(np.isnan(val_y)))
    val_x = val_x[valid_indices]
    val_y = val_y[valid_indices]

    if len(train_y) <= 0 or len(val_y) <= 0:
        return []

    # Build the data processing pipeline, including preprocessing for missing values
    model = Pipeline([
        ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ('scaler', StandardScaler()),
        # ('regressor', SVR(kernel='rbf'))
        ('regressor', SGDRegressor(loss='huber', tol=0.001, max_iter=1000, shuffle=True))
    ])

    # Fit and evaluate the regressor
    model.fit(train_x, train_y)
    predict_y = model.predict(val_x)
    return list(zip(predict_y, val_y))


def predict_classification(data, target_data) -> typing.List[typing.Tuple[float, float]]:
    """
    Train on the first set of data, and evaluate on the second set of data.
    Returns the mean squared error on the target data, which is not used during training.
    This performs classificaion, use when the
    :param data:
    :param target_data:
    :return: A list of pairs of predicted and true values
    """
    # from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import Imputer, StandardScaler
    from sklearn.pipeline import Pipeline

    train_x, train_y = data
    val_x, val_y = target_data

    # Prune out nans in the output, and convert to integers
    valid_indices = np.nonzero(np.invert(np.isnan(train_y)))
    train_x = train_x[valid_indices]
    train_y = np.asarray(train_y[valid_indices], dtype=np.int)
    valid_indices = np.nonzero(np.invert(np.isnan(val_y)))
    val_x = val_x[valid_indices]
    val_y = np.asarray(val_y[valid_indices], dtype=np.int)

    if len(train_y) <= 0 or len(val_y) <= 0 or np.all(train_y == train_y[0]) or np.all(val_y == val_y[0]):
        return []

    # Build the data processing pipeline, including preprocessing for missing values
    model = Pipeline([
        ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ('scaler', StandardScaler()),
        # ('classifier', SVC(kernel='rbf'))
        ('classifier', SGDClassifier(loss='hinge', tol=0.001, max_iter=1000, shuffle=True))
    ])

    # Fit and evaluate the regressor
    model.fit(train_x, train_y)
    predict_y = model.predict(val_x)
    return list(zip(predict_y, val_y))


QUALITY_NAME_COLOURS = {
    'real world': (0, 0, 0),
    'max quality': (255 / 255, 128 / 255, 0 / 255),
    'min quality': (0, 0 / 255, 128 / 255),
}
QUALITY_NAME_ORDER = ['Real World', 'min quality', 'max quality']
QUALITY_NAME_ALPHA = {
    'min quality': 0.5,
    'max quality': 0.3,
    'real world': 1
}


def compute_zoom(all_data):
    middle = np.median(all_data)
    abs_diff = np.abs(all_data - middle)
    avg_diff = np.median(abs_diff)
    if avg_diff == 0:
        avg_diff = np.mean(abs_diff)
    all_data = all_data[abs_diff < 100 * avg_diff]  # Remove ridiculous outliers
    middle = np.mean(all_data)
    std = np.std(all_data)
    return np.max((np.min(all_data), middle - 3 * std)), np.min((np.max(all_data), middle + 3 * std))


def create_distribution_plots(system_name: str, group_name: str, errors_by_quality: typing.Mapping[str, np.ndarray],
                              output_folder: str):
    import matplotlib.pyplot as pyplot
    from scipy.stats import ks_2samp

    for get_error, error_name, units, is_integer, bounds in [
        # (lambda errs: errs[:, 0], 'forward accuracy', 'm', (None, None)),
        # (lambda errs: errs[:, 1], 'sideways accuracy', 'm', (None, None)),
        # (lambda errs: errs[:, 2], 'vertical accuracy', 'm', (None, None)),
        (lambda errs: errs[:, 3], 'translation error', 'm', False, (0, None)),
        # (lambda errs: 1 / (1 + errs[:, 3]), 'inverse translational accuracy', 'm', (0, 1)),
        (lambda errs: errs[:, 5], 'orientation error', 'radians', False, (0, np.pi)),
        # (lambda errs: errs[:, 6], 'forward precision', 'm', (None, None)),
        # (lambda errs: errs[:, 7], 'sideways precision', 'm', (None, None)),
        # (lambda errs: errs[:, 8], 'vertical precision', 'm', (None, None)),
        (lambda errs: errs[:, 9], 'translation consistency', 'm', False, (0, None)),
        # (lambda errs: 1 / (1 + errs[:, 9]), 'inverse translational precision', 'm', (0, 1)),
        (lambda errs: errs[:, 11], 'orientation consistency', 'rad', False, (0, np.pi)),
        # (lambda errs: errs[:, 13], 'feature count', None, True, (0, None)),
        (lambda errs: errs[:, 14], 'feature matches', None, True, (0, None))
    ]:
        title = "{0} on {1} {2} distribution".format(system_name, group_name, error_name)

        rw_error = get_error(errors_by_quality['Real World'])
        rw_error = rw_error[~np.isnan(rw_error)]

        # First, from all the data, find the bounds of the graph
        all_data = np.concatenate(tuple(get_error(errors) for errors in errors_by_quality.values()))
        all_data = all_data[~np.isnan(all_data)]
        if len(all_data) <= 0:  # No data available for this graph, skip it
            continue
        zoom_min, zoom_max = compute_zoom(all_data)
        del all_data

        if bounds[0] is not None:
            zoom_min = max(zoom_min, bounds[0])
        if bounds[1] is not None:
            zoom_max = min(zoom_max, bounds[1])
        bins = 500
        if is_integer and int(zoom_max) - int(zoom_min) < bins - 1:
            bins = list(range(int(zoom_min), int(zoom_max) + 1, 1))

        show = False
        figure, ax = pyplot.subplots(1, 1, figsize=(8, 4), dpi=80)
        for quality_name in QUALITY_NAME_ORDER:
            if quality_name == 'Real World':
                error = rw_error
            else:
                error = get_error(errors_by_quality[quality_name])
                error = error[~np.isnan(error)]

            middle_error = error[(error > zoom_min) * (error < zoom_max)]
            if len(middle_error) > 0 and np.max(middle_error) > np.min(middle_error):
                show = True
                if quality_name == 'Real World':
                    label = "{0} ({1} samples, {2} outliers)".format(quality_name.lower(), len(middle_error),
                                                                     len(error) - len(middle_error))
                else:
                    # Make sure we calculate the ks statistic between the full errors
                    ks_stat, ks_pval = ks_2samp(error, rw_error)
                    label = "{0} ({1} samples, {2} outliers, ks score: {3:.3f})".format(
                        quality_name.lower(), len(middle_error), len(error) - len(middle_error), ks_stat)
                # Integer quantities can't be split into more bins than the range
                ax.hist(
                    middle_error,
                    label=label,
                    density=True,
                    bins=bins,
                    alpha=QUALITY_NAME_ALPHA[quality_name.lower()],
                    color=QUALITY_NAME_COLOURS[quality_name.lower()]
                )
        if show:
            ax.set_xlim(left=zoom_min)
            ax.set_xlim(right=zoom_max)
            if units:
                ax.set_xlabel('{0} ({1})'.format(error_name, units))
            else:
                ax.set_xlabel(error_name)
            ax.set_ylabel('frequency')
            ax.legend()

            #figure.suptitle(title)
            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)
            #figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.png'))
            figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.svg'))
            pyplot.close(figure)


def create_tracking_plot(system_name: str, group_name: str, errors_by_quality: typing.Mapping[str, np.ndarray],
                         output_folder: str):
    import matplotlib.pyplot as pyplot

    scores = []
    for quality_name, errors in errors_by_quality.items():
        error = errors[:, 12]
        error = error[~np.isnan(error)]
        if len(error) > 0:
            score = np.sum(error) / len(error)
            scores.append((score, quality_name))

    if len(scores) > 0:
        scores = sorted(scores)

        for score, quality_name in scores:
            logging.getLogger(__name__).info("RESULT - Tracking probability for {0}: {1}".format(quality_name, score))

        title = "{0} on {1} tracking probability".format(system_name, group_name)

        figure, ax = pyplot.subplots(1, 1, figsize=(5, 5), dpi=80)
        ax.bar(list(range(len(scores))), [score[0] for score in scores])
        ax.set_xticks(list(range(len(scores))))
        ax.set_xticklabels([score[1] for score in scores])
        ax.set_ylabel('probability')

        figure.suptitle(title)
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.png'))
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.svg'))
        pyplot.close(figure)


def create_errors_plots(indexes_and_names: typing.List[typing.Tuple[int, str, str]],
                        group_predictions: typing.Mapping[str, typing.List[typing.List[typing.Tuple[float, float]]]],
                        output_folder: str):
    import pandas as pd

    for idx, error_name, units in indexes_and_names:
        # Build a pandas dataframe for this particular error
        df_data = {'source': [], 'errors': [], 'inv_errors': []}
        for group_name, group_predict_pairs in group_predictions.items():
            if len(group_predict_pairs[idx]) <= 0:
                continue
            error_data = np.array(group_predict_pairs[idx])
            # Calculate absolute error
            df_data['source'] += [group_name for _ in range(error_data.shape[0])]
            error = np.abs(error_data[:, 0] - error_data[:, 1])
            df_data['errors'] += error.tolist()
            df_data['inv_errors'] += (1 / (1 + error)).tolist()
        if len(df_data['errors']) <= 0:
            continue
        dataframe = pd.DataFrame(data=df_data)

        create_boxplot(
            title=error_name,
            dataframe=dataframe,
            column='errors',
            output_folder=output_folder,
            units=units,
            also_zoom=True
        )
        create_boxplot(
            title='Inverse ' + error_name.lower(),
            dataframe=dataframe,
            column='inv_errors',
            output_folder=output_folder,
            units=units,
            also_zoom=False
        )
        create_histogram(
            title=error_name + ' histogram',
            dataframe=dataframe,
            column='errors',
            output_folder=output_folder,
            units=units,
            also_zoom=True
        )
        create_histogram(
            title='Inverse ' + error_name.lower() + ' histogram',
            dataframe=dataframe,
            column='inv_errors',
            output_folder=output_folder,
            units=units,
            also_zoom=False
        )


def create_boxplot(title: str, dataframe, column: str, output_folder: str, units: str, also_zoom: bool = False):
    import matplotlib.pyplot as pyplot

    # Boxplot the data
    figure, ax = pyplot.subplots(1, 1, figsize=(12, 10), dpi=80)
    dataframe.boxplot(column=column, by='source', ax=ax)
    ax.set_title('')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('')
    ax.set_ylabel('Absolute Error ({0})'.format(units))

    figure.suptitle(title)
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=0.95, right=0.99)
    figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.png'))
    figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.svg'))

    # Re-save the figure zoomed in
    if also_zoom:
        ax.set_ylim(bottom=0, top=3 * np.std(dataframe[column].values))
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '_zoomed.png'))
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '_zoomed.svg'))
    pyplot.close(figure)


def create_histogram(title: str, dataframe, column: str, output_folder: str, units: str, also_zoom: bool = False):
    import matplotlib.pyplot as pyplot

    # Assuming we got some amount of data, boxplot it
    figure, ax = pyplot.subplots(1, 1, figsize=(12, 10), dpi=80)
    show = False
    for group_name, group_df in dataframe.groupby(by='source'):
        data = group_df[column].values
        if len(data) > 0 and np.max(data) > np.min(data):
            show = True
            ax.hist(data, label=group_name, density=True, bins=1000, alpha=0.5)
    if show:
        ax.set_xlim(left=0)
        ax.set_xlabel('Absolute Error ({0})'.format(units))
        ax.set_ylabel('frequency')
        ax.legend()

        figure.suptitle(title)
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.95, right=0.99)
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.png'))
        figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '.svg'))
        pyplot.close(figure)

    # Re-save the figure zoomed in
    if also_zoom:
        zoom_max = 3 * np.std(dataframe[column].values)
        show = False
        figure, ax = pyplot.subplots(1, 1, figsize=(12, 10), dpi=80)
        for group_name, group_df in dataframe.groupby(by='source'):
            data = group_df[column].values
            data = data[np.where((data != np.nan) * (data > 0) * (data < zoom_max))]
            if len(data) > 0 and np.max(data) > np.min(data):
                show = True
                ax.hist(data, label=group_name, density=True, bins=1000, alpha=0.5)
        if show:
            ax.set_xlim(left=0, right=zoom_max)
            ax.set_xlabel('Absolute Error ({0})'.format(units))
            ax.set_ylabel('frequency')
            ax.legend()
    
            figure.suptitle(title + ' central 3 standard deviations')
            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)
            figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '_zoomed.png'))
            figure.savefig(os.path.join(output_folder, title.replace(' ', '_') + '_zoomed.svg'))
            pyplot.close(figure)
