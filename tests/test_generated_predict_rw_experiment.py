import unittest
import unittest.mock as mock
import time
import os
import numpy as np
import bson
import arvet.util.dict_utils as du
import arvet.util.database_helpers as dh
import arvet.config.path_manager
import arvet.metadata.image_metadata as imeta
import arvet.database.tests.test_entity as entity_test
import arvet.database.tests.mock_database_client as mock_db_client_fac
import arvet.batch_analysis.tests.mock_task_manager as mtm
import trajectory_group as tg
import generated_predict_rw_experiment as gprwe

import sklearn


class TestBaseGeneratedDataExperiment(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return gprwe.GeneratedPredictRealWorldExperiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'systems': {
                'ORBSLAM monocular': bson.ObjectId(),
                'ORBSLAM stereo': bson.ObjectId(),
                'LibVisO2': bson.ObjectId()
            },
            'simulators': {'Block World': bson.ObjectId()},
            'trajectory_groups': {
                'KITTI trajectory 1': tg.TrajectoryGroup(
                    name='KITTI trajectory 1',
                    reference_id=bson.ObjectId(),
                    mappings=[('Block World', {'location': [12, -63.2, 291.1], 'rotation': [-22, -214, 121]})],
                    baseline_configuration={'test': bson.ObjectId()},
                    controller_id=bson.ObjectId(),
                    generated_datasets={
                        'Block World': {
                            'max_quality': bson.ObjectId(),
                            'min_quality': bson.ObjectId()
                        }
                    }
                ),
                'KITTI trajectory 2': tg.TrajectoryGroup(
                    name='KITTI trajectory 2',
                    reference_id=bson.ObjectId(),
                    mappings=[('Block World', {'location': [12, -63.2, 291.1], 'rotation': [-22, -214, 121]})],
                    baseline_configuration={'test': bson.ObjectId()},
                    controller_id=bson.ObjectId(),
                    generated_datasets={
                        'Block World': {
                            'max_quality': bson.ObjectId(),
                            'min_quality': bson.ObjectId()
                        }
                    }
                )
            },
            'benchmarks': {
                'Estimate Errors': bson.ObjectId()
            },
            'trial_map': {
                bson.ObjectId(): {
                    bson.ObjectId(): {
                        'trials': [bson.ObjectId(), bson.ObjectId(), bson.ObjectId()],
                        'results': {
                            bson.ObjectId(): bson.ObjectId()
                        }
                    }
                }
            },
            'enabled': True
        })
        return gprwe.GeneratedPredictRealWorldExperiment(*args, **kwargs)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()

        # Fix the count return values for systems, image sources and benchmarks, so that we can
        mock_cursor = mock.Mock()
        mock_cursor.count.return_value = 1

        self.db_client.system_collection.find.return_value = mock_cursor
        self.db_client.image_source_collection.find.return_value = mock_cursor
        self.db_client.benchmarks_collection.find.return_value = mock_cursor

        return self.db_client

    def assert_models_equal(self, experiment1, experiment2):
        """
        Helper to assert that two image entities are equal
        :param experiment1: ImageEntity
        :param experiment2: ImageEntity
        :return:
        """
        if not isinstance(experiment1, gprwe.GeneratedPredictRealWorldExperiment) or \
                not isinstance(experiment2, gprwe.GeneratedPredictRealWorldExperiment):
            self.fail('object was not an experiment')
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1.systems, experiment2.systems)
        self.assertEqual(experiment1.simulators, experiment2.simulators)
        self.assertEqual(experiment1.benchmarks, experiment2.benchmarks)

        # Check the trajectory groups in detail
        self.assertEqual(len(experiment1.trajectory_groups), len(experiment2.trajectory_groups))
        for name, group1 in experiment1.trajectory_groups.items():
            self.assertIn(name, experiment2.trajectory_groups)
            group2 = experiment2.trajectory_groups[name]
            self.assertEqual(group1.name, group2.name)
            self.assertEqual(group1.reference_dataset, group2.reference_dataset)
            self.assertEqual(group1.mappings, group2.mappings)
            self.assertEqual(group1.baseline_configuration, group2.baseline_configuration)
            self.assertEqual(group1.follow_controller_id, group2.follow_controller_id)
            self.assertEqual(group1.generated_datasets, group2.generated_datasets)

    @mock.patch('generated_predict_rw_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_system_creates_object(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        mock_db_client = self.create_mock_db_client()
        mock_system = mock.Mock()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_system('NewSystem', mock_system, mock_db_client)
        self.assertIn(mock.call(mock_db_client.system_collection, mock_system), mock_add_unique.call_args_list)
        self.assertIn('NewSystem', subject.systems)
        self.assertEqual(mock_add_unique.return_value, subject.systems['NewSystem'])

    @mock.patch('generated_predict_rw_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_system_does_not_add_existing_systems(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_system('LibVisO2', mock.Mock(), self.create_mock_db_client())
        self.assertFalse(mock_add_unique.called)

    @mock.patch('generated_predict_rw_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_simulator_creates_object(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_simulator(
            world_name='DemoWorld',
            executable_path='/tmp/notafile',
            environment_type=imeta.EnvironmentType.OUTDOOR_LANDSCAPE,
            light_level=imeta.LightingLevel.PITCH_BLACK,
            time_of_day=imeta.TimeOfDay.DAWN,
            db_client=self.create_mock_db_client())
        self.assertTrue(mock_add_unique.called)
        self.assertIn('DemoWorld', subject.simulators)
        self.assertEqual(mock_add_unique.return_value, subject.simulators['DemoWorld'])

    @mock.patch('generated_predict_rw_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_benchmark_creates_object(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        mock_db_client = self.create_mock_db_client()
        mock_benchmark = mock.Mock()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_benchmark('NewBenchmark', mock_benchmark, mock_db_client)
        self.assertIn(mock.call(mock_db_client.benchmarks_collection, mock_benchmark), mock_add_unique.call_args_list)
        self.assertIn('NewBenchmark', subject.benchmarks)
        self.assertEqual(mock_add_unique.return_value, subject.benchmarks['NewBenchmark'])

    @mock.patch('generated_predict_rw_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_benchmark_does_not_add_existing_systems(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_benchmark('Estimate Errors', mock.Mock(), self.create_mock_db_client())
        self.assertFalse(mock_add_unique.called)

    def test_import_dataset_schedules_task(self):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        zombie_task_manager = mtm.create()
        mock_path_manager = mock.create_autospec(arvet.config.path_manager.PathManager)
        mock_path_manager.check_path.return_value = True
        zombie_task_manager.mock.get_import_dataset_task.side_effect = None
        zombie_task_manager.mock.get_import_dataset_task.return_value = mock_task

        subject = self.make_instance()
        self.assertFalse(zombie_task_manager.mock.get_import_dataset_task.called)
        subject.import_dataset(
            name='TestDataset',
            mappings=[('First', {'location': [1, 2, 3], 'rotation': [4, 5, 6]})],
            module_name='importmodule',
            path='/tmp/place',
            task_manager=zombie_task_manager.mock,
            path_manager=mock_path_manager)
        self.assertTrue(zombie_task_manager.mock.get_import_dataset_task.called)
        self.assertIn(mock.call(mock_task), zombie_task_manager.mock.do_task.call_args_list)

    def test_import_dataset_creates_trajectory_group(self):
        mock_task = mock.Mock()
        mock_task.is_finished = True
        mock_task.result = bson.ObjectId()

        zombie_task_manager = mtm.create()
        mock_path_manager = mock.create_autospec(arvet.config.path_manager.PathManager)
        mock_path_manager.check_path.return_value = True
        zombie_task_manager.mock.get_import_dataset_task.side_effect = None
        zombie_task_manager.mock.get_import_dataset_task.return_value = mock_task

        subject = self.make_instance()
        self.assertFalse(zombie_task_manager.mock.get_import_dataset_task.called)
        subject.import_dataset(
            name='TestDataset',
            mappings=[('First', {'location': [1, 2, 3], 'rotation': [4, 5, 6]})],
            module_name='importmodule',
            path='/tmp/place',
            task_manager=zombie_task_manager.mock,
            path_manager=mock_path_manager)
        self.assertIn('TestDataset', subject.trajectory_groups)
        trajectory_group = subject.trajectory_groups['TestDataset']
        self.assertEqual('TestDataset', trajectory_group.name)
        self.assertEqual(mock_task.result, trajectory_group.reference_dataset)
        self.assertEqual([('First', {'location': [1, 2, 3], 'rotation': [4, 5, 6]})], trajectory_group.mappings)

    def test_split_datasets_validation_and_training_does_not_return_same_result_more_than_once(self):
        mock_db_client = self.create_mock_db_client()
        subject = self.make_instance(trial_map={})

        # Create results for each system, dataset, and benchmark
        for system_id in subject.systems.values():
            for trajectory_group in subject.trajectory_groups.values():
                for dataset_id in trajectory_group.get_all_dataset_ids():
                    subject.store_trial_results(system_id, dataset_id, [bson.ObjectId() for _ in range(10)],
                                                mock_db_client)
                    for benchmark_id in subject.benchmarks.values():
                        subject.store_benchmark_result(system_id, dataset_id, benchmark_id, bson.ObjectId())

        for system_id in subject.systems.values():
            output = subject.split_datasets_validation_and_training(system_id, {'KITTI trajectory 1'})
            validation_real_world_datasets, training_real_world_datasets, virtual_datasets_by_quality = output

            returned_datasets = set(validation_real_world_datasets)
            self.assertEqual(0, len(returned_datasets & training_real_world_datasets))
            returned_datasets |= training_real_world_datasets
            for validation_virtual_datasets, training_virtual_datasets in virtual_datasets_by_quality.values():
                self.assertEqual(0, len(returned_datasets & validation_virtual_datasets))
                returned_datasets |= validation_virtual_datasets
                self.assertEqual(0, len(returned_datasets & training_virtual_datasets))
                returned_datasets |= training_virtual_datasets


class TestPerformAnalysis(unittest.TestCase):

    def test_predict_regression(self):
        predictors = create_test_predictors()
        error = create_trans_error(predictors)
        val_size = predictors.shape[0] // 10

        train_x = predictors[val_size:, :]
        train_y = error[val_size:]
        val_x = predictors[:val_size, :]
        val_y = error[:val_size]

        start = time.time()
        result = gprwe.predict_regression((train_x, train_y), (val_x, val_y))
        end = time.time()
        print("Regression time: {0} for {1} data points".format(end - start, len(train_y)))
        self.assertLess(end - start, 10)
        self.assertEqual(val_size, len(result))
        for idx, (estimaged, gt) in enumerate(result):
            self.assertEqual(gt, val_y[idx])

    def test_predict_classification(self):
        predictors = create_test_predictors()
        error = create_trans_error(predictors)
        error = np.asarray(error > np.mean(error), dtype=np.int)
        val_size = predictors.shape[0] // 10

        train_x = predictors[val_size:, :]
        train_y = error[val_size:]
        val_x = predictors[:val_size, :]
        val_y = error[:val_size]

        start = time.time()
        result = gprwe.predict_classification((train_x, train_y), (val_x, val_y))
        end = time.time()
        print("Classification time: {0} for {1} data points".format(end - start, len(train_y)))
        self.assertLess(end - start, 15)
        self.assertEqual(val_size, len(result))
        for idx, (estimaged, gt) in enumerate(result):
            self.assertEqual(gt, val_y[idx])

    def test_predict_errors(self):
        predictors = create_test_predictors()
        errors = create_all_errors(predictors)
        val_size = predictors.shape[0] // 10

        train_x = predictors[val_size:, :]
        train_y = errors[val_size:]
        val_x = predictors[:val_size, :]
        val_y = errors[:val_size]

        start = time.time()
        result = gprwe.predict_errors((train_x, train_y), (val_x, val_y))
        end = time.time()
        print("All regression time: {0} for {1} data points".format(end - start, len(train_y)))
        self.assertLess(end - start, 100)
        self.assertEqual(errors.shape[1], len(result))
        for err_idx, error_results in enumerate(result):
            self.assertEqual(val_size, len(error_results))
            for idx, (estimaged, gt) in enumerate(error_results):
                self.assertEqual(gt, val_y[idx, err_idx])

    def test_predict_real_and_virtual_errors(self):
        validation_results = [bson.ObjectId() for _ in range(10)]
        real_world_results = [bson.ObjectId() for _ in range(10)]
        virtual_results_by_quality = {
            'quality_{0}'.format(idx): ({bson.ObjectId() for _ in range(10)}, {bson.ObjectId() for _ in range(10)})
            for idx in range(4)
        }

        # Build a results cache so we don't have to do anything for the db client
        results_ids = set(validation_results) | set(real_world_results)
        for result_set_1, result_set_2 in virtual_results_by_quality.values():
            results_ids |= set(result_set_1) | set(result_set_2)
        predictors = create_test_predictors()
        errors = create_all_errors(predictors)
        obs_size = len(predictors) // len(results_ids)
        results_cache = {
            result_id: np.hstack((
                errors[obs_size * ridx:obs_size * (ridx + 1), :],
                predictors[obs_size * ridx:obs_size * (ridx + 1), :]
            ))
            for ridx, result_id in enumerate(results_ids)
        }

        # predict the errors
        start = time.time()
        real_world_scores, errors_by_group = gprwe.predict_real_and_virtual_errors(
            validation_results=validation_results,
            real_world_results=real_world_results,
            virtual_results_by_quality=virtual_results_by_quality,
            db_client=mock_db_client_fac.create().mock,
            results_cache=results_cache
        )
        end = time.time()

        print("predict real and virtual errors time: {0}".format(end - start))
        self.assertLess(end - start, 200)
        self.assertEqual(13, len(real_world_scores))
        for result_list in real_world_scores:
            self.assertEqual(obs_size * len(validation_results), len(result_list))
        for quality in virtual_results_by_quality.keys():
            for group in {' all data', ' no validation trajectory', ' only validation trajectory'}:
                self.assertIn(quality + group, errors_by_group)
                self.assertEqual(13, len(errors_by_group[quality + group]))
                for result_list in errors_by_group[quality + group]:
                    self.assertEqual(obs_size * len(validation_results), len(result_list),
                                     "wrong number of results for group {0}".format(quality + group))

    def test_collect_errors_and_input_loads_from_cache(self):
        result_ids = [bson.ObjectId() for _ in range(5)]
        mock_db_client = mock_db_client_fac.create().mock
        results_cache = {
            result_id: np.array([[2500 * k + 25 * j + i for i in range(25)] for j in range(100)])
            for k, result_id in enumerate(result_ids)
        }
        x, y = gprwe.collect_errors_and_input(result_ids, mock_db_client, results_cache)
        self.assertEqual(500, x.shape[0])
        self.assertEqual(12, x.shape[1])
        self.assertEqual(500, y.shape[0])
        self.assertEqual(13, y.shape[1])

    def test_collect_errors_and_input_loads_from_database(self):

        def mock_deserialize(s_result):
            mock_result = mock.Mock()
            mock_result.identifier = s_result['_id']
            mock_result.observations = np.random.normal(0, 1, size=(100, 25))
            return mock_result

        mock_db_client = mock_db_client_fac.create().mock
        mock_db_client.results_collection = mock.Mock()
        mock_db_client.results_collection.find.side_effect = \
            lambda query: [{'_id': oid} for oid in query['_id']['$in']]
        mock_db_client.deserialize_entity.side_effect = mock_deserialize

        cached_ids = {bson.ObjectId() for _ in range(5)}
        unloaded_ids = {bson.ObjectId() for _ in range(5)}
        results_cache = {
            result_id: np.array([[2500 * k + 25 * j + i for i in range(25)] for j in range(100)])
            for k, result_id in enumerate(cached_ids)
        }
        gprwe.collect_errors_and_input(cached_ids | unloaded_ids, mock_db_client, results_cache)
        self.assertTrue(mock_db_client.results_collection.find.called)
        self.assertEqual({'_id': {'$in': list(unloaded_ids)}}, mock_db_client.results_collection.find.call_args[0][0])
        for unloaded_id in unloaded_ids:
            self.assertIn(unloaded_id, results_cache)

    def test_collect_errors_and_input_loads_copies_from_cache(self):
        result_id = bson.ObjectId()
        mock_db_client = mock_db_client_fac.create().mock
        results_cache = {
            result_id: np.array([[25 * j + i for i in range(25)] for j in range(100)])
        }
        x, y = gprwe.collect_errors_and_input({result_id}, mock_db_client, results_cache)
        x[:, 0] = -10
        y[:, 0] = -20
        self.assertEqual(0, results_cache[result_id][0, 0])
        self.assertEqual(13, results_cache[result_id][0, 13])
        self.assertTrue(np.all(results_cache[result_id] >= 0))

    def test_create_error_plots(self):
        output_folder = 'temp-test-create-error-plots'
        system_name = 'test_system'
        group_names = [
            'Real world',
            'Max quality all data',
            'Max quality no validation trajectory',
            'Max quality only validation trajectory',
            'Min quality all data',
            'Min quality no validation trajectory',
            'Min quality only validation trajectory'
        ]

        group_predictions = {}
        data_len = 100
        for group_name in group_names:
            group_predictions[group_name] = []
            for _ in range(12):
                values = np.random.exponential(1, size=data_len)
                estimates = values + np.random.choice((-1, 1), size=data_len) * \
                            np.random.exponential(0.1, size=data_len)
                group_predictions[group_name].append(
                    [(values[obs_idx], estimates[obs_idx]) for obs_idx in range(data_len)]
                )

        os.makedirs(output_folder, exist_ok=True)
        gprwe.create_errors_plots(
            indexes_and_names=[
                (0, '{0} forward error'.format(system_name), 'm'),
                (1, '{0} sideways error'.format(system_name), 'm'),
                (2, '{0} vertical error'.format(system_name), 'm'),
                (3, '{0} translational error length'.format(system_name), 'm'),
                (5, '{0} rotational error'.format(system_name), 'radians'),
                (6, '{0} forward noise'.format(system_name), 'm'),
                (7, '{0} sideways noise'.format(system_name), 'm'),
                (8, '{0} vertical noise'.format(system_name), 'm'),
                (9, '{0} translational noise length'.format(system_name), 'm'),
                (11, '{0} rotational noise'.format(system_name), 'rad')
            ],
            group_predictions=group_predictions,
            output_folder=output_folder
        )

    def test_create_distibution_plots(self):
        output_folder = 'temp-test-create-distribution-plots'
        quality_names = ['Real World', 'Max quality', 'Min quality']
        predictors = create_test_predictors()
        errors = create_all_errors(predictors)
        obs_size = len(predictors) // len(quality_names)
        errors_by_quality = {
            quality_name: errors[obs_size * ridx:obs_size * (ridx + 1), :]
            for ridx, quality_name in enumerate(quality_names)
        }

        os.makedirs(output_folder, exist_ok=True)
        gprwe.create_distribution_plots(
            system_name='test system',
            group_name='all data',
            errors_by_quality=errors_by_quality,
            output_folder=output_folder,
            also_zoom=True
        )


def create_all_errors(predictors):
    data_len = predictors.shape[0]
    feature_errors = (10 / (np.min((predictors[:, 0:1], predictors[:, 1:2]), axis=0)))
    errors = np.hstack((
        predictors[:, 2:3] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 3:4] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 4:5] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        np.zeros(shape=(data_len, 1)),
        np.random.uniform(-np.pi, np.pi, size=(data_len, 2)),
        predictors[:, 2:3] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 3:4] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 4:5] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        np.zeros(shape=(data_len, 1)),
        np.random.uniform(-np.pi, np.pi, size=(data_len, 2)),
        np.random.choice((0, 1), size=(data_len, 1))
    ))
    errors[:, 3] = np.linalg.norm(errors[:, 0:3], axis=1)
    errors[:, 9] = np.linalg.norm(errors[:, 6:9], axis=1)
    return errors


def create_trans_error(predictors):
    data_len = predictors.shape[0]
    feature_errors = (10 / (np.min((predictors[:, 0:1], predictors[:, 1:2]), axis=0)))
    error = np.hstack((
        predictors[:, 2:3] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 3:4] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1)),
        predictors[:, 4:5] + feature_errors + np.random.normal(0, 0.01, size=(data_len, 1))
    ))
    return np.linalg.norm(error, axis=1)

def create_test_predictors(data_len: int = 10000):
    random = np.random.RandomState()
    predictors = np.hstack((
        np.asarray(random.randint(10, 600, size=(data_len, 2)), dtype=np.float32),
        random.choice((-1, 1), size=(data_len, 3)) * random.exponential(1, size=(data_len, 3)),
        np.zeros(shape=(data_len, 1)),
        random.uniform(-np.pi, np.pi, size=(data_len, 3)),
        random.exponential(1, size=(data_len, 3))
    ))
    predictors[:, 5] = np.linalg.norm(predictors[:, 2:5], axis=1)
    return predictors
