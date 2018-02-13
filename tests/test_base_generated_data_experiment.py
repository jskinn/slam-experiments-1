import unittest
import unittest.mock as mock
import bson
import arvet.util.dict_utils as du
import arvet.util.database_helpers as dh
import arvet.config.path_manager
import arvet.metadata.image_metadata as imeta
import arvet.database.tests.test_entity as entity_test
import arvet.batch_analysis.tests.mock_task_manager as mtm
import base_generated_data_experiment as bgde


class TestBaseGeneratedDataExperiment(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return bgde.GeneratedDataExperiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'systems': {
                'ORBSLAM monocular': bson.ObjectId(),
                'ORBSLAM stereo': bson.ObjectId(),
                'LibVisO2': bson.ObjectId()
            },
            'simulators': {'Block World': bson.ObjectId()},
            'trajectory_groups': {
                'KITTI trajectory 1': bgde.TrajectoryGroup(
                    name='KITTI trajectory 1',
                    reference_id=bson.ObjectId(),
                    mappings=[('Block World', {'location': [12, -63.2, 291.1], 'rotation': [-22, -214, 121]})],
                    baseline_configuration={'test': bson.ObjectId()},
                    controller_id=bson.ObjectId(),
                    generated_datasets={'Block World': bson.ObjectId()}
                )
            },
            'benchmarks': {
                'benchmark_rpe': bson.ObjectId(),
                'benchmark_ate': bson.ObjectId(),
                'benchmark_trajectory_drift': bson.ObjectId(),
                'benchmark_tracking': bson.ObjectId(),
            },
            'trial_map': {bson.ObjectId(): {bson.ObjectId(): [bson.ObjectId()]}},
            'result_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId()}},
            'enabled': True
        })
        return bgde.GeneratedDataExperiment(*args, **kwargs)

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
        if not isinstance(experiment1, bgde.GeneratedDataExperiment) or \
                not isinstance(experiment2, bgde.GeneratedDataExperiment):
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

    @mock.patch('base_generated_data_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
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

    @mock.patch('base_generated_data_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_system_does_not_add_existing_systems(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_system('LibVisO2', mock.Mock(), self.create_mock_db_client())
        self.assertFalse(mock_add_unique.called)

    @mock.patch('base_generated_data_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
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

    @mock.patch('base_generated_data_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
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

    @mock.patch('base_generated_data_experiment.arvet.util.database_helpers.add_unique', autospec=dh.add_unique)
    def test_import_benchmark_does_not_add_existing_systems(self, mock_add_unique):
        mock_add_unique.return_value = bson.ObjectId()
        subject = self.make_instance()
        self.assertFalse(mock_add_unique.called)
        subject.import_benchmark('benchmark_rpe', mock.Mock(), self.create_mock_db_client())
        self.assertFalse(mock_add_unique.called)

    def test_import_dataset_schedules_task(self):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        zombie_task_manager = mtm.create()
        mock_db_client = self.create_mock_db_client()
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
            path_manager=mock_path_manager,
            db_client=mock_db_client)
        self.assertTrue(zombie_task_manager.mock.get_import_dataset_task.called)
        self.assertIn(mock.call(mock_task), zombie_task_manager.mock.do_task.call_args_list)

    def test_import_dataset_creates_trajectory_group(self):
        mock_task = mock.Mock()
        mock_task.is_finished = True
        mock_task.result = bson.ObjectId()

        zombie_task_manager = mtm.create()
        mock_db_client = self.create_mock_db_client()
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
            path_manager=mock_path_manager,
            db_client=mock_db_client)
        self.assertIn('TestDataset', subject.trajectory_groups)
        trajectory_group = subject.trajectory_groups['TestDataset']
        self.assertEqual('TestDataset', trajectory_group.name)
        self.assertEqual(mock_task.result, trajectory_group.reference_dataset)
        self.assertEqual([('First', {'location': [1, 2, 3], 'rotation': [4, 5, 6]})], trajectory_group.mappings)
