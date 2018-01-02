import unittest
import unittest.mock as mock
import bson
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity as entity_test
import kitti_generated_data_experiment as kitti_ex


class TestKITTIGeneratedDataExperiment(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return kitti_ex.KITTIGeneratedDataExperiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'libviso_system': bson.ObjectId(),
            'orbslam_systems': {'MONOCULAR': bson.ObjectId()},
            'simulators': {'Block World': bson.ObjectId()},
            'trajectory_groups': {
                'KITTI trajectory 1': kitti_ex.TrajectoryGroup(
                    name='KITTI trajectory 1',
                    reference_id=bson.ObjectId(),
                    baseline_configuration={'test': bson.ObjectId()},
                    simulators={'Block World': (bson.ObjectId(), {'conf': bson.ObjectId()})},
                    controller_id=bson.ObjectId(),
                    generated_datasets={'Block World': bson.ObjectId()}
                )
            },
            'benchmark_rpe': bson.ObjectId(),
            'benchmark_ate': bson.ObjectId(),
            'benchmark_trajectory_drift': bson.ObjectId(),
            'benchmark_tracking': bson.ObjectId(),
            'trial_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId()}},
            'result_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId()}},
            'enabled': True
        })
        return kitti_ex.KITTIGeneratedDataExperiment(*args, **kwargs)

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
        if not isinstance(experiment1, kitti_ex.KITTIGeneratedDataExperiment) or \
                not isinstance(experiment2, kitti_ex.KITTIGeneratedDataExperiment):
            self.fail('object was not an experiment')
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._libviso_system, experiment2._libviso_system)
        self.assertEqual(experiment1._orbslam_systems, experiment2._orbslam_systems)
        self.assertEqual(experiment1._simulators, experiment2._simulators)
        self.assertEqual(experiment1._benchmark_rpe, experiment2._benchmark_rpe)
        self.assertEqual(experiment1._benchmark_ate, experiment2._benchmark_ate)
        self.assertEqual(experiment1._benchmark_trajectory_drift, experiment2._benchmark_trajectory_drift)
        self.assertEqual(experiment1._benchmark_tracking, experiment2._benchmark_tracking)

        # Check the trajectory groups in detail
        self.assertEqual(len(experiment1._trajectory_groups), len(experiment2._trajectory_groups))
        for name, group1 in experiment1._trajectory_groups.items():
            self.assertIn(name, experiment2._trajectory_groups)
            group2 = experiment2._trajectory_groups[name]
            self.assertEqual(group1.name, group2.name)
            self.assertEqual(group1.baseline_configuration, group2.baseline_configuration)
            self.assertEqual(group1.simulators, group2.simulators)
            self.assertEqual(group1.follow_controller_id, group2.follow_controller_id)
            self.assertEqual(group1.generated_datasets, group2.generated_datasets)
