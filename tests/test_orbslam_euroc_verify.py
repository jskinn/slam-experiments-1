import unittest
import unittest.mock as mock
import bson
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity as entity_test
import orbslam_euroc_verify as euroc_verify


class TestOrbslamEuRoCVerify(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return euroc_verify.OrbslamEuRoCVerify

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'orbslam_mono': bson.ObjectId(),
            'orbslam_stereo': bson.ObjectId(),
            'datasets': {'KITTI Trajectory 00': bson.ObjectId()},
            'benchmark_rpe': bson.ObjectId(),
            'trial_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId()}},
            'result_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId()}},
            'enabled': True
        })
        return euroc_verify.OrbslamEuRoCVerify(*args, **kwargs)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()

        # Fix the count return values for systems, image sources and benchmarks, so that the don't get removed
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
        if not isinstance(experiment1, euroc_verify.OrbslamEuRoCVerify) or \
                not isinstance(experiment2, euroc_verify.OrbslamEuRoCVerify):
            self.fail('object was not an experiment')
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._orbslam_mono, experiment2._orbslam_mono)
        self.assertEqual(experiment1._orbslam_stereo, experiment2._orbslam_stereo)
        self.assertEqual(experiment1._datasets, experiment2._datasets)
        self.assertEqual(experiment1._benchmark_rpe, experiment2._benchmark_rpe)
