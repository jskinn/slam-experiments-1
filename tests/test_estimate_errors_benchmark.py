# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import pickle
import bson
import copy
import transforms3d as tf3d
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity
import arvet.core.sequence_type
import arvet.core.trial_result
import arvet.core.benchmark
import arvet_slam.trials.slam.tracking_state as tracking_state
import estimate_errors_benchmark as eeb


def create_random_trajectory(random_state, duration=600, length=10):
    trajectory = {}
    current_pose = tf.Transform(
        random_state.uniform(-1000, 1000, 3),
        random_state.uniform(-1, 1, 4)
    )
    velocity = random_state.uniform(-10, 10, 3)
    angular_velocity = tf3d.quaternions.axangle2quat(
        vector=random_state.uniform(-1, 1, 3),
        theta=random_state.uniform(-np.pi / 30, np.pi / 30)
    )
    for time in range(duration):
        current_pose = tf.Transform(
            location=current_pose.location + velocity,
            rotation=tf3d.quaternions.qmult(current_pose.rotation_quat(w_first=True), angular_velocity)
        )
        velocity += random_state.normal(0, 1, 3)
        angular_velocity = tf3d.quaternions.qmult(angular_velocity, tf3d.quaternions.axangle2quat(
            vector=random_state.uniform(-1, 1, 3),
            theta=random_state.normal(0, np.pi / 30)
        ))
        trajectory[time + random_state.normal(0, 0.1)] = current_pose

    return {random_state.uniform(0, duration):
            tf.Transform(location=random_state.uniform(-1000, 1000, 3), rotation=random_state.uniform(0, 1, 4))
            for _ in range(length)}


def create_noise(trajectory, random_state, time_offset=0, time_noise=0.01, loc_noise=10, rot_noise=np.pi/64):
    if not isinstance(loc_noise, np.ndarray):
        loc_noise = np.array([loc_noise, loc_noise, loc_noise])

    noise = {}
    for time, pose in trajectory.items():
        noise[time] = tf.Transform(location=random_state.uniform(-loc_noise, loc_noise),
                                   rotation=tf3d.quaternions.axangle2quat(random_state.uniform(-1, 1, 3),
                                                                          random_state.uniform(-rot_noise, rot_noise)),
                                   w_first=True)

    relative_frame = tf.Transform(location=random_state.uniform(-1000, 1000, 3),
                                  rotation=random_state.uniform(0, 1, 4))

    changed_trajectory = {}
    for time, pose in trajectory.items():
        relative_pose = relative_frame.find_relative(pose)
        noisy_time = time + time_offset + random_state.uniform(-time_noise, time_noise)
        noisy_pose = relative_pose.find_independent(noise[time])
        changed_trajectory[noisy_time] = noisy_pose

    return changed_trajectory, noise


class MockTrialResult(arvet.core.trial_result.TrialResult):

    def __init__(self, gt_trajectory, comp_trajectory, system_id):
        super().__init__(
            system_id=system_id,
            success=True,
            system_settings={},
            sequence_type=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
            id_=bson.ObjectId()
        )
        self._gt_traj = gt_trajectory
        self._comp_traj = comp_trajectory

        self.tracking_states = {time: tracking_state.TrackingState.OK for time in self._comp_traj.keys()}
        min_time = min(self.tracking_states.keys())
        max_time = max(self.tracking_states.keys())
        for idx in range(4):
            self.tracking_states[min_time - idx] = tracking_state.TrackingState.NOT_INITIALIZED
            self.tracking_states[max_time + idx] = tracking_state.TrackingState.LOST

        self.num_features = {time: np.random.uniform(20, 400) for time in self.tracking_states.keys()}
        self.num_matches = {time: max(features - np.random.uniform(20, 400), 20)
                            for time, features in self.num_features.items()}

    @property
    def ground_truth_trajectory(self):
        return self._gt_traj

    @ground_truth_trajectory.setter
    def ground_truth_trajectory(self, ground_truth_trajectory):
        self._gt_traj = ground_truth_trajectory

    @property
    def computed_trajectory(self):
        return self._comp_traj

    @computed_trajectory.setter
    def computed_trajectory(self, computed_trajectory):
        self._comp_traj = computed_trajectory

    def get_ground_truth_camera_poses(self):
        return self._gt_traj

    def get_computed_camera_poses(self):
        return self._comp_traj

    def get_ground_truth_motions(self):
        return th.trajectory_to_motion_sequence(self._gt_traj)

    def get_computed_camera_motions(self):
        return th.trajectory_to_motion_sequence(self._comp_traj)

    def get_tracking_states(self):
        return self.tracking_states


class TestEstimateErrorsBenchmark(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1311)   # Use a random stream to make the results consistent
        trajectory = create_random_trajectory(self.random)
        self.system_id = bson.ObjectId()
        self.trial_results = []
        self.noise = []
        for _ in range(10):
            noisy_trajectory, noise = create_noise(trajectory, self.random)
            self.trial_results.append(MockTrialResult(
                gt_trajectory=trajectory,
                comp_trajectory=noisy_trajectory,
                system_id=self.system_id
            ))
            self.noise.append(noise)

    def get_class(self):
        return eeb.EstimateErrorsBenchmark

    def make_instance(self, *args, **kwargs):
        return eeb.EstimateErrorsBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: BenchmarkRPE
        :param benchmark2: BenchmarkRPE
        :return:
        """
        if (not isinstance(benchmark1, eeb.EstimateErrorsBenchmark) or
                not isinstance(benchmark2, eeb.EstimateErrorsBenchmark)):
            self.fail('object was not a EstimateErrorsBenchmark')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)

    def test_benchmark_results_returns_a_benchmark_result(self):
        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(set(trial_result.identifier for trial_result in self.trial_results), set(result.trial_results))

    def test_benchmark_results_fails_for_trials_from_different_systems(self):
        trajectory = create_random_trajectory(self.random)
        mixed_trial_results = self.trial_results + [MockTrialResult(
            gt_trajectory=trajectory,
            comp_trajectory=trajectory,
            system_id=bson.ObjectId()
        )]

        # Perform the benchmark
        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(mixed_trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

    def test_benchmark_results_fails_for_no_observations(self):
        # Adjust the computed timestamps so none of them match
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = {
                time + 10000: pose
                for time, pose in trial_result.computed_trajectory.items()
            }
            trial_result.tracking_states = {
                time + 10000: state for time, state in trial_result.tracking_states.items()
            }
            trial_result.num_features = {
                time + 10000: features for time, features in trial_result.num_features.items()
            }
            trial_result.num_matches = {
                time + 10000: features for time, features in trial_result.num_matches.items()
            }

        # Perform the benchmark
        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

    def test_benchmark_results_one_observation_per_motion_per_trial(self):
        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        if isinstance(result, arvet.core.benchmark.FailedBenchmark):
            print(result.reason)

        self.assertEqual((len(self.trial_results) * (len(self.trial_results[0].ground_truth_trajectory) - 1), 20),
                         result.observations.shape)

    def test_benchmark_results_estimates_no_error_for_identical_trajectory(self):
        # Copy the ground truth exactly
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = copy.deepcopy(trial_result.ground_truth_trajectory)

        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        if isinstance(result, arvet.core.benchmark.FailedBenchmark):
            print(result.reason)

        # Check all the errors are zero
        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]), result.observations[:, 3])))
        # We need more tolerance for the rotational error, because of the way the arccos
        # results in the smallest possible change producing a value around 2e-8
        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]),
                                          result.observations[:, 5], atol=1e-7)))

    def test_benchmark_results_estimates_no_error_for_noiseless_trajectory(self):
        # Create a new computed trajectory with no noise, but a fixed offset from the real trajectory
        # That is, the relative motions are the same, but the start point is different
        for trial_result in self.trial_results:
            comp_traj, _ = create_noise(
                trajectory=trial_result.ground_truth_trajectory,
                random_state=self.random,
                time_offset=0,
                time_noise=0,
                loc_noise=0,
                rot_noise=0
            )
            trial_result.computed_trajectory = comp_traj

        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]), result.observations[:, 3])))
        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]),
                                          result.observations[:, 5], atol=1e-7)))

    def test_benchmark_results_estimates_no_noise_for_identical_trajectory(self):
        # Make all the trial results have exactly the same computed trajectories
        for trial_result in self.trial_results[1:]:
            trial_result.computed_trajectory = copy.deepcopy(self.trial_results[0].computed_trajectory)

        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]), result.observations[:, 9])))
        self.assertTrue(np.all(np.isclose(np.zeros(result.observations.shape[0]),
                                          result.observations[:, 11], atol=1e-7)))

    def test_benchmark_results_estimates_reasonable_trajectory_error_per_frame(self):
        benchmark = eeb.EstimateErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)
        # The noise added to each location is <10, converting to motions makes that <20, in each axis
        # But then, changes to the orientation tweak the relative location, and hence the motion
        self.assertLessEqual(np.max(result.observations[:, 3]), 150)
        self.assertLessEqual(np.max(result.observations[:, 5]), np.pi/32)


class TestEstimateErrorsResult(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1311)   # Use a random stream to make the results consistent
        trajectory = create_random_trajectory(self.random)
        self.system_id = bson.ObjectId()
        self.trial_results = []
        self.noise = []
        for _ in range(10):
            noisy_trajectory, noise = create_noise(trajectory, self.random)
            self.trial_results.append(MockTrialResult(
                gt_trajectory=trajectory,
                comp_trajectory=noisy_trajectory,
                system_id=self.system_id
            ))
            self.noise.append(noise)

    def get_class(self):
        return eeb.EstimateErrorsResult

    def make_instance(self, *args, **kwargs):
        du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_ids': [bson.ObjectId() for _ in range(4)],
            'estimate_errors': [
                [np.random.uniform(-100, 100) for _ in range(20)]
                for _ in range(1000)
            ],
        })
        return eeb.EstimateErrorsResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark_result1: EstimateErrorsResult
        :param benchmark_result2: EstimateErrorsResult
        :return:
        """
        if (not isinstance(benchmark_result1, eeb.EstimateErrorsResult) or
                not isinstance(benchmark_result2, eeb.EstimateErrorsResult)):
            self.fail('object was not a EstimateErrorsResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_results, benchmark_result2.trial_results)
        self.assertNPEqual(benchmark_result1.observations, benchmark_result2.observations)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1:
        :param s_model2:
        :return:
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'estimate_errors' and key is not 'trial_results':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for sets
        self.assertEqual(set(s_model1['trial_results']), set(s_model2['trial_results']))

        # Special case for BSON
        errors1 = pickle.loads(s_model1['estimate_errors'])
        errors2 = pickle.loads(s_model2['estimate_errors'])
        self.assertEqual(errors1, errors2)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))
