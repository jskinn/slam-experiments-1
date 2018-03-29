# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import bson
import copy
import transforms3d as tf3d
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
import arvet.database.tests.test_entity
import arvet.core.sequence_type
import arvet.core.trial_result
import arvet.core.benchmark
import arvet_slam.trials.slam.tracking_state as tracking_state
import frame_errors_benchmark as feb


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


class TestFrameErrorsBenchmark(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

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
        return feb.FrameErrorsBenchmark

    def make_instance(self, *args, **kwargs):
        return feb.FrameErrorsBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: BenchmarkRPE
        :param benchmark2: BenchmarkRPE
        :return:
        """
        if (not isinstance(benchmark1, feb.FrameErrorsBenchmark) or
                not isinstance(benchmark2, feb.FrameErrorsBenchmark)):
            self.fail('object was not a EstimateErrorsBenchmark')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)

    def test_benchmark_results_returns_a_benchmark_result(self):
        benchmark = feb.FrameErrorsBenchmark()
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
        benchmark = feb.FrameErrorsBenchmark()
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
        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

    def test_benchmark_results_one_observation_per_frame(self):
        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        if isinstance(result, arvet.core.benchmark.FailedBenchmark):
            print(result.reason)

        self.assertEqual(len(self.trial_results[0].ground_truth_trajectory) - 1, len(result.frame_errors))
        for error_measurement in result.frame_errors.values():
            self.assertEqual(18, len(error_measurement))

    def test_benchmark_results_estimates_no_error_for_identical_trajectory(self):
        # Copy the ground truth exactly
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = copy.deepcopy(trial_result.ground_truth_trajectory)

        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        if isinstance(result, arvet.core.benchmark.FailedBenchmark):
            print(result.reason)

        # Check all the errors are zero
        values = collect_values(result, 0)
        self.assertNPClose(np.zeros(values.shape), values)
        values = collect_values(result, 1)
        self.assertNPClose(np.zeros(values.shape), values)
        # We need more tolerance for the rotational error, because of the way the arccos
        # results in the smallest possible change producing a value around 2e-8
        values = collect_values(result, 2)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)
        values = collect_values(result, 3)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)

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

        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        # Check all the errors are zero
        values = collect_values(result, 0)
        self.assertNPClose(np.zeros(values.shape), values)
        values = collect_values(result, 1)
        self.assertNPClose(np.zeros(values.shape), values)
        values = collect_values(result, 2)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)
        values = collect_values(result, 3)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)

    def test_benchmark_results_estimates_no_noise_for_identical_trajectory(self):
        # Make all the trial results have exactly the same computed trajectories
        for trial_result in self.trial_results[1:]:
            trial_result.computed_trajectory = copy.deepcopy(self.trial_results[0].computed_trajectory)

        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)

        # Check all the errors are zero
        values = collect_values(result, 4)
        self.assertNPClose(np.zeros(values.shape), values)
        values = collect_values(result, 5)
        self.assertNPClose(np.zeros(values.shape), values)
        values = collect_values(result, 6)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)
        values = collect_values(result, 7)
        self.assertNPClose(np.zeros(values.shape), values, atol=1e-7)

    def test_benchmark_results_estimates_reasonable_trajectory_error_per_frame(self):
        benchmark = feb.FrameErrorsBenchmark()
        result = benchmark.benchmark_results(self.trial_results)
        # The noise added to each location is <10, converting to motions makes that <20, in each axis
        # But then, changes to the orientation tweak the relative location, and hence the motion
        self.assertLessEqual(np.max(collect_values(result, 0)), 100)
        self.assertLessEqual(np.max(collect_values(result, 2)), np.pi/32)

    def assertNPClose(self, arr1, arr2, atol=1e-12):
        self.assertTrue(np.all(np.isclose(arr1, arr2, atol=atol)),
                        "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def collect_values(result, idx):
    return np.array([err[idx] for err in result.frame_errors.values()])
