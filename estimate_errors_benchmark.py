# Copyright (c) 2018, John Skinner
import typing
import bson
import pickle
import numpy as np
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.util.associate
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
import arvet_slam.trials.slam.tracking_state


class EstimateErrorsBenchmark(arvet.core.benchmark.Benchmark):

    def __init__(self, id_=None):
        """
        Measure and collect the errors in estimated trajectories per estimate.
        For n repeats of a trajectory of length m, this gives us n * m data points.
        We measure:
        - cartesian error
        - polar error
        - cartesian noise
        - polar noise
        - Whether or not we tracked successfully
        - The number of features
        - The number of matched features
        """
        super().__init__(id_=id_)

    def is_trial_appropriate(self, trial_result):
        return (hasattr(trial_result, 'identifier') and
                hasattr(trial_result, 'system_id') and
                hasattr(trial_result, 'get_ground_truth_camera_poses') and
                hasattr(trial_result, 'get_computed_camera_poses') and
                hasattr(trial_result, 'get_tracking_states') and
                hasattr(trial_result, 'num_features') and
                hasattr(trial_result, 'num_matches'))

    def benchmark_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
            -> arvet.core.benchmark.BenchmarkResult:
        """
        Collect the errors
        :param trial_results: The results of several trials to aggregate
        :return:
        :rtype BenchmarkResult:
        """
        trial_results = list(trial_results)
        invalid_reason = arvet.core.benchmark.check_trial_collection(trial_results)
        if invalid_reason is not None:
            return arvet.core.benchmark.FailedBenchmark(
                benchmark_id=self.identifier,
                trial_result_ids=[trial_result.identifier for trial_result in trial_results],
                reason=invalid_reason
            )

        # First, we need to find the average computed trajectory, so we can estimate noise
        mean_computed_motions = th.compute_average_trajectory([
            trial_result.get_computed_camera_motions() for trial_result in trial_results
        ])

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = []
        for trial_result in trial_results:
            ground_truth_motions = trial_result.get_ground_truth_motions()
            computed_motions = trial_result.get_computed_camera_motions()
            tracking_statistics = trial_result.get_tracking_states()
            num_features = trial_result.num_features
            num_matches = trial_result.num_matches

            unified_times = merge_timestamps((computed_motions.keys(), tracking_statistics.keys(),
                                              num_features.keys(), num_matches.keys()))

            to_average = {k: v for k, v in arvet.util.associate.associate(unified_times, mean_computed_motions,
                                                                          offset=0, max_difference=0.1)}
            to_computed_motions = {k: v for k, v in arvet.util.associate.associate(unified_times, computed_motions,
                                                                                   offset=0, max_difference=0.1)}
            to_tracking_statistics = {k: v for k, v in
                                      arvet.util.associate.associate(unified_times, tracking_statistics,
                                                                     offset=0, max_difference=0.1)}
            to_num_features = {k: v for k, v in arvet.util.associate.associate(unified_times, num_features,
                                                                               offset=0, max_difference=0.1)}
            to_num_matches = {k: v for k, v in arvet.util.associate.associate(unified_times, num_matches,
                                                                              offset=0, max_difference=0.1)}

            matches = arvet.util.associate.associate(ground_truth_motions, unified_times,
                                                     offset=0, max_difference=0.1)
            for match in matches:
                gt_motion = ground_truth_motions[match[0]]

                # Get estimate errors
                motion_errors = tuple(np.nan for _ in range(12))
                if match[1] in to_computed_motions:
                    motion_errors = get_error_from_motion(
                        motion=computed_motions[to_computed_motions[match[1]]],
                        gt_motion=gt_motion,
                        avg_motion=mean_computed_motions[to_average[match[1]]] if match[1] in to_average else None
                    )

                # Express the tracking state as a number
                tracking = np.nan
                if match[1] in to_tracking_statistics:
                    if tracking_statistics[to_tracking_statistics[match[1]]] == \
                            arvet_slam.trials.slam.tracking_state.TrackingState.OK:
                        tracking = 1.0
                    else:
                        tracking = 0.0

                # Tack on more metrics to the motion errors
                estimate_errors.append(motion_errors + (
                    tracking,
                    num_features[to_num_features[match[1]]] if match[1] in to_num_features else np.nan,
                    num_matches[to_num_matches[match[1]]] if match[1] in to_num_matches else np.nan,
                    gt_motion.location[0],
                    gt_motion.location[1],
                    gt_motion.location[2],
                    np.linalg.norm(gt_motion.location),
                    tf.quat_angle(gt_motion.rotation_quat(w_first=True))
                ))

        if len(estimate_errors) <= 0:
            return arvet.core.benchmark.FailedBenchmark(
                benchmark_id=self.identifier,
                trial_result_ids=[trial_result.identifier for trial_result in trial_results],
                reason="No measurable errors for these trajectories"
            )

        return EstimateErrorsResult(
            benchmark_id=self.identifier,
            trial_result_ids=[trial_result.identifier for trial_result in trial_results],
            estimate_errors=estimate_errors
        )


def merge_timestamps(timestamp_sets: typing.Iterable[typing.Iterable[float]]) -> typing.Mapping[float, bool]:
    """
    Join together several groups of timestamps, associating similar timestamps.
    This deals with slight variations in floating points
    :param timestamp_sets:
    :return:
    """
    unified_timestamps = {}
    for timestamp_set in timestamp_sets:
        times = set(timestamp_set)
        # Remove all the timestamps that can be associated to an existing timestamp in the unified map
        times -= {match[1] for match in arvet.util.associate.associate(
            unified_timestamps, {time: True for time in times}, offset=0, max_difference=0.1)}
        # Add all the times in this set that don't associate to anything
        for time in times:
            unified_timestamps[time] = True
    return unified_timestamps


def get_error_from_motion(motion: tf.Transform, gt_motion: tf.Transform, avg_motion: tf.Transform = None) \
        -> typing.Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Given a motion, ground truth motion, and average estimated motion, extract 12 different error statistics
    :param motion:
    :param gt_motion:
    :param avg_motion:
    :return:
    """
    # Error
    trans_error = motion.location - gt_motion.location
    trans_error_length = np.linalg.norm(trans_error)
    trans_error_direction = np.arccos(
        min(1.0, max(-1.0, np.dot(
            trans_error / trans_error_length,
            gt_motion.location / np.linalg.norm(gt_motion.location)))
            )
    ) if trans_error_length > 0 else 0  # No error direction when there is no error
    rot_error = tf.quat_diff(motion.rotation_quat(w_first=True), gt_motion.rotation_quat(w_first=True))

    # Noise
    if avg_motion is None:
        return (
            trans_error[0],
            trans_error[1],
            trans_error[2],
            trans_error_length,
            trans_error_direction,
            rot_error,

            # No average estimate,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )
    else:
        trans_noise = motion.location - avg_motion.location
        trans_noise_length = np.linalg.norm(trans_noise)
        trans_noise_direction = np.arccos(
            min(1.0, max(-1.0, np.dot(
                trans_noise / trans_noise_length,
                gt_motion.location / np.linalg.norm(gt_motion.location)))
                )
        ) if trans_noise_length > 0 else 0  # No noise direction for 0 noise
        rot_noise = tf.quat_diff(motion.rotation_quat(w_first=True), avg_motion.rotation_quat(w_first=True))

        return (
            trans_error[0],
            trans_error[1],
            trans_error[2],
            trans_error_length,
            trans_error_direction,
            rot_error,

            trans_noise[0],
            trans_noise[1],
            trans_noise[2],
            trans_noise_length,
            trans_noise_direction,
            rot_noise
        )


class EstimateErrorsResult(arvet.core.benchmark.BenchmarkResult):
    """
    Error observations per estimate of a pose
    """

    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId],
                 estimate_errors: typing.Iterable[typing.Iterable[float]],
                 id_: bson.ObjectId = None, **kwargs):
        """

        :param benchmark_id:
        :param trial_result_ids:
        :param timestamps:
        :param errors:
        :param rpe_settings:
        :param id_:
        :param kwargs:
        """
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_ids=trial_result_ids, id_=id_, **kwargs)
        self._errors_observations = np.asarray(estimate_errors)

    @property
    def observations(self) -> np.ndarray:
        return self._errors_observations

    def serialize(self):
        output = super().serialize()
        output['estimate_errors'] = bson.Binary(pickle.dumps(self._errors_observations.tolist(),
                                                             protocol=pickle.HIGHEST_PROTOCOL))
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'estimate_errors' in serialized_representation:
            kwargs['estimate_errors'] = pickle.loads(serialized_representation['estimate_errors'])
        return super().deserialize(serialized_representation, db_client, **kwargs)
