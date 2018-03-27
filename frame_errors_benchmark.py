# Copyright (c) 2018, John Skinner
import typing
import bson
import pickle
import numpy as np
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.util.associate
import arvet.util.transform as tf
import arvet_slam.trials.slam.tracking_state


class FrameErrorsBenchmark(arvet.core.benchmark.Benchmark):

    def __init__(self, id_=None):
        """
        Measure and collect the errors in estimated trajectories per frame.
        For n repeats of a trajectory of length m, this gives us m data points, containing some statistics over n.
        We measure:
        - mean & std location error
        - mean & std angle error
        - mean & std location noise
        - mean & std angle noise
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

        ground_truth_motions = None

        # Collect together all the estimates for each frame from all the trial results
        estimates = {}
        for trial_result in trial_results:
            if ground_truth_motions is None:
                ground_truth_motions = trial_result.get_ground_truth_motions()
            computed_motions = trial_result.get_computed_camera_motions()
            tracking_statistics = trial_result.get_tracking_states()
            num_features = trial_result.num_features
            num_matches = trial_result.num_matches

            computed_keys = set(computed_motions.keys()) | set(tracking_statistics.keys())
            computed_keys |= set(num_features.keys()) | set(num_matches.keys())
            matches = arvet.util.associate.associate(ground_truth_motions, {k: True for k in computed_keys},
                                                     offset=0, max_difference=0.1)
            for match in matches:
                if match[0] not in estimates:
                    estimates[match[0]] = {
                        'motion': [],
                        'tracking': [],
                        'num_features': [],
                        'num_matches': []
                    }

                if match[1] in computed_motions:
                    estimates[match[0]]['motion'].append(computed_motions[match[1]])

                # Express the tracking state as a number
                if match[1] in tracking_statistics:
                    if tracking_statistics[match[1]] == arvet_slam.trials.slam.tracking_state.TrackingState.OK:
                        estimates[match[0]]['tracking'].append(1.0)
                    else:
                        estimates[match[0]]['tracking'].append(0.0)

                if match[1] in num_features:
                    estimates[match[0]]['num_features'].append(num_features[match[1]])
                if match[1] in num_matches:
                    estimates[match[0]]['num_matches'].append(num_matches[match[1]])

        # Now that we have all the estimates, aggregate the errors
        frame_errors = {}
        for gt_time, estimates_obj in estimates.items():
            if len(estimates_obj['motion'] > 0):
                mean_estimated_motion = tf.compute_average_pose(estimates_obj['motion'])
                location_errors = [np.linalg.norm(motion.location - ground_truth_motions[gt_time].location)
                                   for motion in estimates_obj['motion']]
                angle_errors = [
                    tf.quat_diff(motion.rotation_quat(w_first=True),
                                 ground_truth_motions[gt_time].rotation_quat(w_first=True))
                    for motion in estimates_obj['motion']
                ]
                location_noise = [np.linalg.norm(motion.location - mean_estimated_motion.location)
                                  for motion in estimates_obj['motion']]
                rotation_noise = [
                    tf.quat_diff(motion.rotation_quat(w_first=True),
                                 mean_estimated_motion.rotation_quat(w_first=True))
                    for motion in estimates_obj['motion']
                ]
                motion_errors = (
                    float(np.mean(location_errors)),
                    float(np.std(location_errors)),
                    float(np.mean(angle_errors)),
                    float(np.std(angle_errors)),
                    float(np.mean(location_noise)),
                    float(np.std(location_noise)),
                    float(np.mean(rotation_noise)),
                    float(np.std(rotation_noise))
                )
            else:
                motion_errors = tuple(np.nan for _ in range(8))

            if len(estimates_obj['tracking']) > 0:
                p_lost = 1.0 - (np.sum(estimates_obj['tracking']) / len(estimates_obj['tracking']))
            else:
                p_lost = np.nan

            if len(estimates_obj['num_features']) > 0:
                mean_features = np.mean(estimates_obj['num_features'])
                std_features = np.mean(estimates_obj['num_features'])
            else:
                mean_features = np.nan
                std_features = np.nan

            if len(estimates_obj['num_matches']) > 0:
                mean_matches = np.mean(estimates_obj['num_matches'])
                std_matches = np.mean(estimates_obj['num_matches'])
            else:
                mean_matches = np.nan
                std_matches = np.nan

            frame_errors[gt_time] = motion_errors + (
                p_lost,
                mean_features,
                std_features,
                mean_matches,
                std_matches,
                ground_truth_motions[gt_time].location[0],
                ground_truth_motions[gt_time].location[1],
                ground_truth_motions[gt_time].location[2],
                float(np.linalg.norm(ground_truth_motions[gt_time].location)),
                tf.quat_angle(ground_truth_motions[gt_time].rotation_quat(True))
            )
        return FrameErrorsResult(
            benchmark_id=self.identifier,
            trial_result_ids=[trial_result.identifier for trial_result in trial_results],
            frame_errors=frame_errors
        )


class FrameErrorsResult(arvet.core.benchmark.BenchmarkResult):
    """
    Error observations per frame of the underlying image sequence, identified by timestamp.
    These errors can be plotted vs time.
    """
    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId],
                 frame_errors: typing.Mapping[float, typing.Iterable[float]],
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
        self._frame_errors = frame_errors

    @property
    def frame_errors(self) -> typing.Mapping[float, typing.Iterable[float]]:
        return self._frame_errors

    def serialize(self):
        output = super().serialize()
        output['timestamps'] = self.timestamps
        output['frame_errors'] = bson.Binary(pickle.dumps(self._frame_errors,
                                                          protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'frame_errors' in serialized_representation:
            kwargs['frame_errors'] = pickle.loads(serialized_representation['frame_errors'])
        return super().deserialize(serialized_representation, db_client, **kwargs)
