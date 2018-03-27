# Copyright (c) 2017, John Skinner
import os
import logging
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.associate as ass
import arvet.util.transform as tf
import arvet.database.client
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import arvet.util.trajectory_helpers as th
import data_helpers
import trajectory_helpers as old_th


class BaseConsistencyExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None,
                 trial_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param datasets:
        :param benchmarks:
        :param trial_map:
        :param enabled:
        :param id_:
        """
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=10,
                         id_=id_, trial_map=trial_map, enabled=enabled)

    def perform_analysis(self, db_client: arvet.database.client.DatabaseClient):
        self._create_axis_plot(db_client)
        self._create_absolute_error_plot(db_client)
        self._create_noise_and_error_distribution_plots(db_client)

    def _create_axis_plot(self, db_client: arvet.database.client.DatabaseClient):
        save_path = os.path.join(type(self).get_output_folder(), 'axis vs time')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting axes vs time to {0} ...".format(save_path))
        colours = ['orange', 'cyan', 'gold', 'magenta', 'green', 'brown', 'purple', 'red',
                   'navy', 'darkkhaki', 'darkgreen', 'crimson']

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... plotting for {0}".format(system_name))
            for dataset_name, dataset_id in self.datasets.items():

                # Collect statistics on all the trajectories
                trajectory_groups = []
                added_ground_truth = False

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_idx, trial_result_id in enumerate(trial_result_list):
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if not added_ground_truth:
                            trajectory_groups.append((
                                'Ground Truth',
                                [th.zero_trajectory(trial_result.get_ground_truth_camera_poses())],
                                {'c': 'black'}
                            ))
                            added_ground_truth = True
                        trajectory_groups.append((
                            'Repeat {0}'.format(trial_idx),
                            [trial_result.get_computed_camera_poses()],
                            {'c': colours[trial_idx % len(colours)]}
                        ))
                data_helpers.create_axis_plot(
                    title="{0} on {1}".format(system_name, dataset_name),
                    trajectory_groups=trajectory_groups,
                    save_path=save_path
                )

    def _create_noise_and_error_distribution_plots(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        noise_save_path = os.path.join(type(self).get_output_folder(), 'noise distribution')
        os.makedirs(noise_save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting noise distributions to {0} ...".format(noise_save_path))

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... plotting for {0}".format(system_name))
            for dataset_name, dataset_id in self.datasets.items():
                computed_trajectories = []
                ground_truth_motions = None
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) <= 0:
                    continue

                # Collect all the computed trajectories
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_motions is None:
                            ground_truth_motions = th.trajectory_to_motion_sequence(
                                trial_result.get_ground_truth_camera_poses())
                        computed_trajectories.append(trial_result.get_computed_camera_poses())

                # Find the average estimated trajectory
                average_motions = th.trajectory_to_motion_sequence(
                    th.compute_average_trajectory(computed_trajectories))

                # Collect statistics on the noise for each frame motion
                times = []
                x_noise = []
                y_noise = []
                z_noise = []
                noise_magnitude = []
                noise_angle = []
                for computed_trajectory in computed_trajectories:
                    computed_motions = th.trajectory_to_motion_sequence(computed_trajectory)
                    matches = ass.associate(average_motions, computed_motions, offset=0, max_difference=0.1)
                    for match in matches:
                        noise = computed_motions[match[1]].location - average_motions[match[0]].location
                        times.append(match[0])
                        x_noise.append(noise[0])
                        y_noise.append(noise[1])
                        z_noise.append(noise[2])
                        noise_magnitude.append(np.linalg.norm(noise))
                        noise_angle.append(tf.quat_diff(computed_motions[match[1]].rotation_quat(True),
                                                        average_motions[match[0]].rotation_quat(True)))

                # Plot error vs motion in that direction
                title = "{0} on {1} noise distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 5, figsize=(50, 8), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_title('x noise distribution')
                ax.set_xlabel('forward noise (m)')
                ax.set_ylabel('Probability')
                ax.hist(x_noise, density=True, bins=300, color='blue')

                ax = axes[1]
                ax.set_title('y noise distribution')
                ax.set_xlabel('sideways noise (m)')
                ax.set_ylabel('Probability')
                ax.hist(y_noise, density=True, bins=300, color='blue')

                ax = axes[2]
                ax.set_title('z noise distribution')
                ax.set_xlabel('upward noise (m)')
                ax.set_ylabel('Probability')
                ax.hist(z_noise, density=True, bins=300, color='blue')

                ax = axes[3]
                ax.set_title('distance noise distribution')
                ax.set_xlabel('noise magnitude (m)')
                ax.set_ylabel('Probability')
                ax.hist(noise_magnitude, density=True, bins=300, color='blue')

                ax = axes[4]
                ax.set_title('angle noise distribution')
                ax.set_xlabel('noise angle (rad)')
                ax.set_ylabel('Probability')
                ax.hist(noise_angle, density=True, bins=300, color='blue')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot noise vs time
                title = "{0} on {1} noise vs time".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(2, 1, figsize=(18, 10), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise distance (m)')
                ax.plot(times, noise_magnitude, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = axes[1]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise angle (rad)')
                ax.plot(times, noise_angle, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

    def _create_absolute_error_plot(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        noise_save_path = os.path.join(type(self).get_output_folder(), 'absolute error distribution')
        os.makedirs(noise_save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting absolute error distributions to {0} ...".format(noise_save_path))

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... plotting for {0}".format(system_name))
            for dataset_name, dataset_id in self.datasets.items():
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) <= 0:
                    continue

                # Collect error measurements for all trials
                times = []
                x_error = []
                y_error = []
                z_error = []
                error_magnitude = []
                error_angle = []
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        ground_truth_trajectory = trial_result.get_ground_truth_camera_poses()
                        computed_trajectory = trial_result.get_computed_camera_poses()
                        matches = ass.associate(ground_truth_trajectory, computed_trajectory,
                                                offset=0, max_difference=0.1)
                        for match in matches:
                            error = computed_trajectory[match[1]].location - ground_truth_trajectory[match[0]].location
                            times.append(match[0])
                            x_error.append(error[0])
                            y_error.append(error[1])
                            z_error.append(error[2])
                            error_magnitude.append(np.linalg.norm(error))
                            error_angle.append(tf.quat_diff(computed_trajectory[match[1]].rotation_quat(True),
                                                            ground_truth_trajectory[match[0]].rotation_quat(True)))

                # Plot error vs motion in that direction
                title = "{0} on {1} noise distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 5, figsize=(40, 8), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_title('x absolute error distribution')
                ax.set_xlabel('error (m)')
                ax.set_ylabel('Probability')
                ax.hist(x_error, density=True, bins=300, color='blue')

                ax = axes[1]
                ax.set_title('y absolute error distribution')
                ax.set_xlabel('error (m)')
                ax.set_ylabel('Probability')
                ax.hist(y_error, density=True, bins=300, color='blue')

                ax = axes[2]
                ax.set_title('z absolute error distribution')
                ax.set_xlabel('error (m)')
                ax.set_ylabel('Probability')
                ax.hist(z_error, density=True, bins=300, color='blue')

                ax = axes[3]
                ax.set_title('total absolute error distribution')
                ax.set_xlabel('error (m)')
                ax.set_ylabel('Probability')
                ax.hist(error_magnitude, density=True, bins=300, color='blue')

                ax = axes[4]
                ax.set_title('angle error distribution')
                ax.set_xlabel('angle error (rad)')
                ax.set_ylabel('Probability')
                ax.hist(error_angle, density=True, bins=300, color='blue')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot noise vs time
                title = "{0} on {1} error vs time".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(2, 1, figsize=(18, 10), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('error magnitude (m)')
                ax.plot(times, error_magnitude, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = axes[1]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('error angle (rad)')
                ax.plot(times, error_angle, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

    def _create_error_vs_same_direction_motion_plot(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        save_path = os.path.join(type(self).get_output_folder(), 'error vs motion heatmaps')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting error vs motion and saving to {0} ...".format(save_path))

        for dataset_name, dataset_id in self.datasets.items():
            for system_name, system_id in self.systems.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))

                forward_motion = []
                sideways_motion = []
                vertical_motion = []
                forward_error = []
                sideways_error = []
                vertical_error = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_traj = None
                gt_scale = None
                computed_trajectories = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_traj is None:
                            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
                            gt_scale = old_th.find_trajectory_scale(ground_truth_traj)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = old_th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")

                        computed_trajectories.append(traj)
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_traj, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                current_pose = None
                total_distance = 0
                for time in sorted(ground_truth_traj.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the distance to the prev frame
                    prev_pose = current_pose
                    current_pose = ground_truth_traj[time]
                    if prev_pose is not None:
                        motion = prev_pose.find_relative(current_pose)
                        to_prev_frame = np.linalg.norm(current_pose.location - prev_pose.location)
                    else:
                        motion = tf.Transform()
                        to_prev_frame = 0
                    total_distance += to_prev_frame

                    errors = []

                    # For each computed pose at this time
                    for idx in range(len(computed_trajectories)):
                        if time in timestamps[idx]:
                            computed_pose = computed_trajectories[idx][timestamps[idx][time]]
                            errors.append(np.linalg.norm(current_pose.location - computed_pose.location))
                            if prev_pose is not None:
                                computed_motion = prev_pose.find_relative(computed_pose)
                                forward_motion.append(motion.location[0])
                                sideways_motion.append(motion.location[1])
                                vertical_motion.append(motion.location[2])
                                forward_error.append(computed_motion.location[0] - motion.location[0])
                                sideways_error.append(computed_motion.location[1] - motion.location[1])
                                vertical_error.append(computed_motion.location[2] - motion.location[2])

                # Plot error vs motion in that direction
                title = "{0} on {1} error vs motion".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)
                ax = figure.add_subplot(131)
                ax.set_title('forward')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(forward_motion, forward_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(132)
                ax.set_title('sideways')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(sideways_motion, sideways_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(133)
                ax.set_title('vertical')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(vertical_motion, vertical_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_error_vs_motion(db_client)
        self._compute_error_vs_motion_correlation(db_client)
        self._plot_variance_vs_time(db_client)
        # self._plot_estimate_variance(db_client)
        # self._plot_error_vs_motion(db_client)
        # self._plot_variations(db_client)

    def _plot_error_vs_motion(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        save_path = os.path.join('figures', type(self).__name__, 'errors vs motions')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting error vs time and saving to {0} ...".format(save_path))

        for dataset_name, dataset_id in self.datasets.items():
            for system_name, system_id in self.systems.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))
                times = []
                motion_distances = []
                rotation_angles = []
                motion_errors = []
                mean_motion_errors = []
                rotation_errors = []
                mean_rotation_errors = []
                motion_noise = []
                rotation_noise = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_motions = None
                gt_scale = None
                computed_motion_sequences = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_motions is None:
                            ground_truth_motions = trial_result.get_ground_truth_camera_poses()
                            gt_scale = old_th.find_trajectory_scale(ground_truth_motions)
                            ground_truth_motions = old_th.trajectory_to_motion_sequence(ground_truth_motions)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = old_th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")
                        computed_motion_sequences.append(old_th.trajectory_to_motion_sequence(traj))
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_motions, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                for time in sorted(ground_truth_motions.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the mean estimated motion for this time
                    computed_motions = [
                        computed_motion_sequences[idx][timestamps[idx][time]].location
                        for idx in range(len(computed_motion_sequences))
                        if time in timestamps[idx] and timestamps[idx][time] in computed_motion_sequences[idx]
                    ]
                    computed_rotations = [
                        computed_motion_sequences[idx][timestamps[idx][time]].rotation_quat(True)
                        for idx in range(len(computed_motion_sequences))
                        if time in timestamps[idx] and timestamps[idx][time] in computed_motion_sequences[idx]
                    ]
                    assert len(computed_motions) == len(computed_rotations)
                    if len(computed_motions) > 0:
                        mean_computed_motion = np.mean(computed_motions, axis=0)
                        mean_computed_rotation = data_helpers.quat_mean(computed_rotations)
                        times += [time for _ in range(len(computed_motions))]
                        motion_distances += [
                            np.linalg.norm(ground_truth_motions[time].location)
                            for _ in range(len(computed_motions))
                        ]
                        rotation_angles += [
                            data_helpers.quat_angle(ground_truth_motions[time].rotation_quat(True))
                            for _ in range(len(computed_rotations))
                        ]
                        motion_errors += [
                            np.linalg.norm(computed_motion - ground_truth_motions[time].location)
                            for computed_motion in computed_motions
                        ]
                        motion_noise += [
                            np.linalg.norm(computed_motion - mean_computed_motion)
                            for computed_motion in computed_motions
                        ]
                        mean_motion_errors += [
                            np.linalg.norm(mean_computed_motion - ground_truth_motions[time].location)
                            for _ in range(len(computed_motions))
                        ]
                        rotation_errors += [
                            data_helpers.quat_diff(computed_rotation, ground_truth_motions[time].rotation_quat(True))
                            for computed_rotation in computed_rotations
                        ]
                        rotation_noise += [
                            data_helpers.quat_diff(computed_rotation, mean_computed_rotation)
                            for computed_rotation in computed_rotations
                        ]
                        mean_rotation_errors += [
                            data_helpers.quat_diff(mean_computed_rotation,
                                                   ground_truth_motions[time].rotation_quat(True))
                            for _ in range(len(computed_rotations))
                        ]

                # Plot every combination of motion vs error as a plot and a heatmap
                figure, axes = pyplot.subplots(12, 2, squeeze=False,
                                               figsize=(14, 66), dpi=80)
                fig_title = '{0} on {1} errors vs motions.png'.format(system_name, dataset_name)
                figure.suptitle(fig_title)
                plot_idx = 0
                for x_title_name, x_axis_name, x_data in [
                    ('motion', 'distance moved (m)', np.array(motion_distances)),
                    ('rotation', 'angle rotated (rad)', np.array(rotation_angles))
                ]:
                    for y_title_name, y_axis_name, y_data in [
                        ('motion error', 'motion error (m)', np.array(motion_errors)),
                        ('motion noise', 'motion noise (m)', np.array(motion_noise)),
                        ('mean motion error', 'mean motion error (m)', np.array(mean_motion_errors)),
                        ('rotation error', 'rotation error (rad)', np.array(rotation_errors)),
                        ('rotation noise', 'rotation noise (rad)', np.array(rotation_noise)),
                        ('mean rotation error', 'mean rotation error (rad)', np.array(mean_rotation_errors)),
                    ]:
                        x_limits = data_helpers.compute_window(x_data, std_deviations=4)
                        y_limits = data_helpers.compute_window(y_data, std_deviations=4)
                        x_outliers = data_helpers.compute_outliers(x_data, x_limits)
                        y_outliers = data_helpers.compute_outliers(y_data, y_limits)

                        ax = axes[plot_idx][0]
                        ax.set_title("{0} vs {1}".format(y_title_name, x_title_name))
                        ax.set_xlim(x_limits)
                        ax.set_ylim(y_limits)
                        ax.set_xlabel(x_axis_name + " ({0} outliers)".format(x_outliers))
                        ax.set_ylabel(y_axis_name + " ({0} outliers)".format(y_outliers))
                        ax.plot(x_data, y_data, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                        ax = axes[plot_idx][1]
                        ax.set_title("{0} vs {1} histogram".format(y_title_name, x_title_name))
                        ax.set_xlabel(x_axis_name + " ({0} outliers)".format(x_outliers))
                        ax.set_ylabel(y_axis_name + " ({0} outliers)".format(y_outliers))
                        heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=300, range=[x_limits, y_limits])
                        ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                                  aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                        plot_idx += 1
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.98, right=0.99)
                figure.savefig(os.path.join(save_path, fig_title + '.png'))
                pyplot.close(figure)

    def _compute_error_vs_motion_correlation(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        save_path = os.path.join('figures', type(self).__name__, 'covariance and correlation')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting correlation and saving to {0} ...".format(save_path))
        for system_name, system_id in self.systems.items():
            sample_observations = []
            aggregate_observations = []

            for dataset_name, dataset_id in self.datasets.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_motions = None
                gt_scale = None
                computed_motion_sequences = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_motions is None:
                            ground_truth_motions = trial_result.get_ground_truth_camera_poses()
                            gt_scale = old_th.find_trajectory_scale(ground_truth_motions)
                            ground_truth_motions = old_th.trajectory_to_motion_sequence(ground_truth_motions)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = old_th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")

                        computed_motion_sequences.append(old_th.trajectory_to_motion_sequence(traj))
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_motions, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                for time in sorted(ground_truth_motions.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the mean estimated motion for this time
                    computed_motions = [
                        computed_motion_sequences[idx][timestamps[idx][time]].location
                        for idx in range(len(computed_motion_sequences))
                        if time in timestamps[idx] and timestamps[idx][time] in computed_motion_sequences[idx]
                    ]
                    computed_rotations = [
                        computed_motion_sequences[idx][timestamps[idx][time]].rotation_quat(True)
                        for idx in range(len(computed_motion_sequences))
                        if time in timestamps[idx] and timestamps[idx][time] in computed_motion_sequences[idx]
                    ]
                    if len(computed_motions) > 0:
                        gt_motion = ground_truth_motions[time].location
                        gt_rotation = ground_truth_motions[time].rotation_quat(True)
                        mean_computed_motion = np.mean(computed_motions, axis=0)
                        mean_computed_rotation = data_helpers.quat_mean(computed_rotations)
                        std_computed_motion = np.std(computed_motions, axis=0)

                        aggregate_observations.append([
                            std_computed_motion[0],
                            std_computed_motion[1],
                            std_computed_motion[2],
                            mean_computed_motion[0],
                            mean_computed_motion[1],
                            mean_computed_motion[2],
                            data_helpers.quat_angle(mean_computed_rotation),
                            gt_motion[0],
                            gt_motion[1],
                            gt_motion[2],
                            data_helpers.quat_angle(gt_rotation)
                        ])

                        for idx, computed_motion in enumerate(computed_motions):
                            trans_error = computed_motion - gt_motion
                            rot_error = data_helpers.quat_diff(computed_rotations[idx], gt_rotation)
                            trans_variance = computed_motion - mean_computed_motion
                            rot_variance = data_helpers.quat_diff(computed_rotations[idx], mean_computed_rotation)
                            sample_observations.append([
                                trans_error[0],
                                trans_error[1],
                                trans_error[2],
                                rot_error,
                                trans_variance[0],
                                trans_variance[1],
                                trans_variance[2],
                                rot_variance,
                                mean_computed_motion[0],
                                mean_computed_motion[1],
                                mean_computed_motion[2],
                                data_helpers.quat_angle(mean_computed_rotation),
                                gt_motion[0],
                                gt_motion[1],
                                gt_motion[2],
                                data_helpers.quat_angle(gt_rotation)
                            ])
            aggregate_observations = np.array(aggregate_observations, dtype=np.float)
            sample_observations = np.array(sample_observations, dtype=np.float)

            aggregate_covariance = np.cov(aggregate_observations.T)
            sample_covariance = np.cov(sample_observations.T)

            std_deviations = np.std(aggregate_observations, axis=0)
            aggregate_correlation = np.divide(aggregate_covariance, np.outer(std_deviations, std_deviations))

            std_deviations = np.std(sample_observations, axis=0)
            sample_correlation = np.divide(sample_covariance, np.outer(std_deviations, std_deviations))

            print("Correlation for aggregate statistics")
            print("Rows/columns are std_x, std_y, std_z, mean_computed_x, mean_computed_y, mean_computed_x, "
                  "mean_computed_rot, actual_x, actual_y, actual_z, actual_rot")
            print(aggregate_correlation.tolist())

            print("Correlation for per sample statistics")
            print("Rows/columns are x_error, y_error, z_error, rot_error, x_variance, y_variance, z_variance, "
                  "rot_variance, mean_computed_y, mean_computed_x, mean_computed_rot, actual_x, actual_y, "
                  "actual_z, actual_rot")
            print(sample_correlation.tolist())

            # Plot aggregate correlation to motion
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Aggregate statistic correlation for {0}".format(system_name))

            ax = figure.add_subplot(111)
            labels = ['std x', 'std y', 'std z', 'comp x motion', 'comp y motion', 'comp z motion', 'comp rotation',
                      'x motion', 'y motion', 'z motion', 'rotation']
            ax.set_xticks([i for i in range(aggregate_correlation.shape[0])])
            ax.set_xticklabels(labels, rotation='vertical')
            ax.set_yticks([i for i in range(aggregate_correlation.shape[1])])
            ax.set_yticklabels(labels)
            ax.imshow(aggregate_correlation.T, aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.90, right=0.99)

            figure.savefig(os.path.join(save_path, "Aggregate statistic correlation for {0}.png".format(system_name)))
            pyplot.close(figure)

            # Plot sample correlation to motion
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("Sample statistic correlation for {0}".format(system_name))

            ax = figure.add_subplot(111)
            labels = ['x error', 'y error', 'z error', 'rot error', 'x variance', 'y variance', 'z variance',
                      'rot variance', 'comp x motion', 'comp y motion', 'comp z motion', 'comp rotation',
                      'x motion', 'y motion', 'z motion', 'rotation']
            ax.set_xticks([i for i in range(sample_correlation.shape[0])])
            ax.set_xticklabels(labels, rotation='vertical')
            ax.set_yticks([i for i in range(sample_correlation.shape[1])])
            ax.set_yticklabels(labels)
            ax.imshow(sample_correlation.T, aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.90, right=0.99)

            figure.savefig(os.path.join(save_path, "Sample statistic correlation for {0}.png".format(system_name)))
            pyplot.close(figure)
        pyplot.show()

    def _plot_variance_vs_time(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        variance_save_path = os.path.join('figures', type(self).__name__, 'variance vs time')
        error_save_path = os.path.join('figures', type(self).__name__, 'motion error vs time')
        os.makedirs(variance_save_path, exist_ok=True)
        os.makedirs(error_save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting variance vs time and saving to {0} ...".format(variance_save_path))

        for dataset_name, dataset_id in self.datasets.items():
            for system_name, system_id in self.systems.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))
                times = []
                absolute_variance = {'x': [], 'y': [], 'z': []}
                motion_times = []
                motion_errors = {'x': [], 'y': [], 'z': []}
                motion_variance = {'x': [], 'y': [], 'z': []}
                motion_variance_normalized = {'x': [], 'y': [], 'z': []}

                aggregate_times = []
                aggregate_motion_std = []
                aggregate_motion_mean_error = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_motions = None
                gt_scale = None
                computed_trajectories = []
                computed_motion_sequences = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_motions is None:
                            ground_truth_motions = trial_result.get_ground_truth_camera_poses()
                            gt_scale = old_th.find_trajectory_scale(ground_truth_motions)
                            ground_truth_motions = old_th.trajectory_to_motion_sequence(ground_truth_motions)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = old_th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")

                        computed_trajectories.append(traj)
                        computed_motion_sequences.append(old_th.trajectory_to_motion_sequence(traj))
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_motions, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                for time in sorted(ground_truth_motions.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the mean estimated location for this time
                    computed_locations = [
                        computed_trajectories[idx][timestamps[idx][time]].location
                        for idx in range(len(computed_trajectories))
                        if time in timestamps[idx]
                    ]
                    if len(computed_locations) > 0:
                        mean_computed_location = np.mean(computed_locations, axis=0)
                        times += [time for _ in range(len(computed_locations))]
                        for idx, axis in enumerate(['x', 'y', 'z']):
                            absolute_variance[axis] += [computed_locations[idx] - mean_computed_location[idx]
                                                        for computed_locations in computed_locations]

                    # Find the mean estimated motion for this time
                    computed_motions = [
                        computed_motion_sequences[idx][timestamps[idx][time]].location
                        for idx in range(len(computed_motion_sequences))
                        if time in timestamps[idx] and timestamps[idx][time] in computed_motion_sequences[idx]
                    ]
                    if len(computed_motions) > 0:
                        mean_computed_motion = np.mean(computed_motions, axis=0)
                        std_computed_motion = np.std(computed_motions, axis=0)
                        motion_times += [time for _ in range(len(computed_motions))]

                        aggregate_times.append(time)
                        aggregate_motion_std.append(std_computed_motion)
                        aggregate_motion_mean_error.append(mean_computed_motion - ground_truth_motions[time].location)

                        for idx, axis in enumerate(['x', 'y', 'z']):
                            motion_variance[axis] += [computed_motion[idx] - mean_computed_motion[idx]
                                                      for computed_motion in computed_motions]
                            motion_variance_normalized[axis] += [
                                (computed_motion[idx] - mean_computed_motion[idx]) / std_computed_motion[idx]
                                for computed_motion in computed_motions
                            ]
                            motion_errors[axis] += [computed_motion[idx] - ground_truth_motions[time].location[idx]
                                                    for computed_motion in computed_motions]

                # Plot absolute location variance vs time as a plot and a heatmap
                title = "{0} on {1} estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                hist_title = "{0} on {1} estimate variation heatmap".format(system_name, dataset_name)
                hist_figure = pyplot.figure(figsize=(30, 10), dpi=80)
                hist_figure.suptitle(hist_title)

                for idx, axis in enumerate(['x', 'y', 'z']):
                    ax = figure.add_subplot(131 + idx)
                    ax.set_title(axis)
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('{0} variance (m)'.format(axis))
                    ax.plot(times, absolute_variance[axis], c='blue', alpha=0.5, marker='.',
                            markersize=2, linestyle='None')

                    ax = hist_figure.add_subplot(131 + idx)
                    ax.set_title(axis)
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('{0} variance (m)'.format(axis))
                    heatmap, xedges, yedges = np.histogram2d(times, absolute_variance[axis], bins=300)
                    ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                              aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(variance_save_path, title + '.png'))
                pyplot.close(figure)
                hist_figure.savefig(os.path.join(variance_save_path, hist_title + '.png'))
                pyplot.close(hist_figure)

                # Plot computed motion variance vs time as a plot and heatmap
                title = "{0} on {1} motion estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                hist_title = "{0} on {1} motion estimate variation heatmap".format(system_name, dataset_name)
                hist_figure = pyplot.figure(figsize=(30, 10), dpi=80)
                hist_figure.suptitle(hist_title)

                combined_title = "{0} on {1} combined motion estimate variation".format(system_name, dataset_name)
                combined_figure = pyplot.figure(figsize=(30, 10), dpi=80)
                combined_figure.suptitle(combined_title)
                combined_ax = combined_figure.add_subplot(111)
                combined_ax.set_xlabel('time (s)')
                combined_ax.set_ylabel('variance (m)')

                for idx, (axis, colour) in enumerate([('x', 'red'), ('y', 'green'), ('z', 'blue')]):
                    ax = figure.add_subplot(131 + idx)
                    ax.set_title(axis)
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('{0} variance (m)'.format(axis))
                    ax.plot(motion_times, motion_variance[axis], c='blue', alpha=0.5, marker='.', markersize=2,
                            linestyle='None')

                    ax = hist_figure.add_subplot(131 + idx)
                    ax.set_title(axis)
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('{0} variance (m)'.format(axis))
                    heatmap, xedges, yedges = np.histogram2d(motion_times, motion_variance[axis], bins=300)
                    ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                              aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                    combined_ax.plot(motion_times, motion_variance[axis], c=colour, alpha=0.5, marker='.',
                                     markersize=2, linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(variance_save_path, title + '.png'))
                pyplot.close(figure)
                hist_figure.savefig(os.path.join(variance_save_path, hist_title + '.png'))
                pyplot.close(hist_figure)
                combined_figure.savefig(os.path.join(variance_save_path, combined_title + '.png'))
                pyplot.close(combined_figure)

                # Plot histogram of normalized variance
                title = "{0} on {1} motion variance histogram".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(111)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('variance (m)')
                if np.min(motion_variance_normalized['x']) < np.max(motion_variance_normalized['x']):
                    ax.hist(motion_variance_normalized['x'], bins=100, color='red', alpha=0.3, label='x axis')
                if np.min(motion_variance_normalized['y']) < np.max(motion_variance_normalized['y']):
                    ax.hist(motion_variance_normalized['y'], bins=100, color='green', alpha=0.3, label='y axis')
                if np.min(motion_variance_normalized['z']) < np.max(motion_variance_normalized['z']):
                    ax.hist(motion_variance_normalized['z'], bins=100, color='blue', alpha=0.3, label='z axis')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(variance_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot aggregate standard deviation over time
                title = "{0} on {1} standard deivation vs time".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(111)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('standard deviation (m)')
                aggregate_motion_std = np.array(aggregate_motion_std)
                for idx, (axis, colour) in enumerate([('x', 'red'), ('y', 'green'), ('z', 'blue')]):
                    ax.plot(aggregate_times, aggregate_motion_std[:, idx], c=colour, alpha=0.5, marker='None',
                            markersize=2, linestyle='-', label='{0} axis'.format(axis))
                ax.legend()
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(variance_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot the motion error vs time
                title = "{0} on {1} motion error vs time".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)
                motion_error_ax = figure.add_subplot(111)
                motion_error_ax.set_xlabel('time (s)')
                motion_error_ax.set_ylabel('error (m)')

                hist_title = "{0} on {1} motion error vs time heatmap".format(system_name, dataset_name)
                hist_figure = pyplot.figure(figsize=(30, 10), dpi=80)
                hist_figure.suptitle(hist_title)

                for idx, (axis, colour) in enumerate([('x', 'red'), ('y', 'green'), ('z', 'blue')]):
                    motion_error_ax.plot(motion_times, motion_errors[axis], c=colour, alpha=0.5, marker='.',
                                         markersize=2, linestyle='None')

                    ax = hist_figure.add_subplot(131 + idx)
                    ax.set_title(axis)
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('{0} variance (m)'.format(axis))
                    heatmap, xedges, yedges = np.histogram2d(motion_times, motion_errors[axis], bins=300)
                    ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                              aspect='auto', cmap=pyplot.get_cmap('inferno_r'))
                motion_error_ax.legend()

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(error_save_path, title + '.png'))
                pyplot.close(figure)
                hist_figure.savefig(os.path.join(error_save_path, hist_title + '.png'))
                pyplot.close(hist_figure)

        # Show all the graphs remaining
        pyplot.show()

    def _plot_estimate_variance(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        save_path = os.path.join('figures', type(self).__name__, 'variance heatmaps')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting error vs motion and saving to {0} ...".format(save_path))

        colours = ['red', 'blue', 'green', 'cyan', 'gold', 'magenta', 'brown', 'purple', 'orange']
        for dataset_name, dataset_id in self.datasets.items():
            colour_map = {}
            colour_idx = 0

            for system_name, system_id in self.systems.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))

                colour_map[system_name] = colours[colour_idx]
                colour_idx += 1

                x_variance = []
                y_variance = []
                z_variance = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_traj = None
                gt_scale = None
                computed_trajectories = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_traj is None:
                            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
                            gt_scale = old_th.find_trajectory_scale(ground_truth_traj)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = old_th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")

                        computed_trajectories.append(traj)
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_traj, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                current_pose = None
                for time in sorted(ground_truth_traj.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the distance to the prev frame
                    prev_pose = current_pose
                    current_pose = ground_truth_traj[time]
                    if prev_pose is not None:
                        # Find the mean estimated location for this time
                        computed_motions = [
                            prev_pose.find_relative(computed_trajectories[idx][timestamps[idx][time]]).location
                            for idx in range(len(computed_trajectories))
                            if time in timestamps[idx]
                        ]
                        mean_computed_motion = np.mean(computed_motions, axis=0)
                        x_variance += [computed_motion[0] - mean_computed_motion[0]
                                       for computed_motion in computed_motions]
                        y_variance += [computed_motion[1] - mean_computed_motion[1]
                                       for computed_motion in computed_motions]
                        z_variance += [computed_motion[2] - mean_computed_motion[2]
                                       for computed_motion in computed_motions]

                # Plot error vs motion in that direction
                title = "{0} on {1} estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                # Find the largest and smallest variance in each axis so we can square our plots
                ax_min = np.min(x_variance + y_variance + z_variance)
                ax_max = np.max(x_variance + y_variance + z_variance)
                ax_std = 3 * np.std(x_variance + y_variance + z_variance)
                if ax_min < -1 * ax_std:
                    logging.getLogger(__name__).warning("    axis min {0} is more than 3 standard deviations away, "
                                                        "clamping".format(ax_min))
                    ax_min = -1 * ax_std
                if ax_max < ax_std:
                    logging.getLogger(__name__).warning("    axis max {0} is more than 3 standard deviations away, "
                                                        "clamping".format(ax_max))
                    ax_max = ax_std

                hist_range = [[ax_min, ax_max], [ax_min, ax_max]]

                ax = figure.add_subplot(131)
                ax.set_title('front')
                ax.set_xlabel('y')
                ax.set_ylabel('z')
                heatmap, xedges, yedges = np.histogram2d(y_variance, z_variance, bins=300, range=hist_range)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(132)
                ax.set_title('side')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                heatmap, xedges, yedges = np.histogram2d(x_variance, z_variance, bins=300, range=hist_range)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(133)
                ax.set_title('top')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                heatmap, xedges, yedges = np.histogram2d(x_variance, y_variance, bins=300, range=hist_range)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

        # Show all the graphs remaining
        pyplot.show()

    def _plot_variations(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the ground-truth and computed trajectories for each system for each trajectory.
        This is important for validation
        :param db_client:
        :return:
        """
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        logging.getLogger(__name__).info("Plotting variations...")

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... variations for {0}".format(system_name))

            # Collect statistics on all the trajectories
            times = []
            variance = []
            rot_variance = []
            time_variance = []
            trans_error = []
            distances = []
            distances_to_prev_frame = []
            normalized_points = []
            for dataset_name, dataset_id in self.datasets.items():

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_traj = None
                computed_trajectories = []
                timestamps = []

                if len(trial_result_list) <= 0:
                    continue

                # Collect all the trial results
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_traj is None:
                            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
                        traj = trial_result.get_computed_camera_poses()
                        computed_trajectories.append(traj)
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_traj, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                prev_location = None
                total_distance = 0
                for time in sorted(ground_truth_traj.keys()):
                    if sum(1 for idx in range(len(timestamps)) if time in timestamps[idx]) <= 1:
                        # Skip locations/results which appear in only one trajectory
                        continue

                    # Find the mean estimated location for this time
                    mean_estimate = np.mean([computed_trajectories[idx][timestamps[idx][time]].location
                                             for idx in range(len(computed_trajectories))
                                             if time in timestamps[idx]], axis=0)

                    mean_time = np.mean([timestamps[idx][time] for idx in range(len(computed_trajectories))
                                         if time in timestamps[idx]])

                    mean_orientation = data_helpers.quat_mean([
                        computed_trajectories[idx][timestamps[idx][time]].rotation_quat(True)
                        for idx in range(len(computed_trajectories))
                        if time in timestamps[idx]
                    ])

                    # Find the distance to the prev frame
                    current_location = ground_truth_traj[time].location
                    if prev_location is not None:
                        to_prev_frame = np.linalg.norm(current_location - prev_location)
                    else:
                        to_prev_frame = 0
                    total_distance += to_prev_frame
                    prev_location = current_location

                    # For each computed pose at this time
                    for idx in range(len(computed_trajectories)):
                        if time in timestamps[idx]:
                            computed_location = computed_trajectories[idx][timestamps[idx][time]].location
                            times.append(time)
                            variance.append(np.dot(mean_estimate - computed_location,
                                                   mean_estimate - computed_location))
                            rot_variance.append(
                                data_helpers.quat_diff(
                                    mean_orientation,
                                    computed_trajectories[idx][timestamps[idx][time]].rotation_quat(True)
                                )**2
                            )
                            time_variance.append((mean_time - timestamps[idx][time]) *
                                                 (mean_time - timestamps[idx][time]))
                            trans_error.append(np.linalg.norm(current_location - computed_location))
                            distances.append(total_distance)
                            distances_to_prev_frame.append(to_prev_frame)
                            normalized_points.append(computed_location - mean_estimate)

            # Plot precision vs error
            # figure = pyplot.figure(figsize=(14, 10), dpi=80)
            # figure.suptitle("{0} precision vs error".format(system_name))
            # ax = figure.add_subplot(111)
            # ax.set_xlabel('error')
            # ax.set_ylabel('absolute deviation')
            # ax.plot(trans_error, trans_precision, 'o', alpha=0.5, markersize=1)

            # Histogram the distribution around the mean estimated
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} location variance".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('meters')
            ax.set_ylabel('frequency')
            ax.hist(variance, 100, label='distance')

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} distribution of time variance".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('seconds')
            ax.set_ylabel('frequency')
            ax.hist(time_variance, 100, label='time')

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} distribution of rotational variance".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('radians')
            ax.set_ylabel('frequency')
            ax.hist(rot_variance, 100, label='angle')

            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} distribution of trans error".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('error (meters)')
            ax.set_ylabel('frequency')
            ax.hist(trans_error, 100, label='angle')

            # Plot motion vs error
            figure = pyplot.figure(figsize=(14, 10), dpi=80)
            figure.suptitle("{0} motion vs error".format(system_name))
            ax = figure.add_subplot(111)
            ax.set_xlabel('error')
            ax.set_ylabel('absolute deviation')
            ax.plot(distances_to_prev_frame, trans_error, 'o', alpha=0.5, markersize=1)

            pyplot.tight_layout()
            pyplot.subplots_adjust(top=0.95, right=0.99)

        # Show all the graphs
        pyplot.show()

    def export_data(self, db_client: arvet.database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to
        :param db_client:
        :return:
        """
        for dataset_name, dataset_id in self.datasets.items():
            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in self.systems.items():
                trial_result_list = self.get_trial_result(system_id, dataset_id)
                for idx, trial_result_id in enumerate(trial_result_list):
                    label = "{0} on {1} repeat {2}".format(system_name, dataset_name, idx)
                    trial_results[label] = trial_result_id
            data_helpers.export_trajectory_as_json(trial_results, "Consistency " + dataset_name, db_client)
