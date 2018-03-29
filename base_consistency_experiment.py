# Copyright (c) 2017, John Skinner
import os
import logging
import numpy as np
import arvet.util.database_helpers as dh
import arvet.util.associate as ass
import arvet.util.transform as tf
import arvet.database.client
import arvet.config.path_manager
import arvet.batch_analysis.simple_experiment
import arvet.batch_analysis.task_manager
import arvet.util.trajectory_helpers as th
import data_helpers
import trajectory_helpers as old_th
import estimate_errors_benchmark
import midpoint_normalize


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
                         id_=id_, trial_map=trial_map, enabled=enabled, do_analysis=True)

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

        # Add a Estimate Error Benchmark so we can correlate it
        self.import_benchmark(
            name='Estimate Error',
            benchmark=estimate_errors_benchmark.EstimateErrorsBenchmark(),
            db_client=db_client
        )

    def perform_analysis(self, db_client: arvet.database.client.DatabaseClient):
        self._create_axis_plot(db_client)
        self._create_absolute_error_plot(db_client)
        self._create_noise_distribution_plots(db_client)
        self._create_error_distribution_plots(db_client)
        self._create_error_vs_same_direction_motion_plot(db_client)
        self._compute_error_correlation(db_client)

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

    def _create_noise_distribution_plots(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        noise_save_path = os.path.join(type(self).get_output_folder(), 'noise distribution')
        os.makedirs(noise_save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting noise distributions to {0} ...".format(noise_save_path))

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... plotting for {0}".format(system_name))
            for dataset_name, dataset_id in self.datasets.items():
                all_computed_motions = []
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) <= 0:
                    continue

                # Collect all the computed trajectories
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        all_computed_motions.append(trial_result.get_computed_camera_motions())

                # Find the average estimated trajectory
                average_motions = th.compute_average_trajectory(all_computed_motions)

                # Collect statistics on the noise for each frame motion
                times = []
                trans_noise = []
                rot_noise = []
                for computed_motions in all_computed_motions:
                    matches = ass.associate(average_motions, computed_motions, offset=0, max_difference=0.1)
                    for match in matches:
                        times.append(match[0])
                        trans_noise.append(computed_motions[match[1]].location - average_motions[match[0]].location)
                        rot_noise.append(tf.quat_diff(computed_motions[match[1]].rotation_quat(True),
                                                      average_motions[match[0]].rotation_quat(True)))

                trans_noise = np.array(trans_noise)
                noise_magnitudes = np.linalg.norm(trans_noise, axis=1)

                # Plot translational noise histograms
                title = "{0} on {1} translational noise distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 2, figsize=(20, 8), dpi=80)
                figure.suptitle(title)

                ax = axes[0]
                ax.set_title('per-axis noise distribution')
                ax.set_xlabel('noise (m)')
                ax.set_ylabel('density')
                for data, colour in [
                    (trans_noise[:, 0], 'red'),
                    (trans_noise[:, 1], 'green'),
                    (trans_noise[:, 2], 'blue')
                ]:
                    ax.hist(data, density=True, bins=300, alpha=0.3, color=colour)

                ax = axes[1]
                ax.set_title('total noise distribution')
                ax.set_xlabel('noise (m)')
                ax.set_ylabel('density')
                ax.hist(np.linalg.norm(trans_noise, axis=1), density=True, bins=300, color='blue')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)
                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot rotational noise histograms
                title = "{0} on {1} noise distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 1, figsize=(16, 8), dpi=80)
                figure.suptitle(title)
                axes.set_xlabel('noise (rad)')
                axes.set_ylabel('density')
                axes.hist(rot_noise, density=True, bins=300, color='blue')
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)
                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot noise vs time
                title = "{0} on {1} noise vs time".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(2, 1, figsize=(18, 10), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise distance (m)')
                ax.plot(times, noise_magnitudes, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = axes[1]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise angle (rad)')
                ax.plot(times, rot_noise, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

    def _create_error_distribution_plots(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot

        noise_save_path = os.path.join(type(self).get_output_folder(), 'error distribution')
        os.makedirs(noise_save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting error distributions to {0} ...".format(noise_save_path))

        for system_name, system_id in self.systems.items():
            logging.getLogger(__name__).info("    .... plotting for {0}".format(system_name))
            for dataset_name, dataset_id in self.datasets.items():
                all_computed_motions = []
                ground_truth_motions = None
                trial_result_list = self.get_trial_results(system_id, dataset_id)
                if len(trial_result_list) <= 0:
                    continue

                # Collect all the computed motions
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        if ground_truth_motions is None:
                            ground_truth_motions = trial_result.get_ground_truth_motions()
                        all_computed_motions.append(trial_result.get_computed_camera_motions())

                # Collect statistics on the error for each frame motion
                times = []
                trans_error = []
                rot_error = []
                for computed_motions in all_computed_motions:
                    matches = ass.associate(ground_truth_motions, computed_motions, offset=0, max_difference=0.1)
                    for match in matches:
                        times.append(match[0])
                        trans_error.append(computed_motions[match[1]].location -
                                           ground_truth_motions[match[0]].location)
                        rot_error.append(tf.quat_diff(computed_motions[match[1]].rotation_quat(True),
                                                      ground_truth_motions[match[0]].rotation_quat(True)))

                trans_error = np.array(trans_error)
                error_magnitudes = np.linalg.norm(trans_error, axis=1)

                # Plot translational noise histograms
                title = "{0} on {1} translational error distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 2, figsize=(20, 8), dpi=80)
                figure.suptitle(title)

                ax = axes[0]
                ax.set_title('per-axis error distribution')
                ax.set_xlabel('noise (m)')
                ax.set_ylabel('density')
                for data, colour in [
                    (trans_error[:, 0], 'red'),
                    (trans_error[:, 1], 'green'),
                    (trans_error[:, 2], 'blue')
                ]:
                    ax.hist(data, density=True, bins=300, alpha=0.3, color=colour)

                ax = axes[1]
                ax.set_title('total error distribution')
                ax.set_xlabel('noise (m)')
                ax.set_ylabel('density')
                ax.hist(error_magnitudes, density=True, bins=300, color='blue')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)
                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot rotational noise histograms
                title = "{0} on {1} error distribution".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 1, figsize=(16, 8), dpi=80)
                figure.suptitle(title)
                axes.set_xlabel('noise (rad)')
                axes.set_ylabel('density')
                axes.hist(rot_error, density=True, bins=300, color='blue')
                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)
                figure.savefig(os.path.join(noise_save_path, title + '.png'))
                pyplot.close(figure)

                # Plot noise vs time
                title = "{0} on {1} noise vs time".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(2, 1, figsize=(18, 10), dpi=80)
                figure.suptitle(title)
                ax = axes[0]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise distance (m)')
                ax.plot(times, error_magnitudes, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = axes[1]
                ax.set_xlabel('time (s)')
                ax.set_ylabel('noise angle (rad)')
                ax.plot(times, rot_error, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

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
                logging.getLogger(__name__).info("    .... error vs motion heatmaps for {0} on {1}".format(
                    system_name, dataset_name))

                # Collect errors from all the trial resuls
                forward_motion = []
                sideways_motion = []
                vertical_motion = []
                total_motion = []
                forward_error = []
                sideways_error = []
                vertical_error = []
                total_error = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                for trial_result_id in trial_result_list:
                    trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                    if trial_result is not None and trial_result.success:
                        ground_truth_motions = trial_result.get_ground_truth_motions()
                        computed_motions = trial_result.get_computed_camera_motions()

                        matches = ass.associate(ground_truth_motions, computed_motions, offset=0, max_difference=0.1)
                        for match in matches:
                            gt_motion = ground_truth_motions[match[0]].location
                            error = gt_motion - computed_motions[match[1]].location
                            forward_motion.append(gt_motion[0])
                            sideways_motion.append(gt_motion[1])
                            vertical_motion.append(gt_motion[2])
                            total_motion.append(np.linalg.norm(gt_motion))
                            forward_error.append(error[0])
                            sideways_error.append(error[1])
                            vertical_error.append(error[2])
                            total_error.append(np.linalg.norm(error))

                if len(total_error) <= 0:
                    # Make sure we have some data to plot
                    continue

                # Plot error vs motion in that direction
                title = "{0} on {1} error vs motion".format(system_name, dataset_name)
                figure, axes = pyplot.subplots(1, 4, figsize=(40, 10), dpi=80)
                figure.suptitle(title)

                ax = axes[0]
                ax.set_title('forward')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(forward_motion, forward_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('magma_r'))

                ax = axes[1]
                ax.set_title('sideways')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(sideways_motion, sideways_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('magma_r'))

                ax = axes[2]
                ax.set_title('vertical')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(vertical_motion, vertical_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('magma_r'))

                ax = axes[3]
                ax.set_title('total')
                ax.set_xlabel('motion (m)')
                ax.set_ylabel('error (m)')
                heatmap, xedges, yedges = np.histogram2d(total_motion, total_error, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('magma_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

    def _compute_error_correlation(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot
        import pandas as pd

        save_path = os.path.join(type(self).get_output_folder(), 'motion correlation')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting correlation and saving to {0} ...".format(save_path))
        for system_name, system_id in self.systems.items():
            for dataset_name, dataset_id in self.datasets.items():
                logging.getLogger(__name__).info("    .... correlations for {0} on {1}".format(
                    system_name, dataset_name))

                result_id = self.get_benchmark_result(system_id, dataset_id, self.benchmarks['Estimate Error'])
                benchmark_result = dh.load_object(db_client, db_client.trials_collection, result_id) \
                    if result_id is not None else None
                if benchmark_result is not None:

                    dataframe = pd.DataFrame(benchmark_result.observations, columns=[
                        'x error',
                        'y error',
                        'z error',
                        'translational error length',
                        'translational error direction',
                        'rotational error',
                        'x noise',
                        'y noise',
                        'z noise',
                        'translational noise length',
                        'translational noise direction',
                        'rotational noise',
                        'tracking',
                        'number of features',
                        'number of matches',
                        'x motion',
                        'y motion',
                        'z motion',
                        'distance moved',
                        'angle rotated'
                    ])
                    correlation = dataframe.corr()
                    print(correlation)

                    # Plot aggregate correlation to motion
                    title = "{0} on {1} statistics correlation".format(system_name, dataset_name)
                    figure = pyplot.figure(figsize=(14, 10), dpi=80)
                    figure.suptitle(title)

                    ax = figure.add_subplot(111)
                    ax.set_xticks(range(len(correlation.columns)))
                    ax.set_xticklabels(correlation.columns, rotation='vertical')
                    ax.set_yticks(range(len(correlation.columns)))
                    ax.set_yticklabels(correlation.columns)
                    ax.matshow(correlation, aspect='auto', cmap=pyplot.get_cmap('RdBu'),
                               norm=midpoint_normalize.MidpointNormalize(midpoint=0))

                    pyplot.tight_layout()
                    pyplot.subplots_adjust(top=0.90, right=0.99)

                    figure.savefig(os.path.join(save_path, title + '.png'))
                    pyplot.close(figure)
                    with open(os.path.join(save_path, title + '.txt'), 'w') as corr_file:
                        corr_file.write(str(correlation))

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_error_vs_motion(db_client)

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
