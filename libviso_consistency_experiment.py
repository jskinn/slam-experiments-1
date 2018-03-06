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
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2
import data_helpers
import trajectory_helpers as th


class LibVisOConsistencyExperiment(arvet.batch_analysis.simple_experiment.SimpleExperiment):

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param datasets:
        :param benchmarks:
        :param trial_map:
        :param result_map:
        :param enabled:
        :param id_:
        """
        super().__init__(systems=systems, datasets=datasets, benchmarks=benchmarks, repeats=10,
                         id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)

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
        # --------- REAL WORLD DATASETS -----------
        # Import KITTI datasets
        for sequence_num in range(11):
            self.import_dataset(
                name='KITTI {0:02}'.format(sequence_num),
                module_name='arvet_slam.dataset.kitti.kitti_loader',
                path=os.path.join('datasets', 'KITTI', 'dataset'),
                additional_args={'sequence_number': sequence_num},
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import EuRoC datasets
        for name, path in [
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy')),
            ('EuRoC MH_02_easy', os.path.join('datasets', 'EuRoC', 'MH_02_easy')),
            ('EuRoC MH_02_medium', os.path.join('datasets', 'EuRoC', 'MH_03_medium')),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult')),
            ('EuRoC MH_05_difficult', os.path.join('datasets', 'EuRoC', 'MH_05_difficult')),
            ('EuRoC V1_01_easy', os.path.join('datasets', 'EuRoC', 'V1_01_easy')),
            ('EuRoC V1_02_medium', os.path.join('datasets', 'EuRoC', 'V1_02_medium')),
            ('EuRoC V1_03_difficult', os.path.join('datasets', 'EuRoC', 'V1_03_difficult')),
            ('EuRoC V2_01_easy', os.path.join('datasets', 'EuRoC', 'V2_01_easy')),
            ('EuRoC V2_02_medium', os.path.join('datasets', 'EuRoC', 'V2_02_medium')),
            ('EuRoC V2_03_difficult', os.path.join('datasets', 'EuRoC', 'V2_03_difficult'))
        ]:
            self.import_dataset(
                name=name,
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # LibVisO
        self.import_system(
            name='LibVisO',
            db_client=db_client,
            system=libviso2.LibVisOSystem(),
        )

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        :param db_client:
        :return:
        """
        # Visualize the different trajectories in each group
        self._plot_variance_vs_time(db_client)
        self._plot_estimate_variance(db_client)
        self._plot_error_vs_motion(db_client)
        self._plot_variations(db_client)

    def _plot_variance_vs_time(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        save_path = os.path.join('figures', type(self).__name__, 'variance vs time')
        os.makedirs(save_path, exist_ok=True)

        logging.getLogger(__name__).info("Plotting variance vs time and saving to {0} ...".format(save_path))

        for dataset_name, dataset_id in self.datasets.items():
            for system_name, system_id in self.systems.items():
                logging.getLogger(__name__).info("    .... distributions for {0} on {1}".format(system_name,
                                                                                                dataset_name))
                times = []
                x_variance = []
                y_variance = []
                z_variance = []
                motion_times = []
                x_motion_variance = []
                y_motion_variance = []
                z_motion_variance = []
                x_motion_normalized = []
                y_motion_normalized = []
                z_motion_normalized = []

                trial_result_list = self.get_trial_results(system_id, dataset_id)
                ground_truth_traj = None
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
                        if ground_truth_traj is None:
                            ground_truth_traj = trial_result.get_ground_truth_camera_poses()
                            gt_scale = th.find_trajectory_scale(ground_truth_traj)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = th.rescale_trajectory(traj, gt_scale)
                            else:
                                logging.getLogger(__name__).warning("Cannot rescale trajectory, missing ground truth")

                        computed_trajectories.append(traj)
                        computed_motion_sequences.append(th.trajectory_to_motion_sequence(traj))
                        timestamps.append({k: v for k, v in ass.associate(ground_truth_traj, traj,
                                                                          max_difference=0.1, offset=0)})

                # Now that we have all the trajectories, we can measure consistency
                for time in sorted(ground_truth_traj.keys()):
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
                        x_variance += [computed_locations[0] - mean_computed_location[0]
                                       for computed_locations in computed_locations]
                        y_variance += [computed_locations[1] - mean_computed_location[1]
                                       for computed_locations in computed_locations]
                        z_variance += [computed_locations[2] - mean_computed_location[2]
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
                        x_motion_variance += [computed_motion[0] - mean_computed_motion[0]
                                              for computed_motion in computed_motions]
                        y_motion_variance += [computed_motion[1] - mean_computed_motion[1]
                                              for computed_motion in computed_motions]
                        z_motion_variance += [computed_motion[2] - mean_computed_motion[2]
                                              for computed_motion in computed_motions]
                        x_motion_normalized += [(computed_motion[0] - mean_computed_motion[0]) / std_computed_motion[0]
                                                for computed_motion in computed_motions]
                        y_motion_normalized += [(computed_motion[1] - mean_computed_motion[1]) / std_computed_motion[1]
                                                for computed_motion in computed_motions]
                        z_motion_normalized += [(computed_motion[2] - mean_computed_motion[2]) / std_computed_motion[2]
                                                for computed_motion in computed_motions]

                # Plot location variance vs time
                title = "{0} on {1} estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(131)
                ax.set_title('x')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('x variance (m)')
                ax.plot(times, x_variance, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = figure.add_subplot(132)
                ax.set_title('y')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('y variance (m)')
                ax.plot(times, y_variance, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                ax = figure.add_subplot(133)
                ax.set_title('z')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('z variance (m)')
                ax.plot(times, z_variance, c='blue', alpha=0.5, marker='.', markersize=2, linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

                # Plot location variance vs time as a heatmap
                title = "{0} on {1} estimate variation heatmap".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(131)
                ax.set_title('x')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('x variance (m)')
                heatmap, xedges, yedges = np.histogram2d(times, x_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(132)
                ax.set_title('y')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('y variance (m)')
                heatmap, xedges, yedges = np.histogram2d(times, y_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(133)
                ax.set_title('z')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('z variance (m)')
                heatmap, xedges, yedges = np.histogram2d(times, z_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

                # Plot computed motion variance vs time
                title = "{0} on {1} motion estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(131)
                ax.set_title('x')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('x variance (m)')
                ax.plot(motion_times, x_motion_variance, c='blue', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')

                ax = figure.add_subplot(132)
                ax.set_title('y')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('y variance (m)')
                ax.plot(motion_times, y_motion_variance, c='blue', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')

                ax = figure.add_subplot(133)
                ax.set_title('z')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('z variance (m)')
                ax.plot(motion_times, z_motion_variance, c='blue', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

                # Plot computed motion variance in each axis
                title = "{0} on {1} combined motion estimate variation".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(111)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('variance (m)')
                ax.plot(motion_times, x_motion_variance, c='red', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')
                ax.plot(motion_times, y_motion_variance, c='green', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')
                ax.plot(motion_times, z_motion_variance, c='blue', alpha=0.5, marker='.', markersize=2,
                        linestyle='None')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

                # Plot histogram of normalized variance
                title = "{0} on {1} motion variance histogram".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(111)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('variance (m)')
                if np.min(x_motion_normalized) < np.max(x_motion_normalized):
                    ax.hist(x_motion_normalized, bins=100, color='red', alpha=0.3, label='x axis')
                if np.min(y_motion_normalized) < np.max(y_motion_normalized):
                    ax.hist(y_motion_normalized, bins=100, color='green', alpha=0.3, label='y axis')
                if np.min(z_motion_normalized) < np.max(z_motion_normalized):
                    ax.hist(z_motion_normalized, bins=100, color='blue', alpha=0.3, label='z axis')

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.95, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

                # Plot computed motion variance vs time as a heatmap
                title = "{0} on {1} motion estimate variation heatmap".format(system_name, dataset_name)
                figure = pyplot.figure(figsize=(30, 10), dpi=80)
                figure.suptitle(title)

                ax = figure.add_subplot(131)
                ax.set_title('x')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('x variance (m)')
                heatmap, xedges, yedges = np.histogram2d(motion_times, x_motion_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(132)
                ax.set_title('y')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('y variance (m)')
                heatmap, xedges, yedges = np.histogram2d(motion_times, y_motion_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                ax = figure.add_subplot(133)
                ax.set_title('z')
                ax.set_xlabel('time (s)')
                ax.set_ylabel('z variance (m)')
                heatmap, xedges, yedges = np.histogram2d(motion_times, z_motion_variance, bins=300)
                ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower',
                          aspect='auto', cmap=pyplot.get_cmap('inferno_r'))

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.90, right=0.99)

                figure.savefig(os.path.join(save_path, title + '.png'))
                pyplot.close(figure)

        # Show all the graphs remaining
        pyplot.show()

    def _plot_estimate_variance(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        save_path = os.path.join('figures', type(self).__name__, 'orbslam variance heatmaps')
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
                            gt_scale = th.find_trajectory_scale(ground_truth_traj)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = th.rescale_trajectory(traj, gt_scale)
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
                    logging.getLogger(__name__).warning("    axis min {0} is more than 3 standard deviations away, clamping".format(ax_min))
                    ax_min = -1 * ax_std
                if ax_max < ax_std:
                    logging.getLogger(__name__).warning("    axis max {0} is more than 3 standard deviations away, clamping".format(ax_max))
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

    def _plot_error_vs_motion(self, db_client: arvet.database.client.DatabaseClient):
        import matplotlib.pyplot as pyplot
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        save_path = os.path.join('figures', type(self).__name__, 'orbslam error vs motion heatmaps')
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
                            gt_scale = th.find_trajectory_scale(ground_truth_traj)
                        traj = trial_result.get_computed_camera_poses()

                        # Normalize monocular trajectories
                        if 'mono' in system_name.lower():
                            if gt_scale is not None:
                                traj = th.rescale_trajectory(traj, gt_scale)
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

                    mean_orientation = quat_mean([computed_trajectories[idx][timestamps[idx][time]].rotation_quat(True)
                                                  for idx in range(len(computed_trajectories))
                                                  if time in timestamps[idx]])

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
                                quat_diff(mean_orientation,
                                          computed_trajectories[idx][timestamps[idx][time]].rotation_quat(True))**2
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


def quat_mean(quaternions):
    """
    Find the mean of a bunch of quaternions
    :param quaternions:
    :return:
    """
    if len(quaternions) <= 0:
        return np.nan
    q_mat = np.asarray(quaternions)
    product = np.dot(q_mat.T, q_mat)
    evals, evecs = np.linalg.eig(product)
    best = -1
    result = None
    for idx in range(len(evals)):
        if evals[idx] > best:
            best = evals[idx]
            result = evecs[idx]
    return result


def quat_diff(q1, q2):
    """
    Find the angle between two quaternions
    Basically, we compose them, and derive the angle from the composition
    :param q1:
    :param q2:
    :return:
    """
    z0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    return 2 * np.arccos(min(1, max(-1, z0)))
