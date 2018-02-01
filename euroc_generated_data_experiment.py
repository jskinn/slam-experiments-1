# Copyright (c) 2017, John Skinner
import os.path
import arvet.util.dict_utils as du
import arvet.util.trajectory_helpers as traj_help
import arvet.util.unreal_transform as uetf
import arvet.database.client
import arvet.config.path_manager
import arvet.metadata.image_metadata as imeta
import arvet.batch_analysis.task_manager
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2
import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe
import arvet_slam.benchmarks.ate.absolute_trajectory_error as ate
import arvet_slam.benchmarks.trajectory_drift.trajectory_drift as traj_drift
import arvet_slam.benchmarks.tracking.tracking_benchmark as tracking_benchmark
import base_generated_data_experiment
import data_helpers


class EurocGeneratedDataExperiment(base_generated_data_experiment.GeneratedDataExperiment):

    def __init__(self, systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmarks=None,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param simulators:
        :param trajectory_groups:
        :param benchmarks:
        :param trial_map:
        :param result_map:
        :param enabled:
        :param id_:
        """
        super().__init__(systems=systems, simulators=simulators, trajectory_groups=trajectory_groups,
                         benchmarks=benchmarks, id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)

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
        super().do_imports(task_manager, path_manager, db_client)

        # --------- SIMULATORS -----------
        # Add simulators explicitly, they have different metadata, so we can't just search
        for exe, world_name, environment_type, light_level, time_of_day in [
            (
                    'simulators/AIUE_V01_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_002/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_002', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_003/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_003', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V01_004/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_004', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            ), (
                    'simulators/AIUE_V02_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V02_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
            )
        ]:
            self.import_simulator(
                executable_path=exe,
                world_name=world_name,
                environment_type=environment_type,
                light_level=light_level,
                time_of_day=time_of_day,
                db_client=db_client
            )

        # --------- REAL WORLD DATASETS -----------

        # Import EuRoC datasets with lists of trajectory start points for each simulator
        for name, path, mappings in [
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy'), get_MH_01_easy()),
            ('EuRoC MH_02_easy', os.path.join('datasets', 'EuRoC', 'MH_02_easy'), get_MH_02_easy()),
            ('EuRoC MH_03_medium', os.path.join('datasets', 'EuRoC', 'MH_03_medium'), get_MH_03_medium()),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult'), get_MH_04_difficult()),
            ('EuRoC MH_05_difficult', os.path.join('datasets', 'EuRoC', 'MH_05_difficult'), get_MH_05_difficult()),
            ('EuRoC V1_01_easy', os.path.join('datasets', 'EuRoC', 'V1_01_easy'), get_V1_01_easy()),
            ('EuRoC V1_02_medium', os.path.join('datasets', 'EuRoC', 'V1_02_medium'), get_V1_02_medium()),
            ('EuRoC V1_03_difficult', os.path.join('datasets', 'EuRoC', 'V1_03_difficult'), get_V1_03_difficult()),
            ('EuRoC V2_01_easy', os.path.join('datasets', 'EuRoC', 'V2_01_easy'), get_V2_01_easy()),
            ('EuRoC V2_02_medium', os.path.join('datasets', 'EuRoC', 'V2_02_medium'), get_V2_02_medium()),
            ('EuRoC V2_03_difficult', os.path.join('datasets', 'EuRoC', 'V2_03_difficult'), get_V2_03_difficult())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                name=name,
                mappings=mappings,
                task_manager=task_manager,
                path_manager=path_manager,
                db_client=db_client,
            )

        # --------- SYSTEMS -----------
        # LibVisO2
        self.import_system(
            name='LibVisO',
            system=libviso2.LibVisOSystem(),
            db_client=db_client
        )

        # ORBSLAM2 - Create 3 variants, with different procesing modes
        vocab_path = os.path.join('systems', 'slam', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.RGBD, orbslam2.SensorMode.MONOCULAR}:
            self.import_system(
                name='ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()),
                system=orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
                    settings={'ORBextractor': {'nFeatures': 1500}}
                ),
                db_client=db_client
            )

        # --------- BENCHMARKS -----------
        # Create and store the benchmarks for camera trajectories
        # Just using the default settings for now

        self.import_benchmark(
            name='Relative Pose Error',
            db_client=db_client,
            benchmark=rpe.BenchmarkRPE(
                max_pairs=10000,
                fixed_delta=False,
                delta=1.0,
                delta_unit='s',
                offset=0,
                scale_=1
            )
        )
        self.import_benchmark(
            name='Absolute Trajectory Error',
            db_client=db_client,
            benchmark=ate.BenchmarkATE(
                offset=0,
                max_difference=0.2,
                scale=1
            )
        )
        self.import_benchmark(
            name='Trajectory Drift',
            db_client=db_client,
            benchmark=traj_drift.BenchmarkTrajectoryDrift(
                segment_lengths=[100, 200, 300, 400, 500, 600, 700, 800],
                step_size=10
            )
        )
        self.import_benchmark(
            name='Tracking Statistics',
            db_client=db_client,
            benchmark=tracking_benchmark.TrackingBenchmark(initializing_is_lost=True)
        )

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Plot the results for this experiment.
        Don't use this, use export data to dump the results to json
        :param db_client:
        :return:
        """
        pass

    def export_data(self, db_client: arvet.database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to
        :param db_client:
        :return:
        """
        # Save trajectory files for import into unreal
        for trajectory_group in self._trajectory_groups.values():
            data_helpers.dump_ue4_trajectory(
                name=trajectory_group.name,
                trajectory=traj_help.get_trajectory_for_image_source(db_client, trajectory_group.reference_dataset)
            )

        # Group and print the trajectories for graphing
        systems = du.defaults({'LIBVISO 2': self._libviso_system}, self._orbslam_systems)
        for trajectory_group in self._trajectory_groups.values():

            # Collect the trial results for each image source in this group
            trial_results = {}
            for system_name, system_id in systems.items():
                for dataset_name, dataset_id in trajectory_group.datasets.items():
                    trial_result_list = self.get_trial_result(system_id, dataset_id)
                    for idx, trial_result_id in enumerate(trial_result_list):
                        label = "{0} on {1} repeat {2}".format(system_name, dataset_name, idx)
                        trial_results[label] = trial_result_id
            data_helpers.export_trajectory_as_json(
                trial_results, "Generated Data " + trajectory_group.name, db_client
            )


def get_MH_01_easy():
    return [
        ('AIUE_V01_002', uetf.create_serialized((2240, 415, 75), (0, 0, 160))),
        ('AIUE_V01_004', uetf.create_serialized((485, -45, 145), (0, 0, 145))),
        ('AIUE_V01_005', uetf.create_serialized((-925, -1135, 120), (0, 0, 0))),
    ]


def get_MH_02_easy():
    return [
        ('AIUE_V01_002', uetf.create_serialized((2240, 415, 75), (0, 0, 156))),
        ('AIUE_V01_004', uetf.create_serialized((225, 0, 145), (0, 0, 140))),
        ('AIUE_V01_005', uetf.create_serialized((770, 180, 60), (0, 0, 180))),
    ]


def get_MH_03_medium():
    return [
        ('AIUE_V01_005', uetf.create_serialized((-490, -450, 275), (0, 0, -90))),
    ]


def get_MH_04_difficult():
    return []


def get_MH_05_difficult():
    return [
        ('AIUE_V01_005', uetf.create_serialized((210, -770, -60), (0, 0, 140))),
    ]


def get_V1_01_easy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-300, 400, 170), (0, 0, -95))),
        ('AIUE_V01_001', uetf.create_serialized((-260, -490, 170), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((-765, 95, 280), (0, 0, 135))),
        ('AIUE_V01_002', uetf.create_serialized((-40, -175, 135), (0, 0, -75))),
        ('AIUE_V01_003', uetf.create_serialized((-115, 520, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((385, 0, 425), (0, 0, -105))),
        ('AIUE_V01_005', uetf.create_serialized((-550, -85, 125), (0, 0, 135))),
    ]


def get_V1_02_medium():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-90, 495, 115), (0, 0, 90))),
        ('AIUE_V01_001', uetf.create_serialized((-310, -460, 115), (0, 0, 10))),
        ('AIUE_V01_002', uetf.create_serialized((207.5, 310, 95), (0, 0, -150))),
        ('AIUE_V01_002', uetf.create_serialized((-7.5, -190, 105), (0, 0, -75))),
        ('AIUE_V01_003', uetf.create_serialized((-115, 520, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((405, 10, 105), (0, 0, -90))),
        ('AIUE_V01_005', uetf.create_serialized((-560, -1305, 480), (0, 0, -175))),
    ]


def get_V1_03_difficult():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-520, -120, 115), (0, 0, 180))),
        ('AIUE_V01_002', uetf.create_serialized((-522.5, 60, 260), (0, 0, -135))),
        ('AIUE_V01_003', uetf.create_serialized((-235, 255, 135), (0, 0, 155))),
        ('AIUE_V01_004', uetf.create_serialized((-230, -135, 435), (0, 0, 0))),
        ('AIUE_V01_005', uetf.create_serialized((-70, -495, 120), (0, 0, 70))),
    ]


def get_V2_01_easy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-185, 400, 160), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((367.5, -95, 120), (0, 0, -105))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 135), (0, 0, -70))),
        ('AIUE_V01_004', uetf.create_serialized((-350, -35, 205), (0, 0, 115))),
        ('AIUE_V01_005', uetf.create_serialized((-490, -450, 100), (0, 0, -90))),
    ]


def get_V2_02_medium():
    return [
        ('AIUE_V01_002', uetf.create_serialized((-355, -10, 205), (0, 0, -170))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 170), (0, 0, -90))),
        ('AIUE_V01_004', uetf.create_serialized((405, 10, 105), (0, 0, -90))),
        ('AIUE_V01_005', uetf.create_serialized((-540, -290, 195), (0, 0, 150))),
    ]


def get_V2_03_difficult():
    return [
        ('AIUE_V01_002', uetf.create_serialized((-355, -10, 205), (0, 0, -170))),
        ('AIUE_V01_003', uetf.create_serialized((-345, 560, 160), (0, 0, -110))),
        ('AIUE_V01_004', uetf.create_serialized((-695, -30, 430), (0, 0, -5))),
        ('AIUE_V01_005', uetf.create_serialized((-540, -290, 195), (0, 0, 150))),
    ]