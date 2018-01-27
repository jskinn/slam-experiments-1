# Copyright (c) 2017, John Skinner
import os
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
import data_helpers
import base_generated_data_experiment


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

        # Import TUM datasets with lists of trajectory start points for each simulator
        for folder, mappings in [
            ('rgbd_dataset_freiburg1_360', get_frieburg1_360()),
            ('rgbd_dataset_frieburg1_rpy', get_frieburg1_rpy()),
            ('rgbd_dataset_frieburg1_xyz', get_frieburg1_xyz()),
            ('rgbd_dataset_frieburg2_desk', get_frieburg2_desk()),
            ('rgbd_dataset_frieburg2_rpy', get_frieburg2_rpy()),
            ('rgbd_dataset_frieburg2_xyz', get_frieburg2_xyz()),
            ('rgbd_dataset_frieburg3_structure_texture_far', get_frieburg3_structure_texture_far()),
            ('rgbd_dataset_frieburg3_walking_xyz', get_frieburg3_walking_xyz())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.tum.tum_loader',
                path=os.path.join('datasets', 'TUM', folder),
                name="TUM {0}".format(folder),
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


def get_frieburg1_360():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-285, 420, 155), (0, 0, 135))),
        ('AIUE_V01_001', uetf.create_serialized((-370, -345, 155), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]

def get_frieburg1_rpy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-325, 200, 105), (0, 0, 180))),
        ('AIUE_V01_001', uetf.create_serialized((-635, 370, 105), (0, 0, 40))),
        ('AIUE_V01_001', uetf.create_serialized((615, -310, 135), (0, 0, 170))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]

def get_frieburg1_xyz():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-325, 200, 105), (0, 0, 180))),
        ('AIUE_V01_001', uetf.create_serialized((-635, 370, 105), (0, 0, 40))),
        ('AIUE_V01_001', uetf.create_serialized((615, -310, 135), (0, 0, 170))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]

def get_frieburg2_desk():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-250, 515, 80), (0, 0, 0))),
        ('AIUE_V01_001', uetf.create_serialized((-235, -255, 80), (0, 0, 180))),
        ('AIUE_V01_002', uetf.create_serialized((-50, 175, 90), (0, 0, 175))),
        ('AIUE_V01_003', uetf.create_serialized((-20, 70, 105), (0, 0, 180))),

        ('AIUE_V01_005', uetf.create_serialized((-775, -475, 15), (0, 0, 0))),
    ]

def get_frieburg2_rpy():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-325, 200, 105), (0, 0, 180))),
        ('AIUE_V01_001', uetf.create_serialized((-635, 370, 105), (0, 0, 40))),
        ('AIUE_V01_001', uetf.create_serialized((615, -310, 135), (0, 0, 170))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]

def get_frieburg2_xyz():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-325, 200, 105), (0, 0, 180))),
        ('AIUE_V01_001', uetf.create_serialized((-635, 370, 105), (0, 0, 40))),
        ('AIUE_V01_001', uetf.create_serialized((615, -310, 135), (0, 0, 170))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]

def get_frieburg3_structure_texture_far():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-400, -490, 100), (0, 0, 125))),
        ('AIUE_V01_001', uetf.create_serialized((-220, 640, 100), (0, 0, 0))),
        ('AIUE_V01_002', uetf.create_serialized((-115, -315, 95), (0, 0, 15))),
        ('AIUE_V01_003', uetf.create_serialized((-20, 70, 105), (0, 0, 180))),

        ('AIUE_V01_005', uetf.create_serialized((430, -250, 145), (0, 0, -165))),
    ]

def get_frieburg3_walking_xyz():
    return [
        ('AIUE_V01_001', uetf.create_serialized((-325, 200, 105), (0, 0, 180))),
        ('AIUE_V01_001', uetf.create_serialized((-635, 370, 105), (0, 0, 40))),
        ('AIUE_V01_001', uetf.create_serialized((615, -310, 135), (0, 0, 170))),
        ('AIUE_V01_002', uetf.create_serialized((-1185, 425, -15), (0, 0, -125))),
        ('AIUE_V01_002', uetf.create_serialized((-990, -135, 315), (0, 0, 25))),
        ('AIUE_V01_003', uetf.create_serialized((-105, -335, 135), (0, 0, 110))),
        ('AIUE_V01_003', uetf.create_serialized((-70, 635, 135), (0, 0, -140))),

        ('AIUE_V01_005', uetf.create_serialized((-95, -695, 65), (0, 0, 150))),
        ('AIUE_V01_005', uetf.create_serialized((-795, -1490, 490), (0, 0, 180))),
    ]
