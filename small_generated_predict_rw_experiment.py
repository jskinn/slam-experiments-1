# Copyright (c) 2017, John Skinner
import os.path
import typing
import numpy as np

import arvet.database.client
import arvet.config.path_manager
import arvet.metadata.image_metadata as imeta
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager

import arvet_slam.systems.slam.orbslam2 as orbslam2
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as libviso2

import euroc_origins
import tum_origins
import base_generated_predict_rw_experiment as bgprwe


class SmallGeneratedPredictRealWorldExperiment(bgprwe.BaseGeneratedPredictRealWorldExperiment):

    def __init__(self, systems=None,
                 simulators=None,
                 trajectory_groups=None,
                 benchmarks=None,
                 trial_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems:
        :param simulators:
        :param trajectory_groups:
        :param benchmarks:
        :param trial_map:
        :param enabled:
        :param id_:
        """
        super().__init__(
            systems=systems,
            simulators=simulators,
            trajectory_groups=trajectory_groups,
            benchmarks=benchmarks,
            id_=id_,
            trial_map=trial_map,
            enabled=enabled
        )

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
                    'simulators/AIUE_V01_005/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    'AIUE_V01_005', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    imeta.TimeOfDay.DAY
                    # ), (
                    #         'simulators/AIUE_V02_001/LinuxNoEditor/tempTest/Binaries/Linux/tempTest',
                    #         'AIUE_V02_001', imeta.EnvironmentType.INDOOR, imeta.LightingLevel.WELL_LIT,
                    #         imeta.TimeOfDay.DAY
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
            ('EuRoC MH_01_easy', os.path.join('datasets', 'EuRoC', 'MH_01_easy'), euroc_origins.get_MH_01_easy()),
            ('EuRoC MH_02_easy', os.path.join('datasets', 'EuRoC', 'MH_02_easy'), euroc_origins.get_MH_02_easy()),
            ('EuRoC MH_03_medium', os.path.join('datasets', 'EuRoC', 'MH_03_medium'),
             euroc_origins.get_MH_03_medium()),
            ('EuRoC MH_04_difficult', os.path.join('datasets', 'EuRoC', 'MH_04_difficult'),
             euroc_origins.get_MH_04_difficult()),
            ('EuRoC MH_05_difficult', os.path.join('datasets', 'EuRoC', 'MH_05_difficult'),
             euroc_origins.get_MH_05_difficult()),
            ('EuRoC V1_01_easy', os.path.join('datasets', 'EuRoC', 'V1_01_easy'), euroc_origins.get_V1_01_easy()),
            ('EuRoC V1_02_medium', os.path.join('datasets', 'EuRoC', 'V1_02_medium'),
             euroc_origins.get_V1_02_medium()),
            ('EuRoC V1_03_difficult', os.path.join('datasets', 'EuRoC', 'V1_03_difficult'),
             euroc_origins.get_V1_03_difficult()),
            ('EuRoC V2_01_easy', os.path.join('datasets', 'EuRoC', 'V2_01_easy'), euroc_origins.get_V2_01_easy()),
            ('EuRoC V2_02_medium', os.path.join('datasets', 'EuRoC', 'V2_02_medium'),
             euroc_origins.get_V2_02_medium()),
            ('EuRoC V2_03_difficult', os.path.join('datasets', 'EuRoC', 'V2_03_difficult'),
             euroc_origins.get_V2_03_difficult())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.euroc.euroc_loader',
                path=path,
                name=name,
                mappings=mappings,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # Import TUM datasets with lists of trajectory start points for each simulator
        for folder, mappings in [
            ('rgbd_dataset_freiburg1_360', tum_origins.get_frieburg1_360()),
            ('rgbd_dataset_frieburg1_rpy', tum_origins.get_frieburg1_rpy()),
            ('rgbd_dataset_frieburg1_xyz', tum_origins.get_frieburg1_xyz()),
            ('rgbd_dataset_frieburg2_desk', tum_origins.get_frieburg2_desk()),
            ('rgbd_dataset_frieburg2_rpy', tum_origins.get_frieburg2_rpy()),
            ('rgbd_dataset_frieburg2_xyz', tum_origins.get_frieburg2_xyz()),
            ('rgbd_dataset_frieburg3_structure_texture_far', tum_origins.get_frieburg3_structure_texture_far()),
            ('rgbd_dataset_frieburg3_walking_xyz', tum_origins.get_frieburg3_walking_xyz())
        ]:
            self.import_dataset(
                module_name='arvet_slam.dataset.tum.tum_loader',
                path=os.path.join('datasets', 'TUM', folder),
                name="TUM {0}".format(folder),
                mappings=mappings,
                task_manager=task_manager,
                path_manager=path_manager
            )

        # --------- SYSTEMS -----------
        # LibVisO2
        self.import_system(
            name='LibVisO',
            system=libviso2.LibVisOSystem(),
            db_client=db_client
        )

        # ORBSLAM2 - Create 3 variants; stereo, mono, and rgbd
        # These datasets don't have
        vocab_path = os.path.join('systems', 'ORBSLAM2', 'ORBvoc.txt')
        for sensor_mode in {orbslam2.SensorMode.STEREO, orbslam2.SensorMode.MONOCULAR, orbslam2.SensorMode.RGBD}:
            self.import_system(
                name='ORBSLAM2 {mode}'.format(mode=sensor_mode.name.lower()),
                system=orbslam2.ORBSLAM2(
                    vocabulary_file=vocab_path,
                    mode=sensor_mode,
                    settings={'ORBextractor': {'nFeatures': 1500}}
                ),
                db_client=db_client
            )

        # Do the superclass imports
        super().do_imports(task_manager, path_manager, db_client)

    def get_quality_variations(self) -> typing.List[typing.Tuple[str, dict]]:
        """
        This experiment only generates max and min quality data
        :return:
        """
        return [('max quality', {
        }), ('min quality', {
            # absolute minimum visual quality, can still reduce FOV and resolution
            'lit_mode': False,
            'depth_of_field_enabled': False,
            'texture_mipmap_bias': 8,
            'normal_maps_enabled': False,
            'roughness_enabled': False,
            'geometry_decimation': 4,
        })]

    def perform_analysis(self, db_client: arvet.database.client.DatabaseClient):
        """
        Use the results from generated data to try and predict the performance on some selected real world data
        We're going to do this system by system because different types of system use different datasets.
        :param db_client:
        :return:
        """
        random_state = np.random.RandomState(48625)
        results_cache = {}
        euroc_sets = ['EuRoC MH_01_easy', 'EuRoC MH_02_easy', 'EuRoC MH_03_medium', 'EuRoC MH_04_difficult',
                      'EuRoC MH_05_difficult', 'EuRoC V1_01_easy', 'EuRoC V1_02_medium', 'EuRoC V1_03_difficult',
                      'EuRoC V2_01_easy', 'EuRoC V2_02_medium', 'EuRoC V2_03_difficult']
        tum_sets = ['TUM rgbd_dataset_freiburg1_360', 'TUM rgbd_dataset_frieburg1_rpy',
                    'TUM rgbd_dataset_frieburg1_xyz', 'TUM rgbd_dataset_frieburg2_desk',
                    'TUM rgbd_dataset_frieburg2_rpy', 'TUM rgbd_dataset_frieburg2_xyz',
                    'TUM rgbd_dataset_frieburg3_structure_texture_far', 'TUM rgbd_dataset_frieburg3_walking_xyz']

        # --------- DISTRIBUTIONS -----------
        # Plot the distributions for the different errors
        for system_name in self.systems.keys():
            self.analyse_distributions(
                system_name=system_name,
                output_folder=os.path.join(type(self).get_output_folder(), system_name, 'distributions'),
                db_client=db_client,
                results_cache=results_cache
            )
