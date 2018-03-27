import bson
import typing
import arvet.util.database_helpers as dh
import arvet.util.dict_utils as du
import arvet.database.client
import arvet.core.sequence_type as sequence_type
import arvet.core.image_collection
import arvet.batch_analysis.task_manager
import arvet.simulation.controllers.trajectory_follow_controller as follow_cont


class TrajectoryGroup:
    """
    A Trajectory Group is a helper structure to manage image datasets grouped by camera trajectory.
    In this class of experiment, it is created from a single reference real-world dataset,
    and produces many synthetic datasets with the same camera trajectory.

    For convenience, it serializes and deserialzes as a group.
    """

    def __init__(self, name: str, reference_id: bson.ObjectId, mappings: typing.List[typing.Tuple[str, dict]],
                 baseline_configuration: dict = None,
                 controller_id: bson.ObjectId = None, generated_datasets: dict = None):
        self.name = name
        self.reference_dataset = reference_id
        self.mappings = mappings
        self.baseline_configuration = baseline_configuration

        self.follow_controller_id = controller_id
        self.generated_datasets = generated_datasets if generated_datasets is not None else {}

    def get_all_dataset_ids(self) -> set:
        """
        Get all the datasets in this group, as a set
        :return:
        """
        return {self.reference_dataset} | set(
            dataset_id
            for quality_map in self.generated_datasets.values()
            for dataset_id in quality_map.values()
        )

    def get_datasets_for_sim(self, sim_name: str) -> typing.Mapping[str, bson.ObjectId]:
        """
        Get all the generated datasets from a particular simulator
        :param sim_name:
        :return:
        """
        return self.generated_datasets[sim_name]

    def get_datasets_for_quality(self, quality_name: str) -> typing.Mapping[str, bson.ObjectId]:
        """
        Get all the generated datasets at a particular quality
        :param quality_name:
        :return:
        """
        return {
            sim_name: quality_map[quality_name]
            for sim_name, quality_map in self.generated_datasets.items()
            if quality_name in quality_map
        }

    def schedule_generation(self, simulators: typing.Mapping[str, bson.ObjectId],
                            quality_variations: typing.List[typing.Tuple[str, dict]],
                            task_manager: arvet.batch_analysis.task_manager.TaskManager,
                            db_client: arvet.database.client.DatabaseClient) -> bool:
        """
        Do imports and dataset generation for this trajectory group.
        Will create a controller, and then generate reduced quality synthetic datasets.
        :param simulators: A Map of simulators, indexed by name
        :param quality_variations: A list of names and quality variations
        :param task_manager:
        :param db_client:
        :return: True if part of the group has changed, and it needs to be re-saved
        """
        changed = False
        # First, make a follow controller for the base dataset if we don't have one.
        # This will be used to generate reduced-quality datasets following the same trajectory
        # as the root dataset
        if self.follow_controller_id is None:
            self.follow_controller_id = follow_cont.create_follow_controller(
                db_client, self.reference_dataset, sequence_type=sequence_type.ImageSequenceType.SEQUENTIAL)
            changed = True

        # Next, if we haven't already, compute baseline configuration from the reference dataset
        if self.baseline_configuration is None or len(self.baseline_configuration) == 0:
            reference_dataset = dh.load_object(db_client, db_client.image_source_collection, self.reference_dataset)
            if isinstance(reference_dataset, arvet.core.image_collection.ImageCollection):
                intrinsics = reference_dataset.get_camera_intrinsics()
                self.baseline_configuration = {
                        # Simulation execution config
                        'stereo_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_stereo_available else 0,
                        'provide_rgb': True,
                        'provide_ground_truth_depth': False,    # We don't care about this
                        'provide_labels': reference_dataset.is_labels_available,
                        'provide_world_normals': reference_dataset.is_normals_available,

                        # Depth settings
                        'provide_depth': reference_dataset.is_depth_available,
                        'depth_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_depth_available else 0,
                        'projector_offset': reference_dataset.get_stereo_baseline() \
                        if reference_dataset.is_depth_available else 0,

                        # Simulator camera settings, be similar to the reference dataset
                        'resolution': {'width': intrinsics.width, 'height': intrinsics.height},
                        'fov': max(intrinsics.horizontal_fov, intrinsics.vertical_fov),
                        'depth_of_field_enabled': False,
                        'focus_distance': None,
                        'aperture': 2.2,

                        # Quality settings - Maximum quality
                        'lit_mode': True,
                        'texture_mipmap_bias': 0,
                        'normal_maps_enabled': True,
                        'roughness_enabled': True,
                        'geometry_decimation': 0,
                        'depth_noise_quality': 1,

                        # Simulation server config
                        'host': 'localhost',
                        'port': 9000,
                    }
                changed = True

        # Then, for each simulator listed for this trajectory group
        origin_counts = {}
        for sim_name, origin in self.mappings:
            # Count how many times each simulator is used, so we can assign a unique world name to each start point
            if sim_name not in origin_counts:
                origin_counts[sim_name] = 1
            else:
                origin_counts[sim_name] += 1

            # Schedule generation of quality variations that don't exist yet
            if sim_name in simulators:
                # For every quality variation
                for quality_name, config in quality_variations:
                    generate_dataset_task = task_manager.get_generate_dataset_task(
                        controller_id=self.follow_controller_id,
                        simulator_id=simulators[sim_name],
                        simulator_config=du.defaults({'origin': origin}, config, self.baseline_configuration),
                        num_cpus=1,
                        num_gpus=0,
                        memory_requirements='3GB',
                        expected_duration='4:00:00'
                    )
                    if generate_dataset_task.is_finished:
                        world_name = "{0} {1}".format(sim_name, origin_counts[sim_name])
                        if world_name not in self.generated_datasets:
                            self.generated_datasets[world_name] = {}
                        self.generated_datasets[world_name][quality_name] = generate_dataset_task.result
                        changed = True
                    else:
                        task_manager.do_task(generate_dataset_task)
        return changed

    def serialize(self) -> dict:
        serialized = {
            'name': self.name,
            'reference_id': self.reference_dataset,
            'mappings': self.mappings,
            'baseline_configuration': self.baseline_configuration,
            'controller_id': self.follow_controller_id,
            'generated_datasets': self.generated_datasets
        }
        dh.add_schema_version(serialized, 'experiments:visual_slam:TrajectoryGroup', 1)
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation: dict, db_client: arvet.database.client.DatabaseClient) \
            -> 'TrajectoryGroup':
        update_trajectory_group_schema(serialized_representation, db_client)
        return cls(
            name=serialized_representation['name'],
            reference_id=serialized_representation['reference_id'],
            mappings=serialized_representation['mappings'],
            baseline_configuration=serialized_representation['baseline_configuration'],
            controller_id=serialized_representation['controller_id'],
            generated_datasets=serialized_representation['generated_datasets']
        )


def update_trajectory_group_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # version = dh.get_schema_version(serialized, 'experiments:visual_slam:TrajectoryGroup')

    # Remove invalid ids
    if 'reference_id' in serialized and \
            not dh.check_reference_is_valid(db_client.image_source_collection, serialized['reference_id']):
        del serialized['reference_id']
    if 'controller_id' in serialized and \
            not dh.check_reference_is_valid(db_client.image_source_collection, serialized['controller_id']):
        del serialized['controller_id']
    if 'generated_datasets' in serialized:

        # Remove invalid dataset ids in each map
        for quality_map in serialized['generated_datasets'].values():
            invalid_values = dh.check_many_references(db_client.image_source_collection, quality_map.values())
            keys = list(quality_map.keys())
            for key in keys:
                if quality_map[key] in invalid_values:
                    del quality_map[key]

        # Remove sim names with no results
        keys = [sim_name for sim_name, quality_map in serialized['generated_datasets'].items() if len(quality_map) <= 0]
        for sim_name in keys:
            del serialized['generated_datasets'][sim_name]
