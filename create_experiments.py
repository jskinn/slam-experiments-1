import arvet.create_experiment
import real_world_experiment
import orbslam_kitti_verify
import orbslam_euroc_verify
import orbslam_tum_verify
import libviso_euroc_verify
import libviso_kitti_verify
import orbslam_consistency_experiment
import euroc_generated_data_experiment
import tum_generated_data_experiment


def main():
    # Simple case verification experiments
    # arvet.create_experiment.create_experiment(simple_motion_experiment.SimpleMotionExperiment)
    arvet.create_experiment.create_experiment(real_world_experiment.RealWorldExperiment)

    # System verification experiments
    arvet.create_experiment.create_experiment(orbslam_kitti_verify.OrbslamKITTIVerify)
    arvet.create_experiment.create_experiment(orbslam_euroc_verify.OrbslamEuRoCVerify)
    arvet.create_experiment.create_experiment(orbslam_tum_verify.OrbslamTUMVerify)
    arvet.create_experiment.create_experiment(libviso_euroc_verify.LibVisOEuRoCVerify)
    arvet.create_experiment.create_experiment(libviso_kitti_verify.LibVisOKITTIVerify)
    arvet.create_experiment.create_experiment(orbslam_consistency_experiment.OrbslamConsistencyExperiment)

    # Synthetic data experiments
    arvet.create_experiment.create_experiment(euroc_generated_data_experiment.EurocGeneratedDataExperiment)
    arvet.create_experiment.create_experiment(tum_generated_data_experiment.TUMGeneratedDataExperiment)


if __name__ == '__main__':
    main()
