import arvet.create_experiment
import simple_motion_experiment
import real_world_experiment
import orbslam_kitti_verify


def main():
    arvet.create_experiment.create_experiment(simple_motion_experiment.SimpleMotionExperiment)
    arvet.create_experiment.create_experiment(real_world_experiment.RealWorldExperiment)
    arvet.create_experiment.create_experiment(orbslam_kitti_verify.OrbslamKITTIVerify)


if __name__ == '__main__':
    main()
