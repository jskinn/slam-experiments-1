import arvet.create_experiment
import simple_motion_experiment
import real_world_experiment


def main():
    arvet.create_experiment.create_experiment(simple_motion_experiment.SimpleMotionExperiment)
    arvet.create_experiment.create_experiment(real_world_experiment.RealWorldExperiment)


if __name__ == '__main__':
    main()
