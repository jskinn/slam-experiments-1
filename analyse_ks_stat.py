import json
import numpy as np
from scipy.stats import ks_2samp


def real_to_real_distance(system_name, error_data):
    all_data = []
    for sequence_name in sorted(error_data.keys()):
        if 'other real' in error_data[sequence_name]:
            all_data += error_data[sequence_name]['other real']
    other_min = np.min(all_data)
    other_mean = np.mean(all_data)
    other_median = np.median(all_data)
    other_max = np.max(all_data)
    print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(
        #sequence_name.replace('_', '\\_'),
        system_name,
        other_min,
        other_mean,
        #other_median,
        other_max
    ))
    print("% {0} distances".format(len(all_data)))


def sim_to_real_distance(system_name, error_data):
    all_data = []
    for sequence_name in sorted(error_data.keys()):
        if 'max quality' in error_data[sequence_name]:
            all_data += error_data[sequence_name]['max quality']
    sim_min = np.min(all_data)
    sim_mean = np.mean(all_data)
    sim_median = np.median(all_data)
    sim_max = np.max(all_data)
    print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(
        #sequence_name.replace('_', '\\_'),
        system_name,
        sim_min,
        sim_mean,
        #sim_median,
        sim_max
    ))
    print("% {0} distances".format(len(all_data)))


def min_to_real_distance(system_name, error_data):
    all_data = []
    for sequence_name in sorted(error_data.keys()):
        if 'min quality' in error_data[sequence_name]:
            all_data += error_data[sequence_name]['min quality']
    sim_min = np.min(all_data)
    sim_mean = np.mean(all_data)
    sim_median = np.median(all_data)
    sim_max = np.max(all_data)
    print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(
        #sequence_name.replace('_', '\\_'),
        system_name,
        sim_min,
        sim_mean,
        #sim_median,
        sim_max
    ))
    print("% {0} distances".format(len(all_data)))


def self_to_self_distance(system_name, error_data):
    all_data = []
    for sequence_name in sorted(error_data.keys()):
        if 'self' in error_data[sequence_name]:
            all_data += error_data[sequence_name]['self']
    self_min = np.min(all_data)
    self_mean = np.mean(all_data)
    self_median = np.median(all_data)
    self_max = np.max(all_data)
    print("{0} & {1:.2f} & {2:.2f} & {3:.2f} \\\\".format(
        #sequence_name.replace('_', '\\_'),
        system_name,
        self_min,
        self_mean,
        #self_median,
        self_max
    ))
    print("% {0} distances".format(len(all_data)))


def meta_ks_stat(system_name, error_data):
    real_data = []
    sim_data = []
    for sequence_name in sorted(error_data.keys()):
        if 'other real' in error_data[sequence_name]:
            real_data += error_data[sequence_name]['other real']
        if 'max quality' in error_data[sequence_name]:
            sim_data += error_data[sequence_name]['max quality']
    ks_stat, _ = ks_2samp(real_data, sim_data)
    print("{0} meta-ks stat: {1}".format(system_name, ks_stat))


def analyse_error(system_name, error_data):
    printed_name = False
    for sequence_name in sorted(error_data.keys()):
        if 'self' in error_data[sequence_name] and 'max quality' in error_data[sequence_name]:
            self_min = np.min(error_data[sequence_name]['self'])
            self_max = np.max(error_data[sequence_name]['self'])
            self_mean = np.mean(error_data[sequence_name]['self'])
            self_median = np.median(error_data[sequence_name]['self'])
            num_scores = len(error_data[sequence_name]['max quality'])
            frac_max = sum(1 for d in error_data[sequence_name]['max quality'] if d < self_max) / num_scores
            frac_min = sum(1 for d in error_data[sequence_name]['max quality'] if d < self_min) / num_scores
            frac_mean = sum(1 for d in error_data[sequence_name]['max quality'] if d < self_mean) / num_scores
            frac_median = sum(1 for d in error_data[sequence_name]['max quality'] if d < self_median) / num_scores
            if frac_max > 0:
                if not printed_name:
                    print("\\multicolumn{{4}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
                    printed_name = True
                print("{0} & {1:.2f} & {2:.2f} \\\\".format(
                    sequence_name.replace('_', '\\_'),
                    100 * frac_mean,
                    #frac_median
                    100 * frac_max
                ))


def main():
    data = {}
    for system_name, data_path in [
        ('LibVisO', 'results/SmallGeneratedPredictRealWorldExperiment/LibVisO/ks_table/self_to_sim.json'),
        ('ORBSLAM2 monocular', 'results/SmallGeneratedPredictRealWorldExperiment/ORBSLAM2 monocular/ks_table/self_to_sim.json'),
        ('ORBSLAM2 stereo', 'results/SmallGeneratedPredictRealWorldExperiment/ORBSLAM2 stereo/ks_table/self_to_sim.json'),
        #('ORBSLAM2 rgbd', 'results/SmallGeneratedPredictRealWorldExperiment/ORBSLAM2 rgbd/ks_table/self_to_sim.json')
    ]:
        with open(data_path, 'r') as fp:
            data[system_name] = json.load(fp)

    print('real to real translation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        real_to_real_distance(system_name, data[system_name]['translation error'])
    print('')
    print('real to real orientation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        real_to_real_distance(system_name, data[system_name]['orientation error'])
    print('')
    print('sim to real translation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        sim_to_real_distance(system_name, data[system_name]['translation error'])
    print('')
    print('sim to real orientation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        sim_to_real_distance(system_name, data[system_name]['orientation error'])
    print('')
    print('min to real translation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        min_to_real_distance(system_name, data[system_name]['translation error'])
    print('')
    print('min to real orientation error')
    for system_name in data.keys():
        #print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        min_to_real_distance(system_name, data[system_name]['orientation error'])
    print('')
    print('self to self translation error')
    for system_name in data.keys():
        # print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        self_to_self_distance(system_name, data[system_name]['translation error'])
    print('')
    print('self to self orientation error')
    for system_name in data.keys():
        # print("\\multicolumn{{5}}{{l}}{{\\textbf{{{0}}}}}\\\\".format(system_name))
        self_to_self_distance(system_name, data[system_name]['orientation error'])

    print('')
    print('translation fraction closer than self')
    for system_name in data.keys():
        analyse_error(system_name, data[system_name]['translation error'])
    print('')
    print('orientation fraction closer than self')
    for system_name in data.keys():
        analyse_error(system_name, data[system_name]['translation error'])

    print('')
    for system_name in data.keys():
        meta_ks_stat(system_name, data[system_name]['translation error'])
        meta_ks_stat(system_name, data[system_name]['orientation error'])

if __name__ == '__main__':
    main()
