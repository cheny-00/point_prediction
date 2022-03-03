
import functools
import os, shutil

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def randn_sampler(split_rate,
                  n_dataset,
                  shuffle_dataset,
                  random_seed):
    # refer to https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    indices = list(range(n_dataset))
    if type(split_rate) == int:
        split_rate = list(split_rate)
    split_point = [0]
    cum_rate = 0
    for rate in split_rate:
        cum_rate += rate
        split_point.append(int(np.floor(cum_rate * n_dataset)))
    split_point.append(n_dataset)
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    samplers = list()
    for i in range(len(split_point) - 1):
        samplers.append(SubsetRandomSampler(indices[split_point[i]:split_point[i + 1]]))

    return samplers if len(samplers) > 1 else samplers[0]
