from __future__ import print_function

import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random
import stat

parser = argparse.ArgumentParser()
parser.add_argument('-save_model',default=0,type=int)
args = parser.parse_args(sys.argv[1:])


slurm_cmd = '!python semantic.py' 

num_labeled = [100]
batch_size = [10]
std = [0.1, 0.2, 0.3, 0.4, 0.5]
wt = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
lr = [1e-4]


all_params = [num_labeled,  batch_size, std, wt, lr]
names = ['num_labeled','batch_size','std','wt','lr']

all_jobs = list(itertools.product(*all_params))
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n.replace('_','.'), str(s)) for n, s in name_setting.items()])
    jobs_list[log_str] = setting_str

print('Running %d jobs' % (len(jobs_list)))
count = 0
for log_str, setting_str in jobs_list.items():
    count += 1
    full_cmd = '{} {}'.format(slurm_cmd, setting_str)
    if count % 4 == 0:
        bash_cmd = full_cmd 
    else:
        bash_cmd = full_cmd + '&'
    print(bash_cmd)
