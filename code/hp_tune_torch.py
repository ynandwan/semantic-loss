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
parser.add_argument('-max_jobs',default=1,type=int)
args = parser.parse_args(sys.argv[1:])

slurm_cmd = 'python semantic_pytorch.py' 


gpu = [1]
num_labeled = [100]
batch_size = [10]
std = [0.1, 0.2, 0.3, 0.4, 0.5]
wt = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
lr = [1e-3, 1e-5]


all_params = [num_labeled,  batch_size, std, wt, lr, gpu]
names = ['num_labeled','batch_size','std','wt','lr', 'gpu']

all_jobs = list(itertools.product(*all_params))
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n.replace('_','.'), str(s)) for n, s in name_setting.items()])
    jobs_list[log_str] = setting_str


fh = open('torch_jobs1.sh','w')
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP

print('Running %d jobs' % (len(jobs_list)))
count = 0
max_jobs = args.max_jobs
for log_str, setting_str in jobs_list.items():
    full_cmd = '{} {}&'.format(slurm_cmd, setting_str)
    print(full_cmd,file =fh)
    print("pids[{}]=$!".format(count%max_jobs),file = fh)
    if count % max_jobs == (max_jobs -1):
        print("echo 'job count is {}'".format(count), file=fh)
        print("for pid in ${pids[*]}; do",file = fh)
        print("\twait $pid",file = fh)
        print("done",file = fh)
    #
    count += 1

print("for pid in ${pids[*]}; do",file = fh)
print("\twait $pid",file = fh)
print("done",file = fh)
os.chmod('torch_jobs1.sh',mode)


