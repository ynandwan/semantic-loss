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
parser.add_argument('-output_dir',default='../logs',type=str)
parser.add_argument('-output_file',default='all_jobs.sh',type=str)
args = parser.parse_args(sys.argv[1:])


slurm_cmd = 'python semantic.py --output_dir {}'.format(args.output_dir) 

num_labeled = [1000]
batch_size = [10,16]
std = [0.2, 0.3, 0.4]
wt = [2,1,0.1, 0.01,0.005]
lr = [1e-4]
num_iter = [25000]

all_params = [num_labeled,  batch_size, std, wt, lr,num_iter]
names = ['num_labeled','batch_size','std','wt','lr','num_iter']

all_jobs = list(itertools.product(*all_params))
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n.replace('_','.'), str(s)) for n, s in name_setting.items()])
    jobs_list[log_str] = setting_str

print('Running %d jobs' % (len(jobs_list)))
fh = open(args.output_file,'w')
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
count = 0
for log_str, setting_str in jobs_list.items():
    count += 1
    full_cmd = '{} {}'.format(slurm_cmd, setting_str)
    if count % 2 == 0:
        bash_cmd = full_cmd 
    else:
        bash_cmd = full_cmd
    print(bash_cmd,file=fh)

fh.close()

os.chmod(args.output_file,mode)
